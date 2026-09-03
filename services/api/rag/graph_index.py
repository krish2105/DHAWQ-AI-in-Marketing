"""Corpus A query layer — GraphRAG over the taxonomy graph.

ARCHITECTURE.md §8.3. Answers PATH queries, which is the thing flat top-k
retrieval cannot do while still returning plausible-looking results:

    "what else sits under this subCategory that this cohort has bought before,
     excluding what they already own"

Loaded once into an in-memory adjacency. At this scale that is a few MB and a
depth-3 traversal is microseconds, with no database round-trip.

READ-ONLY. The agent's `graph_traverse` tool binds here and there is no write
path in this module — the §4 permission boundary is a property of the code, not
a promise in a document.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import polars as pl

GRAPH_DIR = Path(__file__).resolve().parents[3] / "data" / "processed" / "graph"

MAX_DEPTH = 3          # tool contract caps this; deeper is unbounded fan-out
MAX_RESULTS = 200


@dataclass(frozen=True)
class Hop:
    src: str
    dst: str
    relation: str
    weight: float
    source: str        # 'structural' | 'native' | 'derived' | 'predicted'


@dataclass(frozen=True)
class Path:
    """A traversal result. `hops` IS the explanation — the "why this?" overlay
    renders this, it does not narrate over it."""
    target: str
    hops: tuple[Hop, ...]
    score: float

    @property
    def relations(self) -> tuple[str, ...]:
        return tuple(h.relation for h in self.hops)

    @property
    def uses_predicted_evidence(self) -> bool:
        """A path through a predicted edge is weaker evidence than one that is
        not. POL-CLM-04 requires it be labelled when cited."""
        return any(h.source == "predicted" for h in self.hops)

    def describe(self) -> str:
        parts = [self.hops[0].src] if self.hops else []
        for h in self.hops:
            parts.append(f"--[{h.relation}]-->{h.dst}")
        return " ".join(parts)


class TaxonomyGraph:
    """Adjacency stored as FLAT ARRAYS, not Python objects.

    The first version built one frozen `Hop` dataclass per edge. For a 1.9MB
    parquet that cost 212MB resident — a 111x blowup — and combined with the
    catalogue and the embedding matrix it took the deployed instance past its
    512MB limit and got it OOM-killed. Measured on Render, not guessed: the
    instance sat at 51-90MB and spiked to 536MB on a request that touched the
    graph.

    A graph is integers. Node ids, relation names and provenance labels are
    interned once; the edges themselves are four numpy arrays plus a CSR offset
    table, which is ~4MB. `Hop` objects are still the traversal's currency, but
    they are now constructed ONLY for the handful of paths a query returns
    rather than for all 328,318 edges up front.
    """

    __slots__ = ("_nodes", "_node_idx", "_rels", "_srcs",
                 "_dst", "_rel", "_src", "_w", "_start", "_end")

    def __init__(self, edges: pl.DataFrame) -> None:
        import numpy as np

        # Intern every string exactly once.
        nodes = sorted(set(edges.get_column("src")) | set(edges.get_column("dst")))
        self._nodes: list[str] = nodes
        self._node_idx: dict[str, int] = {n: i for i, n in enumerate(nodes)}
        self._rels: list[str] = sorted(set(edges.get_column("relation")))
        rel_idx = {r: i for i, r in enumerate(self._rels)}
        self._srcs: list[str] = sorted(set(edges.get_column("source")))
        src_idx = {v: i for i, v in enumerate(self._srcs)}

        e = edges.with_columns(
            pl.col("src").replace_strict(self._node_idx).cast(pl.Int32).alias("s"),
            pl.col("dst").replace_strict(self._node_idx).cast(pl.Int32).alias("d"),
            pl.col("relation").replace_strict(rel_idx).cast(pl.Int16).alias("r"),
            pl.col("source").replace_strict(src_idx).cast(pl.Int8).alias("p"),
        ).sort(["s", "weight"], descending=[False, True])

        s_arr = e.get_column("s").to_numpy()
        self._dst = e.get_column("d").to_numpy().astype(np.int32, copy=False)
        self._rel = e.get_column("r").to_numpy().astype(np.int16, copy=False)
        self._src = e.get_column("p").to_numpy().astype(np.int8, copy=False)
        self._w = e.get_column("weight").to_numpy().astype(np.float32, copy=False)

        # CSR offsets. Neighbours of node i are the slice [start[i]:end[i]],
        # already ordered by weight desc — deterministic, because traversal
        # results feed a stability metric (§10.4) and must not depend on row
        # order in the parquet.
        n = len(nodes)
        self._start = np.searchsorted(s_arr, np.arange(n), side="left")
        self._end = np.searchsorted(s_arr, np.arange(n), side="right")

    # ── primitives ───────────────────────────────────────────────────────────

    def _hop(self, src_i: int, k: int) -> Hop:
        """Materialise one edge. Called for returned paths only."""
        return Hop(self._nodes[src_i], self._nodes[self._dst[k]],
                   self._rels[self._rel[k]], float(self._w[k]),
                   self._srcs[self._src[k]])

    def neighbours(self, node: str, relations: set[str] | None = None) -> list[Hop]:
        i = self._node_idx.get(node)
        if i is None:
            return []
        out = []
        for k in range(self._start[i], self._end[i]):
            if relations is not None and self._rels[self._rel[k]] not in relations:
                continue
            out.append(self._hop(i, k))
        return out

    def degree(self, node: str) -> int:
        i = self._node_idx.get(node)
        return 0 if i is None else int(self._end[i] - self._start[i])

    # ── the tool surface ─────────────────────────────────────────────────────

    def traverse(
        self,
        start: str,
        relations: list[str] | None = None,
        depth: int = 2,
        *,
        exclude: set[str] | None = None,
        target_type: str = "Article",
        limit: int = MAX_RESULTS,
    ) -> list[Path]:
        """Breadth-first path enumeration from `start`.

        Score is the PRODUCT of hop weights, so a long chain of weak edges
        cannot outrank a short strong one — the failure mode that makes naive
        graph expansion return confident nonsense at depth 3.
        """
        depth = max(1, min(depth, MAX_DEPTH))
        i0 = self._node_idx.get(start)
        if i0 is None:
            return []
        exclude = (exclude or set()) | {start}
        rel_ok = None if not relations else {
            r for r, name in enumerate(self._rels) if name in set(relations)
        }

        found: dict[str, Path] = {}
        queue: deque[tuple[int, tuple[Hop, ...], float]] = deque([(i0, (), 1.0)])
        seen: set[int] = {i0}

        while queue:
            node_i, hops, score = queue.popleft()
            if len(hops) >= depth:
                continue
            for k in range(self._start[node_i], self._end[node_i]):
                if rel_ok is not None and self._rel[k] not in rel_ok:
                    continue
                d = int(self._dst[k])
                if d in seen:
                    continue
                new_score = score * max(float(self._w[k]), 1e-6)
                name = self._nodes[d]
                is_target = (":" not in name if target_type == "Article"
                             else name.startswith(f"{target_type}:"))
                if is_target and name not in exclude:
                    prev = found.get(name)
                    if prev is None or new_score > prev.score:
                        found[name] = Path(name, hops + (self._hop(node_i, k),),
                                           new_score)
                if len(hops) + 1 < depth:
                    seen.add(d)
                    queue.append((d, hops + (self._hop(node_i, k),), new_score))

        return sorted(found.values(), key=lambda p: (-p.score, p.target))[:limit]

    def explain_pair(self, a: str, b: str, max_depth: int = 2) -> list[Path]:
        """Every short path between two articles. This is what the "why this?"
        overlay renders: the traversed path, not a post-hoc narrative."""
        return [p for p in self.traverse(a, depth=max_depth, limit=10_000)
                if p.target == b]

    def cohort_affinity(
        self, cohort_articles: list[str], relations: list[str] | None = None,
        depth: int = 2, exclude: set[str] | None = None, limit: int = 100,
    ) -> list[Path]:
        """The multi-hop query §8.3 names: expand from what a cohort bought,
        excluding what they already own."""
        exclude = (exclude or set()) | set(cohort_articles)
        agg: dict[str, Path] = {}
        for seed in cohort_articles:
            for p in self.traverse(seed, relations, depth, exclude=exclude):
                prev = agg.get(p.target)
                if prev is None:
                    agg[p.target] = p
                else:
                    agg[p.target] = Path(p.target,
                                         prev.hops if prev.score >= p.score else p.hops,
                                         prev.score + p.score)
        return sorted(agg.values(), key=lambda p: (-p.score, p.target))[:limit]


@lru_cache(maxsize=1)
def load_graph() -> TaxonomyGraph:
    path = GRAPH_DIR / "edges.parquet"
    if not path.exists():
        raise RuntimeError("graph missing — run `python3 pipelines/05_build_graph.py`")
    return TaxonomyGraph(pl.read_parquet(path))
