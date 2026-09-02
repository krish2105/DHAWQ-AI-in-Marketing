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

from collections import defaultdict, deque
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
    def __init__(self, edges: pl.DataFrame) -> None:
        self._adj: dict[str, list[Hop]] = defaultdict(list)
        for s, d, r, w, src in edges.iter_rows():
            self._adj[s].append(Hop(s, d, r, float(w), src))
        for hops in self._adj.values():
            # Deterministic neighbour order: weight desc, then id. Traversal
            # results feed a stability metric (§10.4); dict insertion order
            # would make them depend on parquet row order.
            hops.sort(key=lambda h: (-h.weight, h.dst))
        self._adj = dict(self._adj)

    # ── primitives ───────────────────────────────────────────────────────────

    def neighbours(self, node: str, relations: set[str] | None = None) -> list[Hop]:
        hops = self._adj.get(node, [])
        return [h for h in hops if relations is None or h.relation in relations]

    def degree(self, node: str) -> int:
        return len(self._adj.get(node, []))

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
        cannot outrank a short strong one — which is the failure mode that
        makes naive graph expansion return confident nonsense at depth 3.
        """
        depth = max(1, min(depth, MAX_DEPTH))
        exclude = (exclude or set()) | {start}
        rels = set(relations) if relations else None

        found: dict[str, Path] = {}
        queue: deque[tuple[str, tuple[Hop, ...], float]] = deque([(start, (), 1.0)])
        seen: set[str] = {start}

        while queue:
            node, hops, score = queue.popleft()
            if len(hops) >= depth:
                continue
            for h in self.neighbours(node, rels):
                if h.dst in seen:
                    continue
                new_hops, new_score = hops + (h,), score * max(h.weight, 1e-6)
                is_target = (
                    ":" not in h.dst if target_type == "Article"
                    else h.dst.startswith(f"{target_type}:")
                )
                if is_target and h.dst not in exclude:
                    prev = found.get(h.dst)
                    if prev is None or new_score > prev.score:
                        found[h.dst] = Path(h.dst, new_hops, new_score)
                if len(new_hops) < depth:
                    seen.add(h.dst)
                    queue.append((h.dst, new_hops, new_score))

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
        excluding what they already own. Scores accumulate across seeds, so an
        article reachable from many of the cohort's purchases outranks one
        reachable from a single purchase strongly."""
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
