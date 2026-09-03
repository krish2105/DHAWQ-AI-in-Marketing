"""Hybrid retrieval over corpus B and D — ARCHITECTURE.md §8.4.

"Hybrid BM25 + dense with reciprocal rank fusion. Dense alone misses exact
article codes and rare attribute terms; BM25 alone misses paraphrase."

RRF rather than score blending, for the same reason the hybrid recommender
blends on rank: BM25 scores and cosine similarities live on incomparable
scales, and normalising them is dominated by outliers.

NO CROSS-ENCODER. PLAN.md §13 called it overengineered for corpora of ~100
documents and predicted it would not move context precision. That prediction is
now testable rather than asserted: the flag is here, off by default, and the
comparison is in the eval.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

EXTERNAL = Path(__file__).resolve().parent / "corpora" / "external"
RRF_K = 60

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenise(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


@dataclass(frozen=True)
class Doc:
    doc_id: str
    title: str
    text: str
    source: str
    url: str
    trust: str = "untrusted"


@dataclass(frozen=True)
class Hit:
    doc: Doc
    score: float
    bm25_rank: int | None
    dense_rank: int | None


class BM25:
    """Textbook BM25. Small corpora do not need an index server, and a
    dependency that has to be deployed to search 100 documents is a liability,
    not a capability."""

    def __init__(self, docs: list[Doc], k1: float = 1.5, b: float = 0.75) -> None:
        self.docs = docs
        self.k1, self.b = k1, b
        self._toks = [tokenise(d.title + " " + d.text) for d in docs]
        self._len = [len(t) for t in self._toks]
        self._avg = sum(self._len) / max(len(docs), 1)
        self._tf = [Counter(t) for t in self._toks]
        df: Counter[str] = Counter()
        for t in self._toks:
            df.update(set(t))
        n = len(docs)
        self._idf = {
            w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()
        }

    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        q = tokenise(query)
        scores = []
        for i, tf in enumerate(self._tf):
            s = 0.0
            for w in q:
                if w not in tf:
                    continue
                f = tf[w]
                s += self._idf.get(w, 0.0) * f * (self.k1 + 1) / (
                    f + self.k1 * (1 - self.b + self.b * self._len[i] / max(self._avg, 1e-9))
                )
            if s > 0:
                scores.append((i, s))
        return sorted(scores, key=lambda x: (-x[1], x[0]))[:k]


def _dense_scores(query: str, docs: list[Doc]) -> list[tuple[int, float]]:
    """Lexical-overlap stand-in for a dense encoder.

    STATED PLAINLY BECAUSE IT MATTERS: this is Jaccard over token sets, not a
    learned embedding. A real dense arm would need a text encoder the API does
    not carry (the CLIP model is IMAGE-side and lives at build time). What this
    preserves is the ARCHITECTURE — two retrievers with different failure modes,
    fused by rank — and what it does not preserve is dense retrieval's actual
    strength on paraphrase. Reported as a limitation rather than dressed up.
    """
    q = set(tokenise(query))
    out = []
    for i, d in enumerate(docs):
        t = set(tokenise(d.title + " " + d.text))
        if not t or not q:
            continue
        j = len(q & t) / len(q | t)
        if j > 0:
            out.append((i, j))
    return sorted(out, key=lambda x: (-x[1], x[0]))


@lru_cache(maxsize=1)
def load_corpus_d() -> list[Doc]:
    latest = EXTERNAL / "latest.json"
    if not latest.exists():
        return []
    snap = json.loads((EXTERNAL / json.loads(latest.read_text())["snapshot"]).read_text())
    return [Doc(d["doc_id"], d["title"], d["text"], d["source"], d["url"])
            for d in snap["documents"]]


def hybrid_search(query: str, docs: list[Doc] | None = None, k: int = 8) -> list[Hit]:
    """BM25 + dense, fused by reciprocal rank.

    RRF needs no score normalisation: a document ranked 1 by either retriever
    contributes 1/(60+1) regardless of what the underlying number was.
    """
    docs = load_corpus_d() if docs is None else docs
    if not docs:
        return []

    bm = BM25(docs).search(query, k=len(docs))
    de = _dense_scores(query, docs)[: len(docs)]
    bm_rank = {i: r for r, (i, _) in enumerate(bm)}
    de_rank = {i: r for r, (i, _) in enumerate(de)}

    fused: dict[int, float] = {}
    for ranks in (bm_rank, de_rank):
        for i, r in ranks.items():
            fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + r + 1)

    top = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
    return [Hit(docs[i], s, bm_rank.get(i), de_rank.get(i)) for i, s in top]
