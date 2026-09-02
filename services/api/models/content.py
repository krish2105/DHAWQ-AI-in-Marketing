"""Content-based — CLIP kNN.

ARCHITECTURE.md §6: solves cold start for new articles, because it needs no
interaction history at all. This is the arm that can score the 348 articles
(9.9% of test rows) that the collaborative arm structurally cannot.

Embeddings are already L2-normalised at D2, so cosine similarity is a matrix
multiply and nothing here re-normalises.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from services.api.models.base import Recommender

RECENCY_HALF_LIFE = 28.0


class ContentKNN(Recommender):
    name = "content"
    handles_cold_articles = True

    def __init__(self, half_life_days: float = RECENCY_HALF_LIFE, seed: int = 20260903):
        super().__init__(seed)
        self.half_life_days = half_life_days

    def fit(self, train: pl.DataFrame) -> "ContentKNN":
        from services.api.core.artifacts import embeddings

        self._prepare(train)
        self._emb = embeddings()                       # (n, d), unit-norm

        # A customer profile is the recency-weighted mean of what they bought,
        # renormalised. Weighting matters: an unweighted mean lets a purchase
        # from ten weeks ago pull as hard as yesterday's, which for fashion —
        # where taste and season both move — is simply wrong.
        t_end = train.get_column("t_dat").max()
        weighted = (
            train.with_columns(
                (0.5 ** ((pl.lit(t_end) - pl.col("t_dat")).dt.total_days()
                         / self.half_life_days)).alias("w")
            )
            .select("customer_id", "article_id", "w")
        )

        self._profiles: dict[str, np.ndarray] = {}
        grouped = weighted.group_by("customer_id").agg(
            pl.col("article_id"), pl.col("w")
        )
        for cid, arts, ws in zip(grouped.get_column("customer_id"),
                                 grouped.get_column("article_id"),
                                 grouped.get_column("w")):
            rows = [(self._idx[a], w) for a, w in zip(arts, ws) if a in self._idx]
            if not rows:
                continue
            idxs = np.fromiter((r[0] for r in rows), dtype=np.int32, count=len(rows))
            wts = np.fromiter((r[1] for r in rows), dtype=np.float32, count=len(rows))
            v = (self._emb[idxs] * wts[:, None]).sum(axis=0)
            n = np.linalg.norm(v)
            if n > 0:
                self._profiles[cid] = (v / n).astype(np.float32)

        self._fitted = True
        return self

    def score_customer(self, customer_id: str) -> np.ndarray:
        self._check_fitted()
        prof = self._profiles.get(customer_id)
        if prof is None:
            # No profile at all. Return -inf rather than zeros: "cannot score"
            # is not "scores everything equally badly", and the evaluation
            # stratifies on the difference.
            return np.full(self._n, -np.inf, dtype=np.float32)
        return self._emb @ prof

    def similar_items(self, article_id: str, k: int = 10) -> list[tuple[str, float]]:
        self._check_fitted()
        i = self._idx.get(article_id)
        if i is None:
            return []
        sims = self._emb @ self._emb[i]
        sims[i] = -np.inf
        top = np.argsort(-sims)[:k]
        return [(self._ids[j], float(sims[j])) for j in top]
