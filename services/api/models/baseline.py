"""Popularity + recency — the business-as-usual baseline.

ARCHITECTURE.md §6: "Every claim is measured against this, not against
nothing." This is the arm the whole marketing claim is relative to, so it has
to be a FAIR baseline, not a straw man. A merchandiser showing bestsellers to
everyone applies recency — last week's bestseller, not last year's — so the
baseline does too. Beating a deliberately weak baseline proves nothing.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from services.api.models.base import Recommender

HALF_LIFE_DAYS = 21.0


class PopularityRecency(Recommender):
    name = "popularity"
    can_score_cold_articles = True   # scores anything with training support

    def __init__(self, half_life_days: float = HALF_LIFE_DAYS, seed: int = 20260903):
        super().__init__(seed)
        self.half_life_days = half_life_days

    def fit(self, train: pl.DataFrame) -> "PopularityRecency":
        self._prepare(train)
        t_end = train.get_column("t_dat").max()

        # Exponential recency decay. A purchase 21 days before the split counts
        # half as much as one on the final day.
        decayed = (
            train.with_columns(
                (pl.lit(1.0) * (0.5 ** (
                    (pl.lit(t_end) - pl.col("t_dat")).dt.total_days()
                    / self.half_life_days
                ))).alias("w")
            )
            .group_by("article_id")
            .agg(pl.col("w").sum().alias("score"))
        )

        self._scores = np.zeros(self._n, dtype=np.float32)
        for a, s in zip(decayed.get_column("article_id"), decayed.get_column("score")):
            if (i := self._idx.get(a)) is not None:
                self._scores[i] = s

        self._fitted = True
        return self

    def score_customer(self, customer_id: str) -> np.ndarray:
        """Identical for every customer — that is the point. This arm is what
        'no personalisation' looks like."""
        self._check_fitted()
        return self._scores

    def similar_items(self, article_id: str, k: int = 10) -> list[tuple[str, float]]:
        order = np.argsort(-self._scores)
        return [(self._ids[i], float(self._scores[i]))
                for i in order if self._ids[i] != article_id][:k]
