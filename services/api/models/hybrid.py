"""Hybrid — the arm the primary research question is about.

ARCHITECTURE.md §6 offers two shapes and §17 names the falsification condition:
if the hybrid fails to beat both content-only and collaborative-only under the
temporal split, the blend adds complexity for nothing and the simpler model is
the right engineering call. Both shapes are implemented so that is testable
rather than assumed.

WEIGHTED   rank-normalised blend of both arms.
CASCADE    collaborative where the customer has enough history, content where
           they do not — a routing rule, not a blend.

RANK normalisation, not min-max on raw scores. ALS scores and cosine
similarities live on incomparable scales, and min-max is dominated by
outliers, so a single extreme ALS score would silently mute the content arm
for that customer. Rank is scale-free and outlier-proof.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import polars as pl

from services.api.models.base import Recommender
from services.api.models.collaborative import ImplicitALS
from services.api.models.content import ContentKNN

Mode = Literal["weighted", "cascade"]
CASCADE_MIN_HISTORY = 3


def _rank_normalise(scores: np.ndarray) -> np.ndarray:
    """Map finite scores to [0, 1] by rank. -inf stays -inf."""
    out = np.full_like(scores, -np.inf, dtype=np.float32)
    finite = np.isfinite(scores)
    n = int(finite.sum())
    if n == 0:
        return out
    idx = np.flatnonzero(finite)
    order = idx[np.argsort(-scores[idx], kind="stable")]
    out[order] = np.linspace(1.0, 0.0, n, dtype=np.float32)
    return out


class Hybrid(Recommender):
    #: True because the CONTENT arm covers what collaborative cannot. This is
    #: the concrete reason the hybrid exists rather than a preference for
    #: ensembles.
    handles_cold_articles = True

    def __init__(
        self,
        mode: Mode = "weighted",
        w_collab: float = 0.6,
        cascade_min_history: int = CASCADE_MIN_HISTORY,
        seed: int = 20260903,
    ) -> None:
        super().__init__(seed)
        self.mode = mode
        self.w_collab = w_collab
        self.cascade_min_history = cascade_min_history
        self.name = f"hybrid_{mode}"
        self.collab = ImplicitALS(seed=seed)
        self.content = ContentKNN(seed=seed)

    def fit(self, train: pl.DataFrame) -> "Hybrid":
        self._prepare(train)
        self.collab.fit(train)
        self.content.fit(train)
        self._history = (
            train.group_by("customer_id").agg(pl.col("t_dat").n_unique().alias("n"))
            .rows_by_key("customer_id", unique=True)
        )
        self._fitted = True
        return self

    def _depth(self, customer_id: str) -> int:
        row = self._history.get(customer_id)
        return int(row[0]) if row else 0

    def score_customer(self, customer_id: str) -> np.ndarray:
        self._check_fitted()
        cf = self.collab.score_customer(customer_id)
        cb = self.content.score_customer(customer_id)

        if self.mode == "cascade":
            # A routing rule: trust collaborative only where there is enough
            # history for it to mean anything.
            return cf if self._depth(customer_id) >= self.cascade_min_history else cb

        rf, rb = _rank_normalise(cf), _rank_normalise(cb)
        both = np.isfinite(rf) & np.isfinite(rb)
        out = np.full(self._n, -np.inf, dtype=np.float32)
        out[both] = self.w_collab * rf[both] + (1 - self.w_collab) * rb[both]
        # Where only content can score — the cold articles — take content
        # alone, scaled by its weight so it cannot outrank a blended item on
        # the strength of having fewer contributors.
        only_cb = np.isfinite(rb) & ~np.isfinite(rf)
        out[only_cb] = (1 - self.w_collab) * rb[only_cb]
        return out

    def similar_items(self, article_id: str, k: int = 10) -> list[tuple[str, float]]:
        self._check_fitted()
        merged: dict[str, float] = {}
        for arm, w in ((self.collab, self.w_collab), (self.content, 1 - self.w_collab)):
            for rank, (aid, _) in enumerate(arm.similar_items(article_id, k * 3)):
                merged[aid] = merged.get(aid, 0.0) + w / (60 + rank)   # RRF
        return sorted(merged.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
