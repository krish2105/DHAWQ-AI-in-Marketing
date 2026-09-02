"""Collaborative filtering — implicit ALS.

ARCHITECTURE.md §6: captures "bought together" signal that is invisible to
images. Two garments that look nothing alike but sell together — a dress and
the jacket customers pair with it — are near in this space and far in CLIP's.
That complementarity is the entire argument for the hybrid.

The data is IMPLICIT feedback: purchases, never impressions (§3, §16). We
observe conversions only, so an unpurchased item is UNLABELLED, not negative.
ALS with confidence weighting is the right family for that; anything that
treats non-purchase as a negative label is mis-specified on this data.
"""

from __future__ import annotations

import os

import numpy as np
import polars as pl
import scipy.sparse as sp

from services.api.models.base import Recommender

# BLAS threads fight implicit's own threading and make results non-deterministic
# in wall-clock order. §10.4 measures stability; set this before importing.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

FACTORS = 128
REGULARIZATION = 0.05
ITERATIONS = 20
ALPHA = 40.0


class ImplicitALS(Recommender):
    name = "collaborative"
    #: THE structural limitation of this arm. An article with no training
    #: interaction has no latent factor — there is nothing to score. D1 found
    #: 348 such articles covering 9.9% of test rows. Declaring it here means
    #: the evaluation reports it as out-of-support rather than as a miss.
    handles_cold_articles = False

    def __init__(
        self,
        factors: int = FACTORS,
        regularization: float = REGULARIZATION,
        iterations: int = ITERATIONS,
        alpha: float = ALPHA,
        seed: int = 20260903,
    ) -> None:
        super().__init__(seed)
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.alpha = alpha

    def fit(self, train: pl.DataFrame) -> "ImplicitALS":
        from implicit.als import AlternatingLeastSquares

        self._prepare(train)

        # Customer axis is local to this arm; the article axis is canonical.
        cust = train.get_column("customer_id").unique().sort().to_list()
        self._cust_idx = {c: i for i, c in enumerate(cust)}
        self._custs = cust

        counts = (
            train.group_by("customer_id", "article_id").len()
            .rename({"len": "n"})
        )
        rows = np.fromiter(
            (self._cust_idx[c] for c in counts.get_column("customer_id")),
            dtype=np.int32, count=counts.height,
        )
        cols = np.fromiter(
            (self._idx.get(a, -1) for a in counts.get_column("article_id")),
            dtype=np.int32, count=counts.height,
        )
        vals = counts.get_column("n").to_numpy().astype(np.float32)
        keep = cols >= 0

        # Confidence weighting: c_ui = 1 + alpha * count. Repeat purchases are
        # stronger evidence of preference than a single one.
        self._matrix = sp.csr_matrix(
            (1.0 + self.alpha * vals[keep], (rows[keep], cols[keep])),
            shape=(len(cust), self._n),
            dtype=np.float32,
        )

        self._model = AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularization,
            iterations=self.iterations,
            random_state=self.seed,        # §10.4 stability
            use_gpu=False,
        )
        self._model.fit(self._matrix, show_progress=False)

        # Articles with no interaction get an all-zero factor from ALS, which
        # scores 0.0 — indistinguishable from a genuinely mediocre article.
        # Mask them so they read as -inf ("cannot score") instead.
        self._has_support = np.asarray(
            (self._matrix > 0).sum(axis=0)
        ).ravel() > 0

        self._fitted = True
        return self

    def score_customer(self, customer_id: str) -> np.ndarray:
        self._check_fitted()
        u = self._cust_idx.get(customer_id)
        if u is None:
            return np.full(self._n, -np.inf, dtype=np.float32)
        scores = (self._model.item_factors @ self._model.user_factors[u]).astype(np.float32)
        scores[~self._has_support] = -np.inf
        return scores

    def similar_items(self, article_id: str, k: int = 10) -> list[tuple[str, float]]:
        self._check_fitted()
        i = self._idx.get(article_id)
        if i is None or not self._has_support[i]:
            return []
        f = self._model.item_factors
        norms = np.linalg.norm(f, axis=1)
        norms[norms == 0] = 1.0
        sims = (f @ f[i]) / (norms * norms[i])
        sims[~self._has_support] = -np.inf
        sims[i] = -np.inf
        return [(self._ids[j], float(sims[j])) for j in np.argsort(-sims)[:k]]
