"""The recommender contract.

Five arms are benchmarked (ARCHITECTURE.md §6) and the comparison is only
meaningful if they are measured identically. That is enforced here rather than
by convention: every arm implements the same interface, the evaluation harness
calls only this interface, and no arm gets to define its own notion of a hit.

DETERMINISM IS PART OF THE CONTRACT. §10.4 measures stability across repeat
runs. Every arm must return the same slate for the same input, which means
fixed seeds and a fully specified tie-break — never relying on dict or set
iteration order.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import polars as pl


class Recommender(ABC):
    """A ranked-list producer over the frozen catalogue."""

    name: str = "abstract"
    #: False for arms that cannot score an article absent from training.
    #: The collaborative arm sets this False — 9.9% of test rows involve such
    #: articles (D1 manifest), and pretending otherwise would silently score
    #: them as misses rather than as out-of-support.
    handles_cold_articles: bool = True

    def __init__(self, seed: int = 20260903) -> None:
        self.seed = seed
        self._fitted = False

    # ── fitting ──────────────────────────────────────────────────────────────

    @abstractmethod
    def fit(self, train: pl.DataFrame) -> "Recommender":
        """Fit on TRAIN ONLY. Touching the test split here is the leak the
        temporal split exists to prevent."""

    def _prepare(self, train: pl.DataFrame) -> None:
        """Shared fit scaffolding: canonical order, index maps, seen-sets.

        Every arm calls this first so they all agree on what "article 4173"
        means. An arm that built its own ordering would produce slates that
        look fine and are silently mis-indexed.
        """
        from services.api.core.artifacts import article_index, canonical_ids

        self._ids = canonical_ids()
        self._idx = article_index()
        self._n = len(self._ids)

        seen = (
            train.select("customer_id", "article_id")
            .unique()
            .group_by("customer_id")
            .agg(pl.col("article_id"))
        )
        self._seen_idx = {
            c: {self._idx[a] for a in arts if a in self._idx}
            for c, arts in zip(seen.get_column("customer_id"),
                               seen.get_column("article_id"))
        }

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(f"{self.name}: call fit() before recommending")

    # ── scoring ──────────────────────────────────────────────────────────────

    @abstractmethod
    def score_customer(self, customer_id: str) -> np.ndarray:
        """Dense score vector over canonical article order.

        -inf marks 'cannot score' and is distinct from a low score. Collapsing
        the two is how a cold-start limitation gets reported as poor accuracy.
        """

    def recommend(
        self, customer_id: str, k: int = 10, exclude_seen: bool = True
    ) -> list[str]:
        self._check_fitted()
        scores = self.score_customer(customer_id).copy()
        if exclude_seen:
            for idx in self._seen_idx.get(customer_id, ()):
                scores[idx] = -np.inf
        return self._top_k(scores, k)

    def recommend_batch(
        self, customer_ids: list[str], k: int = 10, exclude_seen: bool = True
    ) -> dict[str, list[str]]:
        return {c: self.recommend(c, k, exclude_seen) for c in customer_ids}

    def _top_k(self, scores: np.ndarray, k: int) -> list[str]:
        """Deterministic top-k.

        np.argpartition is fast but its order within the partition is
        unspecified, so ties would resolve differently across runs and the
        §10.4 stability metric would measure library internals. The lexsort
        below breaks ties on ascending article index, which is the documented
        tie-break in POL-SLT-05.
        """
        valid = np.isfinite(scores)
        if not valid.any():
            return []
        idx = np.flatnonzero(valid)
        order = np.lexsort((idx, -scores[idx]))       # score desc, then index asc
        return [self._ids[i] for i in idx[order[:k]]]

    # ── item-to-item ─────────────────────────────────────────────────────────

    def similar_items(self, article_id: str, k: int = 10) -> list[tuple[str, float]]:
        """Item neighbours. Used by the product route and the 3D scene."""
        raise NotImplementedError(f"{self.name} does not expose item similarity")
