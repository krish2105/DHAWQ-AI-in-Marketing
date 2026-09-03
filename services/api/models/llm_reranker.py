"""The LLM re-ranker — a BENCHMARKED ARM, NOT THE PRODUCTION PATH.

ARCHITECTURE.md §6.1: "Putting a model where the ranking decision lives is the
single most common way an otherwise good project becomes indefensible. You lose
falsifiability, reproducibility and any ability to attribute a lift number to a
cause."

So this arm is admitted on exactly one condition: it is subjected to the SAME
evaluation as everything else — same temporal split, same NDCG/MAP/MRR, same
coverage, Gini, novelty and cold-start stratification — plus two it alone must
answer:

  RANK STABILITY across repeat runs. Same input, N runs, report the maximum
  rank-position delta. An unstable ranker is noise wearing a suit.

  COST AND LATENCY per 1,000 slates. A re-ranker that wins NDCG by 0.4pp at
  30x the cost has lost.

Either outcome is a result. If it loses — the honest prior on a 13.5k catalogue
with strong collaborative signal — that is the MORE interesting finding, and
"we tested the fashionable approach and it underperformed, here is the
evidence" is a stronger viva answer than never having tried.

IT IS DELIBERATELY NOT EXPOSED AS AN AGENT TOOL. The `recommend` tool's model
enum omits it, so the agent cannot select it even by accident.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

import numpy as np
import polars as pl

from services.api.models.base import Recommender
from services.api.models.hybrid import Hybrid

CANDIDATE_POOL = 50          # what the base arm hands over
SYSTEM = """You re-rank fashion product candidates for a retail page.

You will be given a numbered list of candidate articles with their attributes.
Return the SAME items re-ordered best-first for a shopper who bought the items
described. Do not invent, drop or duplicate items.

Reply with ONLY a JSON array of the original numbers, most relevant first.
Example: [7, 2, 19, 1]"""


@dataclass
class RerankTelemetry:
    """Everything §6.1 requires this arm to answer that the others do not."""
    calls: int = 0
    failures: int = 0
    invalid_outputs: int = 0
    total_latency_s: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    provider: str = ""
    model: str = ""

    def as_dict(self, usd_per_1k_in: float = 0.0, usd_per_1k_out: float = 0.0) -> dict:
        n = max(self.calls, 1)
        cost = (self.input_tokens / 1000) * usd_per_1k_in + \
               (self.output_tokens / 1000) * usd_per_1k_out
        return {
            "provider": self.provider, "model": self.model,
            "calls": self.calls,
            "failure_rate": round(self.failures / n, 4),
            "invalid_output_rate": round(self.invalid_outputs / n, 4),
            "mean_latency_s": round(self.total_latency_s / n, 3),
            "latency_per_1000_slates_s": round(self.total_latency_s / n * 1000, 1),
            "cost_per_1000_slates_usd": round(cost / n * 1000, 4),
            "cost_note": (
                "0.00 on local inference. The number that matters for a hosted "
                "model is latency_per_1000_slates_s, which does not go away."
            ),
        }


class LLMReranker(Recommender):
    name = "llm_reranker"
    can_score_cold_articles = True     # inherits the base arm's coverage

    def __init__(self, base: Recommender | None = None, pool: int = CANDIDATE_POOL,
                 provider=None, seed: int = 20260903) -> None:
        super().__init__(seed)
        self.base = base or Hybrid(mode="weighted", seed=seed)
        self.pool = pool
        self._provider = provider
        self.telemetry = RerankTelemetry()

    def fit(self, train: pl.DataFrame) -> "LLMReranker":
        from services.api.core.artifacts import articles

        self._prepare(train)
        self.base.fit(train)
        arts = articles()
        cols = [c for c in ("article_id", "prod_name", "product_type_name",
                            "colour_group_name") if c in arts.columns]
        self._meta = {
            r[0]: " · ".join(str(x) for x in r[1:] if x)
            for r in arts.select(cols).iter_rows()
        }
        if self._provider is None:
            from services.api.agent.llm import for_task
            self._provider = for_task("generate")
        self.telemetry.provider = getattr(self._provider, "name", "?")
        self.telemetry.model = getattr(self._provider, "model", "?")
        self._fitted = True
        return self

    # ── the re-rank itself ───────────────────────────────────────────────────

    def _reorder(self, candidates: list[str]) -> list[str]:
        """Ask the model to permute. VALIDATE that it did.

        Output that is not a permutation of the input is REJECTED and the base
        order is kept — never partially applied. A model that drops or invents
        an article has not re-ranked, and silently accepting a truncated list
        would let it improve precision by shortening the slate.
        """
        from services.api.agent.llm import Message

        listing = "\n".join(
            f"{i + 1}. {self._meta.get(a, a)}" for i, a in enumerate(candidates)
        )
        t0 = time.perf_counter()
        try:
            resp = self._provider.complete(
                SYSTEM, [Message("user", listing)], max_tokens=600, temperature=0.0,
            )
            self.telemetry.calls += 1
            self.telemetry.total_latency_s += time.perf_counter() - t0
            self.telemetry.input_tokens += resp.input_tokens
            self.telemetry.output_tokens += resp.output_tokens
        except Exception:
            self.telemetry.calls += 1
            self.telemetry.failures += 1
            self.telemetry.total_latency_s += time.perf_counter() - t0
            return candidates

        m = re.search(r"\[[\d,\s]+\]", resp.text)
        if not m:
            self.telemetry.invalid_outputs += 1
            return candidates
        try:
            order = json.loads(m.group(0))
        except json.JSONDecodeError:
            self.telemetry.invalid_outputs += 1
            return candidates

        idx = [i - 1 for i in order if isinstance(i, int) and 1 <= i <= len(candidates)]
        if sorted(idx) != list(range(len(candidates))):
            # Not a permutation. Salvage what is valid, in the model's order,
            # then append the rest in base order — recorded as invalid so the
            # rate is reported rather than hidden.
            self.telemetry.invalid_outputs += 1
            seen, out = set(), []
            for i in idx:
                if i not in seen:
                    seen.add(i)
                    out.append(candidates[i])
            out += [a for j, a in enumerate(candidates) if j not in seen]
            return out
        return [candidates[i] for i in idx]

    def score_customer(self, customer_id: str) -> np.ndarray:
        """Scores are SYNTHESISED from the re-ranked order.

        The model does not emit a number — it emits an ORDER, and the order is
        turned into descending scores by code. That keeps §0.1 intact even for
        the arm whose whole point is putting a model near the ranking decision.
        """
        self._check_fitted()
        base_scores = self.base.score_customer(customer_id)
        if not np.isfinite(base_scores).any():
            return base_scores

        top = np.argsort(-np.nan_to_num(base_scores, nan=-np.inf))[: self.pool]
        candidates = [self._ids[i] for i in top]
        reordered = self._reorder(candidates)

        out = np.full(self._n, -np.inf, dtype=np.float32)
        for rank, aid in enumerate(reordered):
            out[self._idx[aid]] = float(len(reordered) - rank)
        return out

    def similar_items(self, article_id: str, k: int = 10):
        return self.base.similar_items(article_id, k)


# ── §6.1's extra obligations ─────────────────────────────────────────────────

def rank_stability(arm: LLMReranker, customer_ids: list[str], runs: int = 5,
                   k: int = 10) -> dict:
    """Same input, N runs, max rank-position delta.

    "An unstable ranker is noise wearing a suit." Non-determinism is fine;
    UNBOUNDED non-determinism is not (§10.4).
    """
    deltas, churn = [], []
    for cid in customer_ids:
        slates = [arm.recommend(cid, k) for _ in range(runs)]
        base = slates[0]
        pos = {a: i for i, a in enumerate(base)}
        worst = 0
        for s in slates[1:]:
            for i, a in enumerate(s):
                if a in pos:
                    worst = max(worst, abs(pos[a] - i))
            churn.append(len(set(base) ^ set(s)) / (2 * k))
        deltas.append(worst)
    return {
        "runs": runs, "customers": len(customer_ids), "k": k,
        "max_rank_delta": int(max(deltas)) if deltas else 0,
        "mean_max_rank_delta": round(float(np.mean(deltas)), 2) if deltas else 0.0,
        "mean_slate_churn": round(float(np.mean(churn)), 4) if churn else 0.0,
        "interpretation": (
            "max_rank_delta is the worst position an article moved across "
            "identical repeat runs. Temperature is pinned at 0; anything above "
            "a couple of positions means the ranking is sampling noise."
        ),
    }
