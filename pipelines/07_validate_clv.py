#!/usr/bin/env python3
"""D5b — CLV holdout validation (ARCHITECTURE.md §11).

    python3 pipelines/07_validate_clv.py

"BG/NBD for purchase frequency, Gamma-Gamma for monetary value. Holdout-
validated: fit on the first period, predict the second, plot predicted vs
actual."

Until now the CLV numbers were UNVALIDATED. The model fit, produced plausible
figures, and nothing had ever checked whether those figures corresponded to
anything. A projected CLV nobody has tested against observed behaviour is a
number with a confident shape and no evidence behind it.

THE SPLIT IS INSIDE THE TRAINING WINDOW, DELIBERATELY. The test split is
reserved for ranking evaluation; borrowing it here would spend held-out data
twice and let a CLV tuned against it leak into the §9 numbers. So the 10-week
train period splits again: 8 weeks to calibrate, 2 weeks to validate.

WHAT THIS CAN AND CANNOT SHOW. A 2-week holdout on a 12-week window is short
for a model whose whole premise is long-horizon behaviour. It can catch a model
that is badly wrong; it cannot confirm one that is subtly wrong. That limit is
reported next to the numbers, not buried.
"""

from __future__ import annotations

import json
import sys
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import polars as pl

from pipelines.common import DATA_PROCESSED, require, sha256_file, step, write_manifest

OUT = DATA_PROCESSED / "cohorts"
CALIBRATION_WEEKS = 8


def main() -> int:
    from services.api.core.artifacts import train
    from services.api.marketing.clv import (
        expected_purchases, fit_bgnbd, fit_gamma_gamma, expected_value, rfm_matrix,
    )

    print("D5b — CLV holdout validation")
    tr = train()
    t0 = tr.get_column("t_dat").min()
    cut = t0 + timedelta(days=CALIBRATION_WEEKS * 7)
    t_end = tr.get_column("t_dat").max()
    holdout_days = (t_end - cut).days
    print(f"      calibrate {t0} .. {cut}   holdout {cut} .. {t_end} "
          f"({holdout_days} days)")

    cal = tr.filter(pl.col("t_dat") < cut)
    hold = tr.filter(pl.col("t_dat") >= cut)
    require(cal.height > 0 and hold.height > 0, "V1", "empty calibration or holdout")

    with step("fit BG/NBD + Gamma-Gamma on the calibration period ONLY"):
        summary = rfm_matrix(cal, as_of=cut)
        bg = fit_bgnbd(summary)
        gg = fit_gamma_gamma(summary)

    with step("predict holdout purchases"):
        pred = expected_purchases(
            bg, summary.get_column("frequency"), summary.get_column("recency"),
            summary.get_column("T"), float(holdout_days),
        )
        pred_value = expected_value(gg, summary.get_column("frequency"),
                                    summary.get_column("monetary_value"))

    with step("observe what actually happened"):
        actual = (
            hold.group_by("customer_id")
            .agg(pl.col("t_dat").n_unique().alias("actual_occasions"),
                 pl.col("price").sum().alias("actual_spend"))
        )
        joined = (
            summary.select("customer_id", "frequency")
            .with_columns(pl.Series("predicted", pred),
                          pl.Series("predicted_value", pred_value))
            .join(actual, on="customer_id", how="left")
            .with_columns(pl.col("actual_occasions").fill_null(0),
                          pl.col("actual_spend").fill_null(0.0))
        )

    p = joined.get_column("predicted").to_numpy()
    a = joined.get_column("actual_occasions").to_numpy().astype(float)
    freq = joined.get_column("frequency").to_numpy()

    # THE PLOT §11 ASKS FOR: bucket customers by their CALIBRATION frequency,
    # then compare mean predicted against mean actual in each bucket. Bucketing
    # by the input rather than the prediction is what makes it a validation
    # rather than a restatement — it asks "for customers who looked like this,
    # was the model right?"
    buckets = []
    for lo, hi, label in [(0, 0, "0"), (1, 1, "1"), (2, 2, "2"), (3, 4, "3-4"),
                          (5, 7, "5-7"), (8, 10**9, "8+")]:
        m = (freq >= lo) & (freq <= hi)
        if m.sum() == 0:
            continue
        buckets.append({
            "calibration_frequency": label, "n": int(m.sum()),
            "mean_predicted": round(float(p[m].mean()), 4),
            "mean_actual": round(float(a[m].mean()), 4),
        })

    mae = float(np.mean(np.abs(p - a)))
    bias = float(np.mean(p - a))
    corr = float(np.corrcoef(p, a)[0, 1]) if p.std() > 0 and a.std() > 0 else 0.0
    # A model that cannot beat "everyone repeats their calibration rate" has
    # earned nothing. This is the naive baseline §10.5 asks every metric to have.
    naive = freq / max((cut - t0).days, 1) * holdout_days
    naive_mae = float(np.mean(np.abs(naive - a)))

    payload = {
        "protocol": {
            "calibration": {"from": str(t0), "to": str(cut),
                            "weeks": CALIBRATION_WEEKS},
            "holdout": {"from": str(cut), "to": str(t_end), "days": holdout_days},
            "n_customers": joined.height,
            "why_inside_train": (
                "The test split is reserved for ranking evaluation. Borrowing it "
                "here would spend held-out data twice and let a CLV tuned "
                "against it leak into the §9 numbers."
            ),
        },
        "params": {"bgnbd": bg.__dict__, "gamma_gamma": gg.__dict__},
        "accuracy": {
            "mae_purchases": round(mae, 4),
            "bias_purchases": round(bias, 4),
            "correlation": round(corr, 4),
            "naive_baseline_mae": round(naive_mae, 4),
            "beats_naive": bool(mae < naive_mae),
        },
        "buckets": buckets,
        "limitation": (
            f"A {holdout_days}-day holdout on a 12-week window is short for a "
            "model whose premise is long-horizon behaviour. It can catch a "
            "model that is badly wrong; it cannot confirm one that is subtly "
            "wrong. Read the bucket table as a sanity check, not a validation "
            "of long-run CLV."
        ),
    }

    path = OUT / "clv_holdout.json"
    path.write_text(json.dumps(payload, indent=2, default=str))
    write_manifest("clv_holdout_v1", {
        **payload["protocol"], "accuracy": payload["accuracy"],
        "outputs": {"clv_holdout.json": {"sha256": sha256_file(path)}},
    })

    print()
    print(f"  {'calibration freq':<20}{'n':>8}{'predicted':>12}{'actual':>10}")
    for b in buckets:
        print(f"  {b['calibration_frequency']:<20}{b['n']:>8,}"
              f"{b['mean_predicted']:>12.3f}{b['mean_actual']:>10.3f}")
    print()
    print(f"  MAE {mae:.4f} purchases · bias {bias:+.4f} · corr {corr:.3f}")
    print(f"  naive baseline MAE {naive_mae:.4f} — model "
          f"{'BEATS' if mae < naive_mae else 'DOES NOT BEAT'} it")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
