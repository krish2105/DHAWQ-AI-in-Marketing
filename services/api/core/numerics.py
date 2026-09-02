"""Numerical hygiene shared by the model layer.

Apple's Accelerate BLAS sets the divide-by-zero / overflow / invalid FP status
flags during its vectorised sgemm kernel even when every input and every output
is finite. Verified directly on this data: the CLIP embeddings are all finite
and unit-norm (asserted at D2), and the resulting similarity matrix is finite
in [0.057, 1.000] with zero NaN.

Suppressing the flags without checking would be hiding a real bug behind a
convenient excuse, so `safe_matmul` suppresses the spurious warning AND asserts
the result is finite. The assertion is what provides the guarantee; the
suppression only stops three meaningless warnings per block from burying real
ones.
"""

from __future__ import annotations

import numpy as np


def safe_matmul(a: np.ndarray, b: np.ndarray, *, where: str = "matmul") -> np.ndarray:
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        out = a @ b
    if not np.all(np.isfinite(out)):
        raise FloatingPointError(
            f"{where}: non-finite result — this is a real numerical fault, "
            "not the Accelerate status-flag false positive"
        )
    return out
