#!/usr/bin/env python3
"""D2b — UMAP projection to 3D for the embedding space.

ARCHITECTURE.md §5: "Fit once, cache the coordinates. Never recompute per
request — UMAP is not deterministic across runs and the space must stay stable
between sessions or the demo breaks."

That is why this is a build-time pipeline writing a static artefact, and why
the random seed is pinned. A user who selects a shirt, reloads, and finds the
catalogue rearranged has lost the one thing the 3D view is for: the shape of
the catalogue being a real, stable object.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from pipelines.common import DATA_PROCESSED, require, sha256_file, step, write_manifest

EMB_DIR = DATA_PROCESSED / "embeddings"
SEED = 20260903
N_NEIGHBORS = 25
MIN_DIST = 0.12
METRIC = "cosine"


def main() -> int:
    import umap

    print("D2b — UMAP 768d -> 3d")
    emb = np.load(EMB_DIR / "clip_image.npy")
    ids = json.loads((EMB_DIR / "article_ids.json").read_text())
    require(emb.shape[0] == len(ids), "U1", "embedding/id length mismatch")

    with step(f"fit UMAP (n={emb.shape[0]:,}, seed={SEED})"):
        reducer = umap.UMAP(
            n_components=3, n_neighbors=N_NEIGHBORS, min_dist=MIN_DIST,
            metric=METRIC, random_state=SEED,   # pinned — stability is the point
            verbose=False,
        )
        coords = reducer.fit_transform(emb).astype(np.float32)

    with step("centre and scale to a unit-ish cube"):
        # The scene expects coordinates around the origin at a predictable
        # scale, so the camera framing does not have to be retuned every time
        # the catalogue changes.
        coords -= coords.mean(axis=0)
        scale = np.percentile(np.abs(coords), 99)
        coords = (coords / max(scale, 1e-6)) * 50.0

    require(np.all(np.isfinite(coords)), "U2", "non-finite UMAP coordinates")

    with step("write positions.bin (Float32Array, canonical order)"):
        pos_path = EMB_DIR / "positions.bin"
        coords.tofile(pos_path)

    write_manifest("umap_v1", {
        "params": {"n_components": 3, "n_neighbors": N_NEIGHBORS,
                   "min_dist": MIN_DIST, "metric": METRIC, "seed": SEED},
        "counts": {"points": int(coords.shape[0])},
        "extent": {"min": coords.min(axis=0).tolist(),
                   "max": coords.max(axis=0).tolist()},
        "stability_note": (
            "UMAP is not deterministic across runs without a fixed seed. The "
            "seed is pinned and the coordinates are a frozen artefact; the "
            "scene never recomputes them."
        ),
        "outputs": {"positions.bin": {
            "dtype": "float32", "shape": list(coords.shape),
            "layout": "row-major xyz, index == canonical article index",
            "sha256": sha256_file(pos_path)}},
    })
    print(f"\n  {coords.shape[0]:,} points, extent "
          f"{coords.min():.1f} .. {coords.max():.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
