#!/usr/bin/env python3
"""D2a — CLIP image embeddings.

    python3 pipelines/02_embed.py

open_clip ViT-L-14 (laion2b_s32b_b82k) on Apple Silicon MPS. Free, local, no API.

WHY L/14 AND NOT B/32 (ARCHITECTURE.md §5)
------------------------------------------
Encoding is a ONE-TIME cost. B/32 would finish in ~3 minutes but produces
measurably weaker retrieval on fine-grained visual distinctions — and
fine-grained is the entire point when the catalogue is 13.5k garments that
differ by cut, texture and pattern. Twelve extra minutes, once, for better
embeddings across every downstream metric.

Embeddings are cached to .npy immediately. Never re-encode.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from PIL import Image

from pipelines.common import (
    DATA_PROCESSED,
    load_canonical_ids,
    require,
    sha256_file,
    step,
    write_manifest,
)
from pipelines.subsample import image_path

MODEL = "ViT-L-14"
PRETRAINED = "laion2b_s32b_b82k"
BATCH = 64
OUT_DIR = DATA_PROCESSED / "embeddings"


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main() -> int:
    import open_clip

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    print(f"D2a — CLIP {MODEL} / {PRETRAINED} on {device}")

    ids = load_canonical_ids()
    print(f"      {len(ids):,} articles in canonical order")

    with step(f"load {MODEL} (downloads ~1.7GB on first run)"):
        model, _, preprocess = open_clip.create_model_and_transforms(
            MODEL, pretrained=PRETRAINED
        )
        model = model.to(device).eval()

    dim = model.visual.output_dim
    out = np.zeros((len(ids), dim), dtype=np.float32)
    failed: list[str] = []

    t0 = time.perf_counter()
    batch_size = BATCH
    i = 0
    print(f"      encoding at batch {batch_size} ...")

    while i < len(ids):
        chunk = ids[i : i + batch_size]
        tensors, ok_rows = [], []
        for j, aid in enumerate(chunk):
            try:
                with Image.open(image_path(aid)) as im:
                    tensors.append(preprocess(im.convert("RGB")))
                ok_rows.append(i + j)
            except Exception:
                failed.append(aid)

        if tensors:
            try:
                with torch.no_grad():
                    batch = torch.stack(tensors).to(device)
                    feats = model.encode_image(batch)
                    # L2-normalise here, once. Every downstream consumer then
                    # gets cosine similarity from a plain dot product, and no
                    # one has to remember to normalise.
                    feats = feats / feats.norm(dim=-1, keepdim=True)
                out[ok_rows] = feats.float().cpu().numpy()
            except RuntimeError as exc:
                # ARCHITECTURE.md §5: on MPS OOM drop the batch size before
                # dropping to a weaker model. B/32 is the last resort, not the
                # first.
                if batch_size > 8 and ("out of memory" in str(exc).lower()
                                       or "MPS" in str(exc)):
                    batch_size //= 2
                    print(f"      MPS pressure — batch -> {batch_size}, retrying")
                    continue
                raise

        i += len(chunk)
        if i % (batch_size * 20) < batch_size or i >= len(ids):
            done = i / len(ids)
            el = time.perf_counter() - t0
            eta = el / max(done, 1e-9) - el
            print(f"      {i:>6,}/{len(ids):,}  {done:5.1%}  "
                  f"elapsed {el/60:.1f}m  eta {eta/60:.1f}m")

    elapsed = time.perf_counter() - t0

    if failed:
        print(f"      {len(failed)} images failed to encode")
        keep = np.array([a not in set(failed) for a in ids])
        out, ids = out[keep], [a for a in ids if a not in set(failed)]

    with step("verify embeddings"):
        norms = np.linalg.norm(out, axis=1)
        require(np.all(np.isfinite(out)), "E1", "non-finite values in embeddings")
        require(bool(np.allclose(norms, 1.0, atol=1e-3)), "E2",
                "embeddings are not unit-norm")
        require(out.shape[0] == len(ids), "E3", "row count != id count")

    with step("write .npy + ids"):
        emb_path = OUT_DIR / "clip_image.npy"
        ids_path = OUT_DIR / "article_ids.json"
        np.save(emb_path, out)
        ids_path.write_text(json.dumps(ids))

    write_manifest("embed_v1", {
        "model": {"arch": MODEL, "pretrained": PRETRAINED, "device": device,
                  "dim": int(dim), "normalised": True},
        "counts": {"articles": len(ids), "failed": len(failed)},
        "failed_article_ids": failed[:50],
        "timing": {"seconds": round(elapsed, 1),
                   "images_per_second": round(len(ids) / max(elapsed, 1e-9), 1),
                   "final_batch_size": batch_size},
        "canonical_order": "sorted(article_id) — see pipelines/common.canonical_article_order",
        "outputs": {
            "clip_image.npy": {"shape": list(out.shape), "dtype": "float32",
                               "sha256": sha256_file(emb_path)},
            "article_ids.json": {"n": len(ids), "sha256": sha256_file(ids_path)},
        },
    })

    print(f"\n  {out.shape[0]:,} x {out.shape[1]} embeddings in {elapsed/60:.1f} min "
          f"({len(ids)/max(elapsed,1e-9):.0f} img/s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
