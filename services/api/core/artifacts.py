"""Read-only access to frozen pipeline artefacts.

THE PERMISSION BOUNDARY (ARCHITECTURE.md §4, PLAN.md §1).

services/api/ never imports pipelines/. Artefacts cross that line as frozen
files plus a manifest, and this module is the only door. Everything here reads;
nothing writes. If a function in this file ever gains a write path, the §4
diagram has stopped being true.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[3]
PROCESSED = REPO_ROOT / "data" / "processed"
MANIFESTS = REPO_ROOT / "pipelines" / "manifests"


class ArtifactMissing(RuntimeError):
    """A required frozen artefact is absent. Names the pipeline that makes it,
    because 'file not found' three layers down is a useless error."""


def _require(path: Path, pipeline: str) -> Path:
    if not path.exists():
        raise ArtifactMissing(f"{path.name} missing — run `python3 pipelines/{pipeline}`")
    return path


@lru_cache(maxsize=1)
def manifest(name: str = "subsample_v1") -> dict:
    return json.loads(_require(MANIFESTS / f"{name}.json", "01_subsample.py").read_text())


@lru_cache(maxsize=1)
def articles() -> pl.DataFrame:
    return pl.read_parquet(_require(PROCESSED / "articles.parquet", "01_subsample.py"))


@lru_cache(maxsize=1)
def customers() -> pl.DataFrame:
    return pl.read_parquet(_require(PROCESSED / "customers.parquet", "01_subsample.py"))


@lru_cache(maxsize=1)
def train() -> pl.DataFrame:
    return pl.read_parquet(_require(PROCESSED / "transactions_train.parquet", "01_subsample.py"))


@lru_cache(maxsize=1)
def test() -> pl.DataFrame:
    return pl.read_parquet(_require(PROCESSED / "transactions_test.parquet", "01_subsample.py"))


@lru_cache(maxsize=1)
def canonical_ids() -> list[str]:
    """The single instance index shared by embeddings, positions, colours and
    the atlas. PLAN.md §8."""
    p = PROCESSED / "embeddings" / "article_ids.json"
    if p.exists():
        return json.loads(p.read_text())
    return sorted(articles().get_column("article_id").to_list())


@lru_cache(maxsize=1)
def embeddings() -> np.ndarray:
    """L2-normalised CLIP image embeddings, row i == canonical_ids()[i].

    Already unit-norm from D2, so cosine similarity is a plain dot product.
    """
    return np.load(_require(PROCESSED / "embeddings" / "clip_image.npy", "02_embed.py"))


@lru_cache(maxsize=1)
def article_index() -> dict[str, int]:
    return {a: i for i, a in enumerate(canonical_ids())}


def reset_cache() -> None:
    """Tests only — artefacts are immutable in normal operation."""
    for fn in (manifest, articles, customers, train, test, canonical_ids,
               embeddings, article_index):
        fn.cache_clear()
