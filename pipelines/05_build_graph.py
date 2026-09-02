#!/usr/bin/env python3
"""D6 — corpus A, the taxonomy + co-purchase + visual graph.

    python3 pipelines/05_build_graph.py

ARCHITECTURE.md §8.3: "This is the corpus that justifies the word 'advanced'.
Flat vector search answers 'what looks like this'. It cannot answer 'what else
sits under this subCategory that this cohort has bought before, excluding what
they already own' — that is a path query, and path queries are where top-k
retrieval quietly fails while still returning plausible-looking results."

Path-based reasoning is also what gives the "why this?" overlay something REAL
to render: the explanation is the traversed path, not a post-hoc narrative.

STORAGE — a deviation from PLAN.md §5, recorded deliberately.
PLAN specified Postgres with recursive CTEs, on the grounds that a graph
database buys nothing at this scale. That reasoning holds and extends one step
further: at ~13.5k nodes and ~200k edges the whole graph is a few MB, so it is
emitted as parquet edge lists and loaded into an in-memory adjacency by
services/api/rag/graph_index.py. Depth-3 traversal is then microseconds with no
database round-trip and no schema migration. The edge lists are relational and
would load into Postgres unchanged if the API ever becomes multi-process.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import polars as pl

from pipelines.common import DATA_PROCESSED, require, sha256_file, step, write_manifest

OUT = DATA_PROCESSED / "graph"

# Thresholds — PLAN.md §5. Every one is recorded in the manifest with the edge
# count it produced, so a reviewer can see that lift>=1.5 yielded N edges and
# judge whether it was tuned to flatter a metric.
CO_MIN_SUPPORT = 20
CO_MIN_LIFT = 1.5
CO_TOP_N = 30
VIS_MIN_COSINE = 0.82
VIS_K = 20
VIS_MUTUAL = True


def build_taxonomy_edges(articles: pl.DataFrame) -> pl.DataFrame:
    """is_a — the structural spine. Article -> type -> group -> index."""
    rows = []
    cols = [
        ("product_type_name", "ArticleType"),
        ("product_group_name", "SubCategory"),
        ("index_group_name", "MasterCategory"),
    ]
    for col, node_type in cols:
        if col not in articles.columns:
            continue
        pairs = articles.select("article_id", col).drop_nulls().unique()
        for a, v in zip(pairs.get_column("article_id"), pairs.get_column(col)):
            rows.append((a, f"{node_type}:{v}", "is_a", 1.0, "structural"))

    # type -> group -> index, so a depth-3 traversal can climb the hierarchy.
    if {"product_type_name", "product_group_name"} <= set(articles.columns):
        for t, g in articles.select("product_type_name", "product_group_name").drop_nulls().unique().rows():
            rows.append((f"ArticleType:{t}", f"SubCategory:{g}", "is_a", 1.0, "structural"))
    if {"product_group_name", "index_group_name"} <= set(articles.columns):
        for g, i in articles.select("product_group_name", "index_group_name").drop_nulls().unique().rows():
            rows.append((f"SubCategory:{g}", f"MasterCategory:{i}", "is_a", 1.0, "structural"))

    return pl.DataFrame(rows, schema=["src", "dst", "relation", "weight", "source"],
                        orient="row")


def build_attribute_edges(articles: pl.DataFrame) -> pl.DataFrame:
    """has_colour / has_gender from H&M's OWN fields.

    Native metadata, so source='native' and weight 1.0. The PREDICTED
    attributes (season, usage, fine colour) come from the D6 classifier and
    carry source='predicted' plus the classifier's confidence — a path running
    through a predicted edge is weaker evidence than one that does not, and the
    graph must say so (ARCHITECTURE.md §3).
    """
    rows = []
    for col, rel, node in (("colour_group_name", "has_colour", "Colour"),
                           ("index_group_name", "has_gender", "Gender")):
        if col not in articles.columns:
            continue
        pairs = articles.select("article_id", col).drop_nulls().unique()
        for a, v in zip(pairs.get_column("article_id"), pairs.get_column(col)):
            rows.append((a, f"{node}:{v}", rel, 1.0, "native"))
    return pl.DataFrame(rows, schema=["src", "dst", "relation", "weight", "source"],
                        orient="row")


def build_copurchase_edges(train: pl.DataFrame) -> pl.DataFrame:
    """co_purchased_with — thresholded by support AND lift.

    Support alone would connect every bestseller to every other bestseller:
    two articles bought together 500 times because each is bought 50,000 times
    is not a relationship, it is arithmetic. Lift divides that out.

    A basket is (customer, date) — line items sharing a shopping occasion.
    """
    baskets = (
        train.select("customer_id", "t_dat", "article_id").unique()
        .with_columns((pl.col("customer_id") + "|" + pl.col("t_dat").cast(pl.Utf8))
                      .alias("basket"))
        .select("basket", "article_id")
    )
    n_baskets = baskets.get_column("basket").n_unique()
    item_freq = baskets.group_by("article_id").agg(
        pl.col("basket").n_unique().alias("n")
    )

    pairs = (
        baskets.join(baskets, on="basket")
        .filter(pl.col("article_id") < pl.col("article_id_right"))
        .group_by("article_id", "article_id_right").len()
        .rename({"len": "co"})
        .filter(pl.col("co") >= CO_MIN_SUPPORT)
    )
    pairs = (
        pairs.join(item_freq, on="article_id")
        .join(item_freq.rename({"article_id": "article_id_right", "n": "n_right"}),
              on="article_id_right")
        .with_columns(
            (pl.col("co") * n_baskets / (pl.col("n") * pl.col("n_right"))).alias("lift")
        )
        .filter(pl.col("lift") >= CO_MIN_LIFT)
    )

    # Keep the strongest CO_TOP_N per article, in BOTH directions, so the
    # relation is symmetric in the stored edge list and traversal need not
    # special-case direction.
    both = pl.concat([
        pairs.select(pl.col("article_id").alias("src"),
                     pl.col("article_id_right").alias("dst"), "lift"),
        pairs.select(pl.col("article_id_right").alias("src"),
                     pl.col("article_id").alias("dst"), "lift"),
    ])
    return (
        both.sort("lift", descending=True)
        .group_by("src").head(CO_TOP_N)
        .with_columns(pl.lit("co_purchased_with").alias("relation"),
                      pl.col("lift").alias("weight"),
                      pl.lit("derived").alias("source"))
        .select("src", "dst", "relation", "weight", "source")
    )


def build_visual_edges(ids: list[str], emb: np.ndarray) -> pl.DataFrame:
    """visually_near — CLIP kNN, cosine-thresholded, MUTUAL kNN only.

    The mutual filter removes hub articles — plain black tees that appear in
    everyone's neighbour list and quietly dominate every traversal. It is a
    one-line filter that materially improves path quality: without it, a
    two-hop visual path from almost any garment lands on the same handful of
    basics.
    """
    n = len(ids)
    rows: list[tuple] = []
    neigh: dict[int, dict[int, float]] = {}

    # Apple's Accelerate BLAS sets FP status flags (divide-by-zero, overflow,
    # invalid) during the vectorised sgemm kernel even when every input and
    # every output is finite. Verified on this data: embeddings are all finite
    # and unit-norm, and the resulting similarities are finite in [0.057, 1.0]
    # with zero NaN. Suppressed with that evidence recorded rather than left
    # emitting three alarming warnings per block that mean nothing — and the
    # assertion below is what actually guarantees correctness.
    block = 512
    for start in range(0, n, block):
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            sims = emb[start:start + block] @ emb.T      # unit-norm already
        assert np.all(np.isfinite(sims)), "non-finite similarity — investigate"
        for local, i in enumerate(range(start, min(start + block, n))):
            s = sims[local].copy()
            s[i] = -np.inf
            top = np.argpartition(-s, VIS_K)[:VIS_K]
            neigh[i] = {int(j): float(s[j]) for j in top if s[j] >= VIS_MIN_COSINE}

    for i, cand in neigh.items():
        for j, w in cand.items():
            if VIS_MUTUAL and i not in neigh.get(j, {}):
                continue
            rows.append((ids[i], ids[j], "visually_near", w, "derived"))

    return pl.DataFrame(rows, schema=["src", "dst", "relation", "weight", "source"],
                        orient="row")


def build_substitute_edges(visual: pl.DataFrame, copurchase: pl.DataFrame,
                           articles: pl.DataFrame) -> pl.DataFrame:
    """substitutes_for = visually_near AND same product type AND NOT co-purchased.

    The merchandising-useful distinction: "looks alike but is NOT bought
    alongside" is a substitute; "bought alongside" is a complement. A slot
    optimiser that cannot tell them apart fills a page with eight versions of
    the same shirt and calls it a collection.
    """
    if visual.height == 0:
        return visual.clear()
    types = dict(zip(articles.get_column("article_id"),
                     articles.get_column("product_type_name")))
    co = set(zip(copurchase.get_column("src"), copurchase.get_column("dst"))) \
        if copurchase.height else set()

    keep = [
        (s, d, "substitutes_for", w, "derived")
        for s, d, w in zip(visual.get_column("src"), visual.get_column("dst"),
                           visual.get_column("weight"))
        if types.get(s) is not None and types.get(s) == types.get(d)
        and (s, d) not in co
    ]
    return pl.DataFrame(keep, schema=["src", "dst", "relation", "weight", "source"],
                        orient="row")


def main() -> int:
    from services.api.core.artifacts import articles as _articles

    OUT.mkdir(parents=True, exist_ok=True)
    print("D6 — building corpus A (taxonomy graph)")

    arts = pl.read_parquet(DATA_PROCESSED / "articles.parquet")
    train = pl.read_parquet(DATA_PROCESSED / "transactions_train.parquet")
    ids = __import__("json").loads(
        (DATA_PROCESSED / "embeddings" / "article_ids.json").read_text())
    emb = np.load(DATA_PROCESSED / "embeddings" / "clip_image.npy")

    frames = {}
    with step("is_a taxonomy"):
        frames["taxonomy"] = build_taxonomy_edges(arts)
    with step("has_colour / has_gender"):
        frames["attributes"] = build_attribute_edges(arts)
    with step(f"co_purchased_with (support>={CO_MIN_SUPPORT}, lift>={CO_MIN_LIFT})"):
        frames["copurchase"] = build_copurchase_edges(train)
    with step(f"visually_near (cos>={VIS_MIN_COSINE}, k={VIS_K}, mutual)"):
        frames["visual"] = build_visual_edges(ids, emb)
    with step("substitutes_for"):
        frames["substitutes"] = build_substitute_edges(
            frames["visual"], frames["copurchase"], arts)

    edges = pl.concat([f for f in frames.values() if f.height], how="vertical")
    require(edges.height > 0, "G1", "graph has no edges")

    nodes = pl.DataFrame({
        "node_id": sorted(set(edges.get_column("src")) | set(edges.get_column("dst")))
    }).with_columns(
        pl.when(pl.col("node_id").str.contains(":"))
        .then(pl.col("node_id").str.split(":").list.first())
        .otherwise(pl.lit("Article")).alias("node_type")
    )

    with step("write parquet"):
        e_path, n_path = OUT / "edges.parquet", OUT / "nodes.parquet"
        edges.write_parquet(e_path, compression="zstd")
        nodes.write_parquet(n_path, compression="zstd")

    by_rel = edges.group_by("relation").len().sort("len", descending=True)
    by_type = nodes.group_by("node_type").len().sort("len", descending=True)

    write_manifest("graph_v1", {
        "thresholds": {
            "co_min_support": CO_MIN_SUPPORT, "co_min_lift": CO_MIN_LIFT,
            "co_top_n": CO_TOP_N, "visual_min_cosine": VIS_MIN_COSINE,
            "visual_k": VIS_K, "visual_mutual_knn": VIS_MUTUAL,
        },
        "counts": {
            "nodes": nodes.height, "edges": edges.height,
            "by_relation": {r: int(n) for r, n in by_rel.rows()},
            "by_node_type": {t: int(n) for t, n in by_type.rows()},
        },
        "storage": {
            "format": "parquet edge list, loaded to an in-memory adjacency",
            "deviation_from_plan": (
                "PLAN.md §5 specified Postgres recursive CTEs. At ~13.5k nodes / "
                "this edge count the whole graph is a few MB, so an in-memory "
                "adjacency is faster, needs no migration, and the edge list is "
                "relational and loads into Postgres unchanged if ever needed."
            ),
        },
        "predicted_edges": {
            "present": False,
            "note": ("has_season / has_usage require the FPI attribute classifier. "
                     "Until it exists every article is season_unknown and "
                     "POL-AVL-03 admits everything — see corpus C POL-AVL-05."),
        },
        "outputs": {
            "edges.parquet": {"rows": edges.height, "sha256": sha256_file(e_path)},
            "nodes.parquet": {"rows": nodes.height, "sha256": sha256_file(n_path)},
        },
    })

    print(f"\n  nodes {nodes.height:,} · edges {edges.height:,}")
    for r, n in by_rel.rows():
        print(f"    {r:<22} {n:>9,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
