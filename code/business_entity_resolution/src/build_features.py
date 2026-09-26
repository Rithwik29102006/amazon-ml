"""Step 3: compute pair features for all candidate pairs of a split.

Usage: python build_features.py [train|test ...]
Writes WORK_DIR/feat/<split>/part_XXX.parquet
"""
import shutil
import sys
import time

import polars as pl

import config
from features import load_norm, pair_features

CHUNK = 2_000_000


def build(split):
    t = time.time()
    cand = pl.read_parquet(config.cand_path(split))
    s1, q = load_norm(split)
    s1 = s1.join(cand.select(pl.col("s1_id").alias("entity_id")).unique(), on="entity_id", how="semi")
    q = q.join(cand.select(pl.col("q_id").alias("entity_id")).unique(), on="entity_id", how="semi")
    out = config.feat_dir(split)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    # keep all candidates of a query in the same chunk
    cand = cand.sort("q_id")
    for i, s in enumerate(range(0, cand.height, CHUNK)):
        part = cand.slice(s, CHUNK)
        pair_features(part, s1, q).write_parquet(out / f"part_{i:03d}.parquet")
        print(f"[features] {split} part {i}: {part.height:,} pairs ({time.time() - t:.0f}s)", flush=True)


def load(split, columns=None):
    return pl.read_parquet(config.feat_dir(split) / "part_*.parquet", columns=columns)


if __name__ == "__main__":
    for sp in (sys.argv[1:] or list(config.SPLITS)):
        build(sp)
