"""Step 5: score the test candidates and write the two submission files.

  output/matching_results.tsv  - final matches (scored on the leaderboard)
  output/candidate_pairs.tsv   - exactly the pairs the matcher ran inference on

Usage: python predict.py
"""
import json
import pickle
import time

import numpy as np
import polars as pl

import config
from build_features import load
from train import META_PATH, MODEL_PATH, assign


def write_lists(pairs: pl.DataFrame, s1_ids: pl.Series, col: str, path):
    """One row per Source-1 id, comma-joined S2/S3 ids (empty when none)."""
    grouped = pairs.unique().sort("q_id").group_by("s1_id").agg(pl.col("q_id").str.join(",").alias(col))
    out = (pl.DataFrame({"source1_entity_id": s1_ids})
           .join(grouped.rename({"s1_id": "source1_entity_id"}), on="source1_entity_id", how="left")
           .with_columns(pl.col(col).fill_null("")))
    out.write_csv(path, separator="\t", quote_style="never")
    return out


def main():
    t = time.time()
    meta = json.loads(META_PATH.read_text())
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    feats, tau = meta["features"], meta["tau"]
    df = load("test", columns=["q_id", "s1_id"] + feats)
    df = df.with_columns(pl.Series("p", model.predict_proba(df.select(feats).to_numpy().astype(np.float32))[:, 1]))
    if meta.get("tau_country"):
        # per-country tau (tune_country_tau.py); unseen countries keep the global tau
        s1c = pl.read_parquet(config.norm_path("test", 1), columns=["entity_id", "country"])
        best = assign(df, 0.0).join(s1c.rename({"entity_id": "s1_id"}), on="s1_id", how="left")
        best = best.join(df.select("q_id", "s1_id", "p"), on=["q_id", "s1_id"], how="left")
        t_col = pl.col("country").replace_strict(meta["tau_country"], default=tau, return_dtype=pl.Float64)
        matches = best.filter(pl.col("p") >= t_col).select("s1_id", "q_id")
    else:
        matches = assign(df, tau)

    s1_ids = pl.read_csv(config.raw_path("test", 1), separator="\t", quote_char=None,
                         infer_schema_length=0, columns=["entity_id"])["entity_id"]
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    m = write_lists(matches, s1_ids, "matched_entity_ids", config.OUT_DIR / "matching_results.tsv")
    write_lists(df.select("s1_id", "q_id"), s1_ids, "candidate_entity_ids", config.OUT_DIR / "candidate_pairs.tsv")
    n_single = (m["matched_entity_ids"] == "").sum()
    print(f"[predict] tau={tau:.3f}: {matches.height:,} matches, {n_single:,}/{len(s1_ids):,} S1 without match "
          f"({time.time() - t:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
