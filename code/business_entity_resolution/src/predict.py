"""Step 5: score the test candidates with the two-stage model and write the submission files.

  output/matching_results.tsv  - final matches (scored on the leaderboard)
  output/candidate_pairs.tsv   - exactly the pairs the matcher ran inference on

Usage: python predict.py
"""
import json
import time

import lightgbm as lgb
import numpy as np
import polars as pl

import config
from model import K_FOLDS, MODEL_DIR, load_split, stage2_frame, stage2_matrix, te_features
from train import META_PATH, apply_rule


def write_lists(pairs: pl.DataFrame, s1_ids: pl.Series, col: str, path):
    """One row per Source-1 id, comma-joined S2/S3 ids (empty when none)."""
    grouped = pairs.unique().sort("q_id").group_by("s1_id").agg(pl.col("q_id").str.join(",").alias(col))
    out = (pl.DataFrame({"source1_entity_id": s1_ids})
           .join(grouped.rename({"s1_id": "source1_entity_id"}), on="source1_entity_id", how="left")
           .with_columns(pl.col(col).fill_null("")))
    out.write_csv(path, separator="\t", quote_style="never")
    return out


def predict_folds(tag, X):
    p = np.zeros(X.shape[0], dtype=np.float64)
    for f in range(K_FOLDS):
        p += lgb.Booster(model_file=str(MODEL_DIR / f"{tag}_fold{f}.txt")).predict(X, num_threads=config.N_JOBS)
    return (p / K_FOLDS).astype(np.float32)


def main():
    t = time.time()
    meta_json = json.loads(META_PATH.read_text())
    feats, te_names = meta_json["features"], meta_json["te_names"]
    meta, X, _ = load_split("test", feats, n_extra=len(te_names))
    stats = {k: pl.read_parquet(MODEL_DIR / f"te_{k}.parquet") for k in ("dB", "dA")}
    X[:, len(feats):] = te_features(meta, stats)
    p1 = predict_folds("s1", X)
    pairs = meta.select("q_id", "s1_id").with_columns(pl.Series("p1", p1),
                                                      pl.int_range(0, pl.len(), dtype=pl.UInt32).alias("row"))
    print(f"[predict] stage 1 done ({time.time() - t:.0f}s)", flush=True)

    qtext = pl.concat([pl.read_parquet(config.norm_path("test", s), columns=["entity_id", "name_k", "addr_c"])
                       for s in (2, 3)]).join(meta.select(pl.col("q_id").alias("entity_id")).unique(),
                                              on="entity_id", how="semi")
    b = stage2_frame(pairs, qtext)
    names = feats + te_names
    base_idx = [names.index(n) for n in meta_json["stage2_base"]]
    X2 = stage2_matrix(b, X, base_idx)
    del X
    s1c = pl.read_parquet(config.norm_path("test", 1), columns=["entity_id", "country"]).rename({"entity_id": "s1_id"})
    b = b.with_columns(pl.Series("p", predict_folds("s2", X2))).join(s1c, on="s1_id", how="left")
    matches = apply_rule(b, meta_json["decision"])
    b.select("q_id", "s1_id", "country", "p1", "p").write_parquet(MODEL_DIR / "test_scores.parquet")

    s1_ids = pl.read_csv(config.raw_path("test", 1), separator="\t", quote_char=None,
                         infer_schema_length=0, columns=["entity_id"])["entity_id"]
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    m = write_lists(matches, s1_ids, "matched_entity_ids", config.OUT_DIR / "matching_results.tsv")
    write_lists(meta.select("s1_id", "q_id"), s1_ids, "candidate_entity_ids", config.OUT_DIR / "candidate_pairs.tsv")
    n_single = (m["matched_entity_ids"] == "").sum()
    print(f"[predict] rule {meta_json['decision']}: {matches.height:,} matches, "
          f"{n_single:,}/{len(s1_ids):,} S1 without match ({time.time() - t:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
