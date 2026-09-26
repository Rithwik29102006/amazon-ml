"""Step 4b (M3): per-country decision threshold.

Rebuilds the validation-split scores (same model, sample and split as train.py),
then tunes tau separately for each training country. Macro F0.5 is a mean over
Source-1 entities and every entity has one country, so per-country tuning is
exact. The per-country taus are written to matcher_meta.json ("tau_country")
only if they raise the overall validation F0.5; countries without training
labels (France) fall back to the global tau.

Usage: python tune_country_tau.py
"""
import json

import numpy as np
import polars as pl

import config
from build_features import load
from evaluate import f05_macro, load_gt_pairs, split_s1
from train import MAX_TRAIN_ROWS, META_PATH, assign, fit

SCORES = config.WORK_DIR / "model" / "valid_scores.parquet"
GRID = np.arange(0.20, 0.96, 0.0125)


def valid_scores(feats):
    if SCORES.exists():
        return pl.read_parquet(SCORES)
    gt = load_gt_pairs()
    df = load("train")
    df = (df.join(gt.with_columns(pl.lit(1, pl.Int8).alias("label")), on=["q_id", "s1_id"], how="left")
          .with_columns(pl.col("label").fill_null(0)))
    split = split_s1(pl.read_parquet(config.norm_path("train", 1), columns=["entity_id"])["entity_id"])
    df = df.join(split, on="s1_id", how="left")
    tr = df.filter(~pl.col("is_valid"))
    if tr.height > MAX_TRAIN_ROWS:
        tr = tr.sample(MAX_TRAIN_ROWS, seed=config.SEED)
    model = fit(tr.select(feats).to_numpy().astype(np.float32), tr["label"].to_numpy())
    del tr
    va = df.join(df.filter(pl.col("is_valid")).select("q_id").unique(), on="q_id", how="semi")
    va = va.with_columns(pl.Series("p", model.predict_proba(va.select(feats).to_numpy().astype(np.float32))[:, 1]))
    va = va.select("q_id", "s1_id", "p", "label")
    va.write_parquet(SCORES)
    return va


def main():
    meta = json.loads(META_PATH.read_text())
    va = valid_scores(meta["features"])
    gt = load_gt_pairs()
    s1 = pl.read_parquet(config.norm_path("train", 1), columns=["entity_id", "country"]).rename({"entity_id": "s1_id"})
    split = split_s1(s1["s1_id"])
    valid = s1.join(split.filter(pl.col("is_valid")), on="s1_id", how="semi")
    n_valid = valid.height

    base = f05_macro(assign(va, meta["tau"]), gt, valid["s1_id"])
    total, taus = 0.0, {}
    for (country,), ids in valid.group_by("country"):
        ids = ids["s1_id"]
        best_t, best_f = max(((float(t), f05_macro(assign(va, t), gt, ids)) for t in GRID), key=lambda x: x[1])
        taus[country] = round(best_t, 4)
        total += best_f * len(ids)
        print(f"[tau] {country}: tau {best_t:.4f} F0.5 {best_f:.4f} ({len(ids):,} S1)", flush=True)
    new = total / n_valid
    print(f"[tau] global tau {meta['tau']:.4f}: F0.5 {base:.4f} -> per-country: {new:.4f}", flush=True)
    if new > meta["valid_f05"]:
        meta["tau_country"] = taus
        meta["valid_f05_country_tau"] = new
        META_PATH.write_text(json.dumps(meta, indent=1))
        print("[tau] saved tau_country to matcher_meta.json")
    else:
        print("[tau] no gain, meta unchanged")


if __name__ == "__main__":
    main()
