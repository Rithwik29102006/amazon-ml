"""Step 4: train the pair classifier, choose the decision threshold on the
validation split (macro F0.5), then refit on all training data.

Decision rule (see assign()):
  every Source-2/3 record belongs to at most one Source-1 entity, so for each
  query we keep only its highest-probability Source-1 candidate, and only if
  that probability >= tau.

Usage: python train.py
"""
import json
import pickle
import time

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier

import config
from build_features import load
from evaluate import f05_macro, load_gt_pairs, split_s1
from features import feature_columns

MODEL_PATH = config.WORK_DIR / "model" / "matcher.pkl"
META_PATH = config.WORK_DIR / "model" / "matcher_meta.json"
MAX_TRAIN_ROWS = 12_000_000

PARAMS = dict(max_iter=600, learning_rate=0.08, max_leaf_nodes=127, min_samples_leaf=100,
              l2_regularization=1.0, early_stopping=True, validation_fraction=0.05,
              n_iter_no_change=30, random_state=config.SEED)


def assign(df: pl.DataFrame, tau: float) -> pl.DataFrame:
    """df: q_id, s1_id, p -> accepted (s1_id, q_id) pairs."""
    best = df.filter(pl.col("p") == pl.col("p").max().over("q_id")).unique("q_id", keep="first")
    return best.filter(pl.col("p") >= tau).select("s1_id", "q_id")


def fit(X, y):
    model = HistGradientBoostingClassifier(**PARAMS)
    model.fit(X, y)
    return model


def main():
    t = time.time()
    gt = load_gt_pairs()
    df = load("train")
    feats = feature_columns(df)
    df = (df.join(gt.with_columns(pl.lit(1, pl.Int8).alias("label")), on=["q_id", "s1_id"], how="left")
          .with_columns(pl.col("label").fill_null(0)))
    s1_all = pl.read_parquet(config.norm_path("train", 1), columns=["entity_id"])["entity_id"]
    split = split_s1(s1_all)
    df = df.join(split, on="s1_id", how="left")
    valid_ids = split.filter(pl.col("is_valid"))["s1_id"]
    print(f"[train] {df.height:,} pairs, {len(feats)} features, positives {df['label'].sum():,}", flush=True)

    # ---- validation run -------------------------------------------------------------
    tr = df.filter(~pl.col("is_valid"))
    if tr.height > MAX_TRAIN_ROWS:
        tr = tr.sample(MAX_TRAIN_ROWS, seed=config.SEED)
    model = fit(tr.select(feats).to_numpy().astype(np.float32), tr["label"].to_numpy())
    print(f"[train] validation model: {model.n_iter_} iters ({time.time() - t:.0f}s)", flush=True)
    del tr
    # score every pair whose query could land on a validation entity
    va = df.filter(pl.col("is_valid"))
    va_q = va.select("q_id").unique()
    va = df.join(va_q, on="q_id", how="semi")
    va = va.with_columns(pl.Series("p", model.predict_proba(va.select(feats).to_numpy().astype(np.float32))[:, 1]))

    best_tau, best_f = 0.5, -1.0
    curve = {}
    for tau in np.arange(0.20, 0.96, 0.025):
        f = f05_macro(assign(va, tau), gt, valid_ids)
        curve[round(float(tau), 3)] = f
        if f > best_f:
            best_tau, best_f = float(tau), f
    blk = va.filter(pl.col("label") == 1).select("s1_id", "q_id")
    recall_ceiling = gt.join(pl.DataFrame({"s1_id": valid_ids}), on="s1_id", how="semi")
    recall_ceiling = blk.join(recall_ceiling, on=["s1_id", "q_id"], how="semi").height / max(1, recall_ceiling.height)
    perfect = f05_macro(blk.join(pl.DataFrame({"s1_id": valid_ids}), on="s1_id", how="semi"), gt, valid_ids)
    print(f"[train] valid F0.5 = {best_f:.4f} at tau = {best_tau:.3f} | "
          f"candidate recall {recall_ceiling:.4f} | oracle F0.5 on candidates {perfect:.4f}", flush=True)
    for k, v in curve.items():
        print(f"    tau {k:.3f}: {v:.4f}")

    # feature importance proxy: not available for HGB -> skip; store metadata
    va.select("q_id", "s1_id", "p", "label").write_parquet(config.WORK_DIR / "model" / "valid_scores.parquet")

    # ---- final model on all training pairs -----------------------------------------------
    full = df if df.height <= MAX_TRAIN_ROWS else df.sample(MAX_TRAIN_ROWS, seed=config.SEED)
    final = fit(full.select(feats).to_numpy().astype(np.float32), full["label"].to_numpy())
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(final, f)
    META_PATH.write_text(json.dumps({"features": feats, "tau": best_tau, "valid_f05": best_f,
                                     "candidate_recall_valid": recall_ceiling,
                                     "oracle_f05_valid": perfect, "curve": curve}, indent=1))
    print(f"[train] final model {final.n_iter_} iters saved ({time.time() - t:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
