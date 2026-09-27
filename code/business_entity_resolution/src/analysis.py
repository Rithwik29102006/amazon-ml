"""Analysis experiments for the write-up (not needed to produce the submission).

1. Feature ablation (stage 1 only, fold 0 held out):
     v1 features (66) -> + difference/house-number features -> + token statistics
2. Country transfer (simulates France, which has no labels):
     train on US only, evaluate on India, and vice versa; compare the best
     threshold on the unseen country with the threshold chosen in training.

Usage: python analysis.py   (after build_features.py; prints tables and writes work/model/analysis.json)
"""
import json
import time

import lightgbm as lgb
import numpy as np
import polars as pl

import config
from evaluate import f05_macro, load_gt_pairs
from model import EARLY_STOP, LGB_PARAMS, MODEL_DIR, TE_NAMES, decide_threshold, fold_of, load_split, te_features, te_stats

GRID = np.round(np.arange(0.30, 0.901, 0.025), 3)
SAMPLE_PCT = 40   # train each analysis model on 40% of the training entities (memory / time)


def fit_predict(X, y, tr_idx, te_idx, es_mask, keep_mask, cols, names):
    """Train on tr_idx (restricted to keep_mask entities, to bound memory), predict te_idx."""
    fit = tr_idx[~es_mask[tr_idx] & keep_mask[tr_idx]]
    es = tr_idx[es_mask[tr_idx]]
    booster = lgb.train(LGB_PARAMS, lgb.Dataset(X[np.ix_(fit, cols)], y[fit], feature_name=names),
                        num_boost_round=1500,
                        valid_sets=[lgb.Dataset(X[np.ix_(es, cols)], y[es], feature_name=names)],
                        callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])
    p = np.concatenate([booster.predict(X[np.ix_(te_idx[i:i + 2_000_000], cols)], num_threads=config.N_JOBS)
                        for i in range(0, len(te_idx), 2_000_000)])
    return p, booster.best_iteration


def best_f(meta, idx, p, gt, ids):
    b = (meta[idx].select("q_id", "s1_id").with_columns(pl.Series("p", p))
         .filter(pl.col("p") == pl.col("p").max().over("q_id")).unique("q_id", keep="first"))
    curve = {float(t): f05_macro(decide_threshold(b, float(t)), gt, ids) for t in GRID}
    t = max(curve, key=curve.get)
    return t, curve[t], curve


def main():
    t0 = time.time()
    gt = load_gt_pairs()
    s1 = pl.read_parquet(config.norm_path("train", 1), columns=["entity_id", "country"]).rename({"entity_id": "s1_id"})
    meta, X, feats = load_split("train", n_extra=len(TE_NAMES))
    y = (meta.select("q_id", "s1_id").join(gt.with_columns(pl.lit(1, pl.Int8).alias("y")), on=["q_id", "s1_id"],
                                           how="left")["y"].fill_null(0).to_numpy())
    fold = fold_of(meta["s1_id"]).to_numpy()
    country = meta.select("s1_id").join(s1, on="s1_id", how="left")["country"].to_numpy()
    es_mask = (meta["s1_id"].hash(seed=99) % 50 == 0).to_numpy()
    keep_mask = (meta["s1_id"].hash(seed=7) % 100 < SAMPLE_PCT).to_numpy()   # entity sample for training
    names = feats + TE_NAMES
    X[:, len(feats):] = te_features(meta, te_stats(meta, y, fold), fold)
    out = {}

    # 1. ablation on fold 0
    v1 = json.loads((config.WORK_DIR / "v1_backup" / "matcher_meta.json").read_text())["features"]
    groups = {
        "v1 features (66)": [n for n in names if n in v1],
        "+ difference & house-number features": feats,
        "+ out-of-fold token statistics (all stage-1)": names,
    }
    tr, te = np.where(fold != 0)[0], np.where(fold == 0)[0]
    ids0 = s1.filter(fold_of(s1["s1_id"]) == 0)["s1_id"]
    out["ablation"] = {}
    for g, cols in groups.items():
        ci = [names.index(c) for c in cols]
        p, it = fit_predict(X, y, tr, te, es_mask, keep_mask, ci, cols)
        tau, f, _ = best_f(meta, te, p, gt, ids0)
        out["ablation"][g] = {"f05": f, "tau": tau, "iters": it, "n_features": len(cols)}
        print(f"[ablation] {g}: F0.5 {f:.5f} (tau {tau}, {len(cols)} features, {it} iters) {time.time() - t0:.0f}s",
              flush=True)

    # 2. country transfer (token statistics learned from the source country only)
    out["transfer"] = {}
    for src, dst in (("US", "India"), ("India", "US")):
        tr = np.where(country == src)[0]
        te = np.where(country == dst)[0]
        st = te_stats(meta[tr], y[tr], fold[tr])
        X[tr, len(feats):] = te_features(meta[tr], st, fold[tr])
        X[te, len(feats):] = te_features(meta[te], st)
        p, it = fit_predict(X, y, tr, te, es_mask, keep_mask, list(range(len(names))), names)
        dst_ids = s1.filter(pl.col("country") == dst)["s1_id"]
        tau, f, curve = best_f(meta, te, p, gt, dst_ids)
        out["transfer"][f"{src}->{dst}"] = {"best_tau": tau, "f05_best_tau": f, "curve": curve}
        print(f"[transfer] train {src} -> test {dst}: best tau {tau} F0.5 {f:.5f} | "
              + " ".join(f"{t}:{v:.4f}" for t, v in curve.items() if t in (0.5, 0.6, 0.625, 0.65, 0.7, 0.75)),
              flush=True)
    (MODEL_DIR / "analysis.json").write_text(json.dumps(out, indent=1))
    print(f"[analysis] done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
