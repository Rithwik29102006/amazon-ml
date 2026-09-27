"""Step 4: two-stage matcher with 4-fold cross-validation, and choice of the decision rule.

Folds are a deterministic hash of the Source-1 id, so every S1 entity (with all
of its candidate pairs) is scored by models that never saw it. The chosen
decision rule is the one with the best out-of-fold macro F0.5 over all
2.2M training S1 entities.

Outputs (WORK_DIR/model/):
  s1_fold{k}.txt, s2_fold{k}.txt   LightGBM models (stage 1 / stage 2)
  te_dB.parquet, te_dA.parquet     token statistics for test-time features
  matcher_meta.json                features, decision rule, CV scores
  oof_stage2.parquet               out-of-fold scores (for error analysis)

Usage: python train.py
"""
import json
import time

import lightgbm as lgb
import numpy as np
import polars as pl

import config
from evaluate import f05_macro, load_gt_pairs, split_s1
from model import (EARLY_STOP, K_FOLDS, LGB_PARAMS, MODEL_DIR, N_STAGE2_BASE, S2_NAMES, STAGE1_ROUNDS,
                   STAGE2_ROUNDS, TE_NAMES, decide_expected_f, decide_threshold, fold_of, load_split,
                   stage2_frame, stage2_matrix, te_features, te_stats)

META_PATH = MODEL_DIR / "matcher_meta.json"
TAU_GRID = np.round(np.arange(0.30, 0.901, 0.0125), 4)


def log(msg):
    print(f"[train] {msg}", flush=True)


def cv_fit(X, y, fold, names, rounds, tag, s1_ids):
    """K-fold LightGBM; returns out-of-fold predictions and summed gain importance."""
    ds = lgb.Dataset(X, y, feature_name=names, free_raw_data=False).construct()
    es = (s1_ids.hash(seed=99) % 50 == 0).to_numpy()   # 2% of entities for early stopping
    oof = np.zeros(len(y), dtype=np.float32)
    gain = np.zeros(len(names))
    for f in range(K_FOLDS):
        t = time.time()
        fit_idx = np.where((fold != f) & ~es)[0]
        es_idx = np.where((fold != f) & es)[0]
        va_idx = np.where(fold == f)[0]
        booster = lgb.train(LGB_PARAMS, ds.subset(fit_idx), num_boost_round=rounds,
                            valid_sets=[ds.subset(es_idx)],
                            callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])
        oof[va_idx] = booster.predict(X[va_idx], num_threads=config.N_JOBS)
        booster.save_model(str(MODEL_DIR / f"{tag}_fold{f}.txt"))
        gain += booster.feature_importance("gain")
        log(f"{tag} fold {f}: {booster.best_iteration} iters, {time.time() - t:.0f}s")
    return oof, gain


def best_rule(b, gt, ids, ids_by_country):
    """Search decision rules on out-of-fold scores; returns (rule, score, table)."""
    res = {}
    f_tau = {float(t): f05_macro(decide_threshold(b, float(t)), gt, ids) for t in TAU_GRID}
    tau = max(f_tau, key=f_tau.get)
    res["global_tau"] = ({"type": "threshold", "tau": tau}, f_tau[tau])
    # per-country thresholds (countries unseen in training fall back to the global tau)
    per, total = {}, 0.0
    for c, cids in ids_by_country.items():
        bc = b.filter(pl.col("country") == c)
        fc = {float(t): f05_macro(decide_threshold(bc, float(t)), gt, cids) for t in TAU_GRID}
        per[c] = max(fc, key=fc.get)
        total += fc[per[c]] * len(cids)
    res["country_tau"] = ({"type": "threshold", "tau": {"per_country": per, "default": tau}}, total / len(ids))
    # expected-F0.5 set selection per S1 entity
    best_ef = None
    for m in (0.0, 0.03, 0.065, 0.1):
        for gamma in (0.8, 0.9, 1.0, 1.1, 1.25):
            f = f05_macro(decide_expected_f(b, m, gamma), gt, ids)
            if best_ef is None or f > best_ef[1]:
                best_ef = ({"type": "expected_f", "m": m, "gamma": gamma}, f)
    res["expected_f"] = best_ef
    name = max(res, key=lambda k: res[k][1])
    return res[name][0], res[name][1], {k: v[1] for k, v in res.items()}, f_tau


def apply_rule(b, rule):
    if rule["type"] == "expected_f":
        return decide_expected_f(b, rule["m"], rule["gamma"])
    return decide_threshold(b, rule["tau"])


def main():
    t0 = time.time()
    gt = load_gt_pairs()
    s1 = (pl.read_parquet(config.norm_path("train", 1), columns=["entity_id", "country"])
          .rename({"entity_id": "s1_id"}).filter(config.keep_train_s1(pl.col("s1_id"))))
    ids = s1["s1_id"]
    ids_by_country = {c: g["s1_id"] for (c,), g in s1.group_by("country")}
    holdout = split_s1(ids).filter(pl.col("is_valid"))["s1_id"]   # the earlier 15% validation split

    # ---------------------------------------------------------------- stage 1
    meta, X, feats = load_split("train", n_extra=len(TE_NAMES))
    y = (meta.select("q_id", "s1_id").join(gt.with_columns(pl.lit(1, pl.Int8).alias("y")),
                                           on=["q_id", "s1_id"], how="left")["y"].fill_null(0).to_numpy())
    fold = fold_of(meta["s1_id"]).to_numpy()
    stats = te_stats(meta, y, fold)
    for k, v in stats.items():
        v.write_parquet(MODEL_DIR / f"te_{k}.parquet")
    X[:, len(feats):] = te_features(meta, stats, fold)
    names = feats + TE_NAMES
    log(f"{X.shape[0]:,} pairs x {X.shape[1]} features, {y.sum():,} positives ({time.time() - t0:.0f}s)")
    p1, gain1 = cv_fit(X, y, fold, names, STAGE1_ROUNDS, "s1", meta["s1_id"])

    pairs = meta.select("q_id", "s1_id").with_columns(pl.Series("p1", p1),
                                                      pl.int_range(0, pl.len(), dtype=pl.UInt32).alias("row"))
    b1 = (pairs.filter(pl.col("p1") == pl.col("p1").max().over("q_id")).unique("q_id", keep="first")
          .rename({"p1": "p"}).join(s1, on="s1_id", how="left"))
    f_stage1 = max(f05_macro(decide_threshold(b1, float(t)), gt, ids) for t in TAU_GRID)
    log(f"stage 1 out-of-fold F0.5 (best global tau) = {f_stage1:.5f}")

    # ---------------------------------------------------------------- stage 2
    qtext = pl.concat([pl.read_parquet(config.norm_path("train", s), columns=["entity_id", "name_k", "addr_c"])
                       for s in (2, 3)]).join(meta.select(pl.col("q_id").alias("entity_id")).unique(),
                                              on="entity_id", how="semi")
    b = stage2_frame(pairs, qtext)
    order = np.argsort(-gain1)[:N_STAGE2_BASE]
    base = [names[i] for i in order]
    X2 = stage2_matrix(b, X, order)
    y2, fold2 = y[b["row"].to_numpy()], fold[b["row"].to_numpy()]
    del X
    names2 = base + S2_NAMES
    log(f"stage 2: {X2.shape[0]:,} rows x {X2.shape[1]} features ({time.time() - t0:.0f}s)")
    p2, gain2 = cv_fit(X2, y2, fold2, names2, STAGE2_ROUNDS, "s2", b["s1_id"])
    b = b.with_columns(pl.Series("p", p2)).join(s1, on="s1_id", how="left")
    b.select("q_id", "s1_id", "country", "p1", "p", pl.Series("label", y2)).write_parquet(MODEL_DIR / "oof_stage2.parquet")

    # ---------------------------------------------------------------- decision rule
    rule, f_best, table, curve = best_rule(b, gt, ids, ids_by_country)
    pred = apply_rule(b, rule)
    per_country = {c: f05_macro(pred, gt, cids) for c, cids in ids_by_country.items()}
    f_holdout = f05_macro(pred, gt, holdout)
    log(f"stage 2 out-of-fold F0.5: " + ", ".join(f"{k} {v:.5f}" for k, v in table.items()))
    log(f"chosen rule {rule} -> CV F0.5 {f_best:.5f} | per country {per_country} | old 15% holdout {f_holdout:.5f}")

    imp1 = sorted(zip(names, gain1.tolist()), key=lambda x: -x[1])
    imp2 = sorted(zip(names2, gain2.tolist()), key=lambda x: -x[1])
    META_PATH.write_text(json.dumps({
        "features": feats, "te_names": TE_NAMES, "stage2_base": base, "stage2_names": S2_NAMES,
        "k_folds": K_FOLDS, "decision": rule, "cv_f05": f_best, "cv_f05_stage1": f_stage1,
        "cv_f05_rules": table, "cv_f05_per_country": per_country, "holdout_f05": f_holdout,
        "tau_curve": {str(k): v for k, v in curve.items()},
        "importance_stage1_top25": imp1[:25], "importance_stage2_top25": imp2[:25],
    }, indent=1))
    log(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
