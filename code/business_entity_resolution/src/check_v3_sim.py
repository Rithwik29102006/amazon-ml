"""One-off check: score the v3 models (work/v3_backup/model) out-of-fold on the
simulated test-like training data, to confirm the orphan-record diagnosis."""
import json

import lightgbm as lgb
import numpy as np
import polars as pl

import config
from evaluate import f05_macro, load_gt_pairs
from model import K_FOLDS, fold_of, load_split, stage2_frame, stage2_matrix, te_features
from train import apply_rule

V3 = config.WORK_DIR / "v3_backup" / "model"
meta = json.loads((V3 / "matcher_meta.json").read_text())
feats, te_names = meta["features"], meta["te_names"]
m, X, _ = load_split("train", feats, n_extra=len(te_names))
fold = fold_of(m["s1_id"]).to_numpy()
stats = {k: pl.read_parquet(V3 / f"te_{k}.parquet") for k in ("dB", "dA")}
X[:, len(feats):] = te_features(m, stats, fold)
p1 = np.zeros(len(fold), dtype=np.float32)
for f in range(K_FOLDS):
    idx = np.where(fold == f)[0]
    p1[idx] = lgb.Booster(model_file=str(V3 / f"s1_fold{f}.txt")).predict(X[idx], num_threads=config.N_JOBS)
pairs = m.select("q_id", "s1_id").with_columns(pl.Series("p1", p1), pl.int_range(0, pl.len(), dtype=pl.UInt32).alias("row"))
qtext = pl.concat([pl.read_parquet(config.norm_path("train", s), columns=["entity_id", "name_k", "addr_c"]) for s in (2, 3)])
b = stage2_frame(pairs, qtext)
names = feats + te_names
X2 = stage2_matrix(b, X, [names.index(n) for n in meta["stage2_base"]])
f2 = fold[b["row"].to_numpy()]
p2 = np.zeros(len(f2), dtype=np.float32)
for f in range(K_FOLDS):
    idx = np.where(f2 == f)[0]
    p2[idx] = lgb.Booster(model_file=str(V3 / f"s2_fold{f}.txt")).predict(X2[idx], num_threads=config.N_JOBS)
b = b.with_columns(pl.Series("p", p2))
gt = load_gt_pairs()
s1 = pl.read_parquet(config.norm_path("train", 1), columns=["entity_id"]).filter(config.keep_train_s1(pl.col("entity_id")))["entity_id"]
print(f"[check] v3 models on simulated test-like train data: macro F0.5 = {f05_macro(apply_rule(b, meta['decision']), gt, s1):.5f}", flush=True)
