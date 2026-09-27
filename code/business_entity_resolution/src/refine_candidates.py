"""Step 2b: stage-0b candidate filter (last blocking stage).

The stage-0 pruner in blocking.py only sees TF-IDF cosines and ranks. This
filter adds a handful of cheap string similarities (rapidfuzz on the core name,
its skeleton, the address and the house numbers) and drops candidate pairs it
scores below REFINE_MIN_P. On the training data it removes ~44% of the
candidate pairs while keeping essentially every pair the matcher would accept.
Its output is the final candidate set: candidate_pairs.tsv and the input to the
matching model.

Training pairs are scored out-of-fold (same 4 entity folds as train.py). Test
pairs are scored with the average of the 4 fold models.

Usage: python refine_candidates.py   (after blocking.py; rewrites work/cand/<split>_candidates.parquet,
                                      keeping the stage-0 version as <split>_candidates_stage0.parquet)
"""
import pickle
import shutil
import time

import lightgbm as lgb
import numpy as np
import polars as pl
from rapidfuzz import distance, fuzz, process

import config
from blocking import s1_rank_features
from evaluate import load_gt_pairs
from model import K_FOLDS, fold_of

REFINE_MIN_P = 0.01
BLOCK_FEATS = ["p0", "p0_rank", "p0_gap", "cos_name", "cos_addr", "cos_both", "q_rank", "q_gap_best", "q_margin",
               "q_ncand", "s_ncand", "s_rank", "v_name", "v_addr", "v_both"]
CHEAP = ["c_n_tset", "c_n_ratio", "c_n_jw", "c_ns_tset", "c_a_tset", "c_a_ratio", "c_num_jacc", "c_num_common",
         "c_a_missing_b"]
PARAMS = dict(objective="binary", learning_rate=0.1, num_leaves=63, min_data_in_leaf=500, feature_fraction=0.9,
              bagging_fraction=0.5, bagging_freq=1, num_threads=config.N_JOBS, verbose=-1, seed=config.SEED)
ROUNDS, TRAIN_FRAC = 300, 0.3
MODEL_PATH = config.WORK_DIR / "model" / "refine.pkl"


def stage0_path(split):
    return config.WORK_DIR / "cand" / f"{split}_candidates_stage0.parquet"


def cheap_features(cand: pl.DataFrame, split: str) -> np.ndarray:
    cols = ["entity_id", "name_k", "name_s", "addr_c", "addr_num"]
    s1 = pl.read_parquet(config.norm_path(split, 1), columns=cols)
    q = pl.concat([pl.read_parquet(config.norm_path(split, s), columns=cols) for s in (2, 3)])
    d = (cand.select("q_id", "s1_id").join(s1, left_on="s1_id", right_on="entity_id", how="left")
         .join(q, left_on="q_id", right_on="entity_id", how="left", suffix="_b"))
    g = lambda c: d[c].fill_null("").to_list()
    cp = lambda a, b, sc: process.cpdist(a, b, scorer=sc, workers=config.N_JOBS, dtype=np.float32)
    tok = lambda c: pl.col(c).fill_null("").str.split(" ").list.eval(pl.element().filter(pl.element() != ""))
    nums = d.select(
        tok("addr_num").list.set_intersection(tok("addr_num_b")).list.len().alias("common"),
        tok("addr_num").list.set_union(tok("addr_num_b")).list.len().alias("union"),
        (pl.col("addr_c_b").fill_null("") == "").alias("miss"))
    common = nums["common"].to_numpy().astype(np.float32)
    union = nums["union"].to_numpy().astype(np.float32)
    jacc = np.where(union > 0, common / np.maximum(union, 1), np.nan).astype(np.float32)
    nk_a, nk_b = g("name_k"), g("name_k_b")
    X = np.column_stack([
        cp(nk_a, nk_b, fuzz.token_set_ratio), cp(nk_a, nk_b, fuzz.ratio),
        cp([x.replace(" ", "") for x in nk_a], [x.replace(" ", "") for x in nk_b],
           distance.JaroWinkler.normalized_similarity),
        cp(g("name_s"), g("name_s_b"), fuzz.token_set_ratio),
        cp(g("addr_c"), g("addr_c_b"), fuzz.token_set_ratio), cp(g("addr_c"), g("addr_c_b"), fuzz.ratio),
        jacc, common, nums["miss"].to_numpy().astype(np.float32)])
    return X.astype(np.float32)


def matrix(cand, split):
    blk = np.column_stack([cand[c].cast(pl.Float32).to_numpy() for c in BLOCK_FEATS])
    return np.hstack([blk, cheap_features(cand, split)])


def filter_and_save(split, cand, p):
    cand = cand.with_columns(pl.Series("p0b", p.astype(np.float32)))
    kept = cand.filter(pl.col("p0b") >= REFINE_MIN_P)
    # competition features are recomputed on the final candidate set
    kept = s1_rank_features(kept.drop("s_ncand", "s_rank", "q_ncand_kept", "p0_gap"))
    kept.write_parquet(config.cand_path(split))
    print(f"[refine] {split}: {cand.height:,} -> {kept.height:,} pairs ({kept.height / cand.height - 1:+.1%})",
          flush=True)
    return kept


def main():
    t = time.time()
    for split in ("train", "test"):
        if not stage0_path(split).exists():
            shutil.copy(config.cand_path(split), stage0_path(split))
    gt = load_gt_pairs()

    cand = pl.read_parquet(stage0_path("train"))
    X = matrix(cand, "train")
    y = (cand.select("q_id", "s1_id").join(gt.with_columns(pl.lit(1, pl.Int8).alias("y")), on=["q_id", "s1_id"],
                                           how="left")["y"].fill_null(0).to_numpy())
    fold = fold_of(cand["s1_id"]).to_numpy()
    sub = (cand["s1_id"].hash(seed=11) % 100 < TRAIN_FRAC * 100).to_numpy()
    names = BLOCK_FEATS + CHEAP
    models, oof = [], np.zeros(len(y), dtype=np.float32)
    for f in range(K_FOLDS):
        tr = np.where((fold != f) & sub)[0]
        m = lgb.train(PARAMS, lgb.Dataset(X[tr], y[tr], feature_name=names), num_boost_round=ROUNDS)
        va = np.where(fold == f)[0]
        oof[va] = m.predict(X[va], num_threads=config.N_JOBS)
        models.append(m)
    with open(MODEL_PATH, "wb") as fh:
        pickle.dump([m.model_to_string() for m in models], fh)
    kept = filter_and_save("train", cand, oof)
    rec = kept.join(gt, on=["q_id", "s1_id"], how="semi").height / gt.height
    print(f"[refine] train candidate recall after filter: {rec:.4f} ({time.time() - t:.0f}s)", flush=True)
    del X

    cand = pl.read_parquet(stage0_path("test"))
    Xt = matrix(cand, "test")
    p = np.mean([m.predict(Xt, num_threads=config.N_JOBS) for m in models], axis=0)
    filter_and_save("test", cand, p)
    print(f"[refine] done in {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
