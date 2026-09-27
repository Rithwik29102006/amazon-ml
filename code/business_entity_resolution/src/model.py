"""Shared modelling code for train.py and predict.py.

Stage 1  LightGBM on the pair features plus out-of-fold token statistics.
Stage 2  Every S2/S3 record keeps only its best stage-1 candidate. A second
         LightGBM re-scores that pair with context features: competition
         inside the query, how many other records the same S1 entity already
         attracts, and how similar this record is to them (collective evidence).
Decision Per S1 entity, either a probability threshold (global or per
         country) or the subset of its records that maximises the expected F0.5.
"""
import numpy as np
import polars as pl
from rapidfuzz import fuzz, process

import config
from build_features import load
from features import feature_columns

K_FOLDS = 4
TE_ALPHA = 30.0
TE_MAX_TOK = 4

LGB_PARAMS = dict(objective="binary", learning_rate=0.1, num_leaves=255, min_data_in_leaf=200,
                  feature_fraction=0.8, bagging_fraction=0.7, bagging_freq=1, lambda_l2=1.0,
                  max_bin=255, num_threads=config.N_JOBS, verbose=-1, seed=config.SEED)
STAGE1_ROUNDS = 1500
STAGE2_ROUNDS = 1000
EARLY_STOP = 50
N_STAGE2_BASE = 30          # top stage-1 features (by gain) passed on to stage 2
MODEL_DIR = config.WORK_DIR / "model"


# ---------------------------------------------------------------------------- data
def fold_of(ids: pl.Series) -> pl.Series:
    return (ids.hash(seed=config.SEED + 1) % K_FOLDS).cast(pl.Int8)


def load_split(split, feats=None, n_extra=0):
    """Returns (meta frame with ids / helper strings, float32 feature matrix, feature names).

    The matrix has n_extra spare columns at the end (filled later with token statistics).
    """
    lf = pl.scan_parquet(config.feat_dir(split) / "part_*.parquet")
    if feats is None:
        feats = feature_columns(lf.head(1).collect())
    meta = lf.select("q_id", "s1_id", "dA", "dB").collect()
    X = np.full((meta.height, len(feats) + n_extra), np.nan, dtype=np.float32)
    for i, c in enumerate(feats):
        X[:, i] = lf.select(pl.col(c).cast(pl.Float32)).collect().to_series().to_numpy()
    return meta, X, feats


# --------------------------------------------------------------- token statistics
def _tok_rows(meta: pl.DataFrame, col: str) -> pl.DataFrame:
    return (meta.select(pl.int_range(0, pl.len(), dtype=pl.UInt32).alias("rid"),
                        pl.col(col).fill_null("").str.split(" ").list.head(TE_MAX_TOK).alias("t"))
            .explode("t").filter(pl.col("t").is_not_null() & (pl.col("t") != "")))


def te_stats(meta: pl.DataFrame, y: np.ndarray, fold: np.ndarray) -> dict:
    """Per-token (count, positives) per fold for the differing tokens dB / dA."""
    out = {}
    for col in ("dB", "dA"):
        tr = _tok_rows(meta, col)
        tr = tr.with_columns(pl.Series("y", y.astype(np.int32)).gather(tr["rid"]).alias("y"),
                             pl.Series("f", fold.astype(np.int8)).gather(tr["rid"]).alias("f"))
        out[col] = tr.group_by("t", "f").agg(pl.len().alias("n"), pl.col("y").sum().alias("pos"))
    return out


def te_features(meta: pl.DataFrame, stats: dict, fold: np.ndarray = None) -> np.ndarray:
    """Smoothed positive rate of the differing tokens.

    With `fold`, every row uses statistics from the *other* folds only (out of fold).
    Without it (test), statistics from all training folds are used.
    """
    cols = []
    for col in ("dB", "dA"):
        st = stats[col]
        tot = st.group_by("t").agg(pl.col("n").sum().alias("N"), pl.col("pos").sum().alias("P"))
        prior = float(tot["P"].sum() / max(1, tot["N"].sum()))
        rows = _tok_rows(meta, col)
        if fold is not None:
            rows = rows.with_columns(pl.Series("f", fold.astype(np.int8)).gather(rows["rid"]).alias("f"))
            rows = (rows.join(tot, on="t", how="left")
                    .join(st.rename({"n": "fn", "pos": "fpos"}), on=["t", "f"], how="left")
                    .with_columns((pl.col("N").fill_null(0) - pl.col("fn").fill_null(0)).alias("n"),
                                  (pl.col("P").fill_null(0) - pl.col("fpos").fill_null(0)).alias("pos")))
        else:
            rows = rows.join(tot, on="t", how="left").with_columns(
                pl.col("N").fill_null(0).alias("n"), pl.col("P").fill_null(0).alias("pos"))
        rows = rows.with_columns(((pl.col("pos") + TE_ALPHA * prior) / (pl.col("n") + TE_ALPHA)).alias("r"),
                                 pl.col("n").cast(pl.Float32).log1p().alias("ln"))
        agg = rows.group_by("rid").agg(pl.col("r").max().alias("mx"), pl.col("r").min().alias("mn"),
                                       pl.col("ln").min().alias("lnmin"))
        full = pl.DataFrame({"rid": np.arange(meta.height, dtype=np.uint32)}).join(agg, on="rid", how="left").sort("rid")
        cols += [full["mx"].to_numpy(), full["mn"].to_numpy(), full["lnmin"].to_numpy()]
    return np.column_stack(cols).astype(np.float32)


TE_NAMES = ["te_b_max", "te_b_min", "te_b_lncount", "te_a_max", "te_a_min", "te_a_lncount"]


# ------------------------------------------------------------------------ stage 2
S2_NAMES = ["p1", "q_gap1", "q_nconf1", "q_ncand1", "s_nbest", "s_nconf_other", "s_sum_other",
            "s_max_other", "s_rank1", "cons_name_max", "cons_name_mean", "cons_addr_max",
            "cons_addr_mean", "cons_n"]


def stage2_frame(pairs: pl.DataFrame, qtext: pl.DataFrame) -> pl.DataFrame:
    """pairs: q_id, s1_id, p1, row (index into the stage-1 matrix) for all candidates.

    Returns one row per query (its best stage-1 candidate) with stage-2 features.
    qtext: entity_id, name_k, addr_c of the Source-2/3 records.
    """
    p = pairs.with_columns(pl.col("p1").rank("ordinal", descending=True).over("q_id").alias("r"))
    second = p.filter(pl.col("r") == 2).select("q_id", pl.col("p1").alias("p1_2nd"))
    qagg = p.group_by("q_id").agg((pl.col("p1") > 0.5).sum().alias("q_nconf1"), pl.len().alias("q_ncand1"))
    b = (p.filter(pl.col("r") == 1).drop("r").join(second, on="q_id", how="left").join(qagg, on="q_id", how="left")
         .with_columns((pl.col("p1") - pl.col("p1_2nd").fill_null(0)).alias("q_gap1")).drop("p1_2nd"))
    conf = (pl.col("p1") > 0.5).cast(pl.Int32)
    top2 = pl.col("p1").sort(descending=True)
    b = b.with_columns(
        (pl.len().over("s1_id") - 1).alias("s_nbest"),
        (conf.sum().over("s1_id") - conf).alias("s_nconf_other"),
        (pl.col("p1").sum().over("s1_id") - pl.col("p1")).alias("s_sum_other"),
        pl.when(pl.col("p1") == top2.first().over("s1_id"))
        .then(top2.slice(1, 1).first().over("s1_id")).otherwise(top2.first().over("s1_id")).alias("s_max_other"),
        pl.col("p1").rank("ordinal", descending=True).over("s1_id").alias("s_rank1"),
    )
    # collective evidence: similarity to the entity's other confident records
    c = b.filter(pl.col("p1") > 0.5).select("s1_id", pl.col("q_id").alias("q2"))
    parts = []
    bucket = b.select("s1_id", "q_id").with_columns((pl.col("s1_id").hash(seed=3) % 8).alias("h"))
    for h in range(8):
        x = (bucket.filter(pl.col("h") == h).drop("h").join(c, on="s1_id").filter(pl.col("q_id") != pl.col("q2"))
             .join(qtext, left_on="q_id", right_on="entity_id", how="left")
             .join(qtext, left_on="q2", right_on="entity_id", how="left", suffix="_2"))
        if x.height == 0:
            continue
        g = lambda col: x[col].fill_null("").to_list()
        x = x.select("s1_id", "q_id").with_columns(
            pl.Series("sn", process.cpdist(g("name_k"), g("name_k_2"), scorer=fuzz.token_set_ratio,
                                           workers=config.N_JOBS, dtype=np.float32)),
            pl.Series("sa", process.cpdist(g("addr_c"), g("addr_c_2"), scorer=fuzz.token_set_ratio,
                                           workers=config.N_JOBS, dtype=np.float32)))
        parts.append(x.group_by("s1_id", "q_id").agg(
            pl.col("sn").max().alias("cons_name_max"), pl.col("sn").mean().alias("cons_name_mean"),
            pl.col("sa").max().alias("cons_addr_max"), pl.col("sa").mean().alias("cons_addr_mean"),
            pl.len().alias("cons_n")))
    if parts:
        b = b.join(pl.concat(parts), on=["s1_id", "q_id"], how="left")
    else:
        b = b.with_columns([pl.lit(None, pl.Float32).alias(n) for n in S2_NAMES[9:]])
    return b.with_columns(pl.col("cons_n").fill_null(0))


def stage2_matrix(b: pl.DataFrame, X1: np.ndarray, base_idx) -> np.ndarray:
    rows = b["row"].to_numpy()
    s2 = np.column_stack([b[c].cast(pl.Float32).to_numpy() for c in S2_NAMES]).astype(np.float32)
    return np.hstack([X1[rows][:, base_idx], s2])


# ----------------------------------------------------------------------- decisions
def decide_threshold(b: pl.DataFrame, tau) -> pl.DataFrame:
    """b: s1_id, q_id, p (one row per query), optional country. tau: float or {country: tau}."""
    if isinstance(tau, dict):
        t = pl.col("country").replace_strict(tau["per_country"], default=tau["default"], return_dtype=pl.Float64)
        return b.filter(pl.col("p") >= t).select("s1_id", "q_id")
    return b.filter(pl.col("p") >= tau).select("s1_id", "q_id")


def decide_expected_f(b: pl.DataFrame, m: float, gamma: float, beta2: float = 0.25) -> pl.DataFrame:
    """Per S1 entity pick the top-k of its records maximising the expected F-beta.

    E[F | top k] ~ (1+b2) * sum_{i<=k} p_i / (b2 * gamma * (sum_all p + m) + k)
    E[F | empty] = P(no true match) ~ prod(1 - p_i) * exp(-m)
    m: expected number of true matches that blocking never retrieved.
    """
    eps = 1e-6
    x = (b.select("s1_id", "q_id", pl.col("p").clip(eps, 1 - eps))
         .sort(["s1_id", "p"], descending=[False, True])
         .with_columns(pl.int_range(1, pl.len() + 1).over("s1_id").alias("k"),
                       pl.col("p").cum_sum().over("s1_id").alias("S"),
                       pl.col("p").sum().over("s1_id").alias("P"),
                       (1 - pl.col("p")).log().sum().over("s1_id").alias("L0")))
    x = x.with_columns(((1 + beta2) * pl.col("S") / (beta2 * gamma * (pl.col("P") + m) + pl.col("k"))).alias("E"),
                       (pl.col("L0") - m).exp().alias("E0"))
    x = x.with_columns(pl.col("E").max().over("s1_id").alias("Emax"))
    kstar = x.filter(pl.col("E") == pl.col("Emax")).group_by("s1_id").agg(pl.col("k").min().alias("kstar"))
    x = x.join(kstar, on="s1_id").filter((pl.col("Emax") > pl.col("E0")) & (pl.col("k") <= pl.col("kstar")))
    return x.select("s1_id", "q_id")
