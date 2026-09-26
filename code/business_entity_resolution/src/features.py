"""Step 3: pairwise features for every candidate pair.

All string similarities are computed with rapidfuzz.process.cpdist which
scores aligned lists in C++ with all cores. No country one-hot is used, so
the model transfers to countries unseen in training (France).
"""
import numpy as np
import polars as pl
from rapidfuzz import distance, fuzz, process

import config

NORM_COLS = ["entity_id", "name_c", "name_k", "name_s", "name_parts", "legal",
             "nonlatin", "is_domain", "addr_c", "addr_num", "addr_s"]


def load_norm(split):
    s1 = pl.read_parquet(config.norm_path(split, 1), columns=NORM_COLS)
    q = pl.concat([pl.read_parquet(config.norm_path(split, s), columns=NORM_COLS + ["src"])
                   for s in (2, 3)])
    return s1, q


def _cp(a, b, scorer, dtype=np.float32):
    return process.cpdist(a, b, scorer=scorer, workers=config.N_JOBS, dtype=dtype).astype(np.float32)


def _list_jacc(a, b):
    inter = a.list.set_intersection(b).list.len()
    union = a.list.set_union(b).list.len()
    return pl.when(union > 0).then(inter / union).otherwise(None)


def pair_features(cand: pl.DataFrame, s1: pl.DataFrame, q: pl.DataFrame) -> pl.DataFrame:
    """cand must have q_id, s1_id and the blocking score/rank columns."""
    d = (cand.join(s1, left_on="s1_id", right_on="entity_id", how="left")
         .join(q, left_on="q_id", right_on="entity_id", how="left", suffix="_b"))
    g = lambda c: d[c].fill_null("").to_list()
    nk_a, nk_b = g("name_k"), g("name_k_b")
    ns_a, ns_b = g("name_s"), g("name_s_b")
    nc_a, nc_b = g("name_c"), g("name_c_b")
    ac_a, ac_b = g("addr_c"), g("addr_c_b")
    as_a, as_b = g("addr_s"), g("addr_s_b")
    nsp_a = [x.replace(" ", "") for x in nk_a]
    nsp_b = [x.replace(" ", "") for x in nk_b]
    JW = distance.JaroWinkler.normalized_similarity
    f = {
        "n_ratio": _cp(nk_a, nk_b, fuzz.ratio),
        "n_tset": _cp(nk_a, nk_b, fuzz.token_set_ratio),
        "n_tsort": _cp(nk_a, nk_b, fuzz.token_sort_ratio),
        "n_partial": _cp(nk_a, nk_b, fuzz.partial_ratio),
        "n_wratio": _cp(nk_a, nk_b, fuzz.WRatio),
        "n_jw_ns": _cp(nsp_a, nsp_b, JW),
        "n_ratio_ns": _cp(nsp_a, nsp_b, fuzz.ratio),
        "n_partial_ns": _cp(nsp_a, nsp_b, fuzz.partial_ratio),
        "nc_tset": _cp(nc_a, nc_b, fuzz.token_set_ratio),
        "ns_ratio": _cp(ns_a, ns_b, fuzz.ratio),
        "ns_tset": _cp(ns_a, ns_b, fuzz.token_set_ratio),
        "ns_jw": _cp([x.replace(" ", "") for x in ns_a], [x.replace(" ", "") for x in ns_b], JW),
        "a_ratio": _cp(ac_a, ac_b, fuzz.ratio),
        "a_tset": _cp(ac_a, ac_b, fuzz.token_set_ratio),
        "a_tsort": _cp(ac_a, ac_b, fuzz.token_sort_ratio),
        "a_partial": _cp(ac_a, ac_b, fuzz.partial_ratio),
        "as_tset": _cp(as_a, as_b, fuzz.token_set_ratio),
    }
    out = d.select("q_id", "s1_id").with_columns(**{k: pl.Series(v) for k, v in f.items()})

    tok = lambda c: pl.col(c).fill_null("").str.split(" ").list.eval(pl.element().filter(pl.element() != ""))
    extra = d.select(
        # name token sets
        _list_jacc(tok("name_k"), tok("name_k_b")).alias("n_jacc"),
        _list_jacc(tok("name_s"), tok("name_s_b")).alias("ns_jacc"),
        (tok("name_k").list.first() == tok("name_k_b").list.first()).alias("n_first_eq"),
        (tok("name_s").list.first() == tok("name_s_b").list.first()).alias("ns_first_eq"),
        tok("name_k").list.len().alias("n_ntok_a"),
        tok("name_k_b").list.len().alias("n_ntok_b"),
        pl.col("name_k").str.len_chars().alias("n_len_a"),
        pl.col("name_k_b").str.len_chars().alias("n_len_b"),
        # space-free containment (domains / handles / concatenations)
        (pl.col("name_k_b").str.replace_all(" ", "").str.contains(pl.col("name_k").str.replace_all(" ", ""), literal=True)
         | pl.col("name_k").str.replace_all(" ", "").str.contains(pl.col("name_k_b").str.replace_all(" ", ""), literal=True))
        .alias("n_contains"),
        # legal form agreement: 1 same, 0 one missing, -1 conflicting
        pl.when((pl.col("legal") == "") | (pl.col("legal_b") == "")).then(0)
        .when(pl.col("legal") == pl.col("legal_b")).then(1).otherwise(-1).alias("legal_agree"),
        pl.col("nonlatin").cast(pl.Int8).alias("nonlatin_a"),
        pl.col("nonlatin_b").cast(pl.Int8).alias("nonlatin_b"),
        pl.col("is_domain_b").cast(pl.Int8).alias("domain_b"),
        (pl.col("name_parts_b") != "").cast(pl.Int8).alias("alias_b"),
        pl.col("src").alias("src_b"),
        # address
        (pl.col("addr_c").fill_null("") == "").cast(pl.Int8).alias("a_missing_a"),
        (pl.col("addr_c_b").fill_null("") == "").cast(pl.Int8).alias("a_missing_b"),
        _list_jacc(tok("addr_c"), tok("addr_c_b")).alias("a_jacc"),
        _list_jacc(tok("addr_s"), tok("addr_s_b")).alias("as_jacc"),
        _list_jacc(tok("addr_num"), tok("addr_num_b")).alias("num_jacc"),
        tok("addr_num").list.set_intersection(tok("addr_num_b")).list.len().alias("num_common"),
        tok("addr_num").list.len().alias("num_n_a"),
        tok("addr_num_b").list.len().alias("num_n_b"),
        (tok("addr_num").list.first() == tok("addr_num_b").list.first()).alias("num_first_eq"),
        (tok("addr_num_b").list.set_difference(tok("addr_num")).list.len() == 0).alias("num_b_subset"),
        tok("addr_c").list.len().alias("a_ntok_a"),
        tok("addr_c_b").list.len().alias("a_ntok_b"),
    )
    out = pl.concat([out, extra], how="horizontal")

    # alias names (DBA / FKA / "|"): best token-set similarity over the parts
    alias = d.select("q_id", "s1_id", "name_k", "name_parts_b").filter(pl.col("name_parts_b") != "")
    if alias.height:
        ex = alias.with_columns(pl.col("name_parts_b").str.split("|")).explode("name_parts_b")
        ex = ex.with_columns(pl.Series("sc", _cp(ex["name_k"].to_list(), ex["name_parts_b"].to_list(),
                                                 fuzz.token_set_ratio)))
        best = ex.group_by("q_id", "s1_id").agg(pl.col("sc").max().alias("n_alias_best"))
        out = out.join(best, on=["q_id", "s1_id"], how="left")
    else:
        out = out.with_columns(pl.lit(None, pl.Float32).alias("n_alias_best"))
    out = out.with_columns(pl.max_horizontal("n_tset", pl.col("n_alias_best").fill_null(0)).alias("n_best"))

    keep = [c for c in cand.columns if c not in ("q_id", "s1_id")]
    out = out.join(cand.select(["q_id", "s1_id"] + keep), on=["q_id", "s1_id"], how="left")
    bool_cols = [c for c, t in out.schema.items() if t == pl.Boolean]
    return out.with_columns(pl.col(bool_cols).cast(pl.Int8))


# the model sees every column except the ids / label
def feature_columns(df):
    return [c for c in df.columns if c not in ("q_id", "s1_id", "label", "is_valid")]
