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


ADMIN_WORDS = ["cdp", "city", "village", "borough", "town", "township", "twp", "cnty", "county",
               "dist", "of", "municipality"]
MAX_DIFF = 4  # cap on differing tokens compared pairwise


def _tok(c):
    return pl.col(c).fill_null("").str.split(" ").list.eval(pl.element().filter(pl.element() != ""))


def _cross(x, a, b, scorers):
    """Pairwise-compare list columns a x b per row; returns rid + aggregated scores."""
    ex = (x.select("rid", pl.col(a).list.head(MAX_DIFF).alias("u"), pl.col(b).list.head(MAX_DIFF).alias("v"))
          .filter((pl.col("u").list.len() > 0) & (pl.col("v").list.len() > 0))
          .explode("u").explode("v"))
    if ex.height == 0:
        return None
    u, v = ex["u"].to_list(), ex["v"].to_list()
    return ex.select("rid", "u", "v").with_columns(
        **{name: pl.Series(_cp(u, v, fn)) for name, fn in scorers.items()})


def diff_features(d: pl.DataFrame, vocab: pl.DataFrame) -> pl.DataFrame:
    """Features on the tokens that differ between the two names / addresses.

    Typos ("brotenrs" vs "brothers") signal a true match. Substituted *real*
    words ("technology" vs "technologies", "bioworks" vs "plumbing") signal a
    generated distractor. `vocab` holds the document frequency of every
    core-name token in the split's Source-1 file (no labels involved).
    """
    x = d.select(
        pl.int_range(0, pl.len(), dtype=pl.UInt32).alias("rid"), "q_id", "s1_id",
        _tok("name_k").alias("A"), _tok("name_k_b").alias("B"),
        _tok("addr_num").alias("NA"), _tok("addr_num_b").alias("NB"),
        _tok("addr_c").list.eval(pl.element().filter(~pl.element().is_in(ADMIN_WORDS))).list.join(" ").alias("AC"),
        _tok("addr_c_b").list.eval(pl.element().filter(~pl.element().is_in(ADMIN_WORDS))).list.join(" ").alias("ACb"),
    ).with_columns(
        pl.col("A").list.set_difference("B").list.sort().alias("dA"),
        pl.col("B").list.set_difference("A").list.sort().alias("dB"),
    )
    out = x.select(
        "rid", "q_id", "s1_id",
        pl.col("dA").list.len().cast(pl.Int16).alias("nd_a"),
        pl.col("dB").list.len().cast(pl.Int16).alias("nd_b"),
        pl.col("dA").list.join(" ").alias("dA"),
        pl.col("dB").list.join(" ").alias("dB"),
    )
    both = (out["nd_a"] > 0) & (out["nd_b"] > 0)
    typo = _cp(out["dA"].to_list(), out["dB"].to_list(), fuzz.ratio)
    typo[~both.to_numpy()] = np.nan
    out = out.with_columns(pl.Series("nd_ratio", typo),
                           pl.Series("a_tset_admin", _cp(x["AC"].to_list(), x["ACb"].to_list(), fuzz.token_set_ratio)))

    # best token-to-token typo similarity between the differing words
    cr = _cross(x, "dA", "dB", {"jw": distance.JaroWinkler.normalized_similarity,
                                "lev": distance.Levenshtein.distance})
    if cr is not None:
        agg = cr.group_by("rid").agg(pl.col("jw").max().alias("nd_best_jw"), pl.col("lev").min().alias("nd_min_lev"))
        out = out.join(agg, on="rid", how="left")
    else:
        out = out.with_columns(pl.lit(None, pl.Float32).alias("nd_best_jw"), pl.lit(None, pl.Float32).alias("nd_min_lev"))

    # are the differing words real words (frequent in Source-1) or typos (unseen)?
    for side, col in (("b", "dB"), ("a", "dA")):
        ex = x.select("rid", pl.col(col).list.head(MAX_DIFF).alias("t")).explode("t").drop_nulls("t")
        ex = ex.join(vocab, left_on="t", right_on="tok", how="left").with_columns(
            pl.col("df").fill_null(0).cast(pl.Float32).log1p().alias("ldf"))
        agg = ex.group_by("rid").agg(
            pl.col("ldf").max().alias(f"nd_{side}_ldf_max"),
            pl.col("ldf").min().alias(f"nd_{side}_ldf_min"),
            (pl.col("ldf") >= np.log1p(3)).sum().cast(pl.Int16).alias(f"nd_{side}_known"),
            (pl.col("ldf") == 0).sum().cast(pl.Int16).alias(f"nd_{side}_unseen"))
        out = out.join(agg, on="rid", how="left")

    # house numbers: truncation ("3900" vs "390"), digit typos ("5327" vs "5325"), suffixes ("1056c")
    cr = _cross(x, "NA", "NB", {"lev": distance.Levenshtein.distance})
    if cr is not None:
        cr = cr.with_columns(
            ((pl.col("u") != pl.col("v")) & (pl.col("u").str.starts_with(pl.col("v")) | pl.col("v").str.starts_with(pl.col("u"))))
            .alias("pref"),
            (pl.col("u").str.replace_all(r"\D", "") == pl.col("v").str.replace_all(r"\D", "")).alias("digits_eq"))
        agg = cr.group_by("rid").agg(
            pl.col("lev").min().alias("num_min_lev"),
            pl.col("pref").any().cast(pl.Int8).alias("num_prefix"),
            pl.col("digits_eq").any().cast(pl.Int8).alias("num_digits_eq"))
        out = out.join(agg, on="rid", how="left")
    else:
        out = out.with_columns(pl.lit(None, pl.Float32).alias("num_min_lev"),
                               pl.lit(None, pl.Int8).alias("num_prefix"), pl.lit(None, pl.Int8).alias("num_digits_eq"))
    return out.drop("rid")


def pair_features(cand: pl.DataFrame, s1: pl.DataFrame, q: pl.DataFrame, vocab: pl.DataFrame) -> pl.DataFrame:
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

    out = out.join(diff_features(d, vocab), on=["q_id", "s1_id"], how="left")
    keep = [c for c in cand.columns if c not in ("q_id", "s1_id")]
    out = out.join(cand.select(["q_id", "s1_id"] + keep), on=["q_id", "s1_id"], how="left")
    bool_cols = [c for c, t in out.schema.items() if t == pl.Boolean]
    return out.with_columns(pl.col(bool_cols).cast(pl.Int8))


NON_FEATURES = ("q_id", "s1_id", "label", "is_valid", "fold", "dA", "dB", "country")


# the model sees every numeric column except ids / labels / helper strings
def feature_columns(df):
    return [c for c, t in df.schema.items() if c not in NON_FEATURES and t != pl.Utf8]


def name_vocab(s1: pl.DataFrame) -> pl.DataFrame:
    """Document frequency of core-name tokens in a Source-1 file (unsupervised)."""
    return (s1.select(pl.int_range(0, pl.len()).alias("r"), _tok("name_k").alias("tok"))
            .explode("tok").drop_nulls("tok").unique(["r", "tok"])
            .group_by("tok").agg(pl.len().cast(pl.UInt32).alias("df")))
