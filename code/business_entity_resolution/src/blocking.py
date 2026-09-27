"""Step 2: candidate generation (blocking).

For every Source-2/3 record ("query") we retrieve the most similar Source-1
records ("index") of the same country with three sparse TF-IDF views:

  * NAME : char 3-grams of the space-free core name, char 3-grams of the
           consonant skeleton, and whole core-name words
  * ADDR : address words + address word skeletons + numbers
  * BOTH : NAME and ADDR concatenated (each L2-normalised)

Top-k neighbours per view are computed with sparse_dot_topn (multi-threaded
sparse matmul that keeps only the k best per row). The union of the three
lists is scored by a small stage-0 gradient-boosting "pruner" that only uses
the cheap blocking scores (cosines, ranks, margins); pairs it keeps
(p0 >= 0.001, at most 8 per query) form the final candidate set that the
matching model scores (= candidate_pairs.tsv).

Usage: python blocking.py [train|test ...]
"""
import pickle
import sys
import time

import numpy as np
import polars as pl
import scipy.sparse as sp
from numba import njit, prange
from sparse_dot_topn import sp_matmul_topn

import config

K_NAME, K_ADDR, K_BOTH = 10, 10, 10
MAX_DF_FRAC = 0.005     # drop features that occur in >0.5% of the index docs
MAX_DF_MIN = 500        # ... but never drop features with df below this
ADDR_GRAM_DF = 0.002    # tighter cap for address 4-grams (they are the most numerous)
QUERY_CHUNK = 400_000
PRUNER_PATH = config.WORK_DIR / "model" / "pruner.pkl"

VIEWS = {
    "name": [("name_ns", "char4", "n"), ("name_sns", "char4", "k"), ("name_k", "word", "w")],
    "addr": [("addr_c", "word", "a"), ("addr_s", "word", "s"), ("addr_c", "char4", "c", ADDR_GRAM_DF)],
}


def load(split):
    cols = ["entity_id", "country", "name_k", "name_s", "addr_c", "addr_s", "src"]
    s1 = pl.read_parquet(config.norm_path(split, 1), columns=cols)
    if split == "train":
        s1 = s1.filter(config.keep_train_s1(pl.col("entity_id")))
    q = pl.concat([pl.read_parquet(config.norm_path(split, s), columns=cols) for s in (2, 3)])
    return s1, q


def _add_derived(df):
    return df.with_columns(
        pl.col("name_k").str.replace_all(" ", "").alias("name_ns"),
        pl.col("name_s").str.replace_all(" ", "").alias("name_sns"),
    )


def _tokens(df, col, kind, prefix):
    """Return a frame (row:u32, h:u64) with one row per feature occurrence."""
    base = df.select(pl.int_range(0, pl.len(), dtype=pl.UInt32).alias("row"), pl.col(col).fill_null(""))
    if kind == "word":
        t = base.with_columns(pl.col(col).str.split(" ")).explode(col).rename({col: "t"})
        t = t.filter(pl.col("t").str.len_chars() > 0)
    else:
        n = int(kind[-1])
        s = base.with_columns(("^" + pl.col(col) + "$").alias(col))
        s = s.filter(pl.col(col).str.len_chars() >= n)
        s = s.with_columns(pl.int_ranges(0, pl.col(col).str.len_chars() - (n - 1)).alias("off")).explode("off")
        t = s.select("row", pl.col(col).str.slice(pl.col("off"), n).alias("t"))
    return t.select("row", (pl.lit(prefix + ":") + pl.col("t")).hash(seed=7).alias("h")).unique()


class View:
    """TF-IDF view: vocabulary + idf fitted on the Source-1 index.

    spec entries are (column, kind, prefix[, max_df_frac]); features whose
    document frequency in the index exceeds the cap are dropped (they carry
    little information and dominate the sparse matmul cost).
    """

    def __init__(self, spec):
        self.spec = spec

    def fit(self, index_df):
        n_index = index_df.height
        vocabs = []
        for sp_ in self.spec:
            col, kind, prefix = sp_[:3]
            frac = sp_[3] if len(sp_) > 3 else MAX_DF_FRAC
            max_df = max(MAX_DF_MIN, int(frac * n_index))
            it = _tokens(index_df, col, kind, prefix)
            vocabs.append(it.group_by("h").len().filter(pl.col("len") <= max_df))
        self.vocab = (pl.concat(vocabs).unique("h")
                      .with_columns(pl.int_range(0, pl.len(), dtype=pl.UInt32).alias("fid"),
                                    (np.log((n_index + 1) / (pl.col("len") + 1)) + 1.0)
                                    .cast(pl.Float32).alias("idf"))
                      .select("h", "fid", "idf"))
        return self.transform(index_df)

    def transform(self, df):
        tok = pl.concat([_tokens(df, *sp_[:3]) for sp_ in self.spec])
        j = tok.join(self.vocab, on="h", how="inner")
        m = sp.csr_matrix((j["idf"].to_numpy(), (j["row"].to_numpy(), j["fid"].to_numpy())),
                          shape=(df.height, self.vocab.height), dtype=np.float32)
        m.sum_duplicates()
        m.sort_indices()
        return _l2(m)


def _l2(m):
    norms = np.sqrt(np.asarray(m.multiply(m).sum(axis=1)).ravel())
    norms[norms == 0] = 1.0
    return sp.csr_matrix(sp.diags(1.0 / norms).dot(m), dtype=np.float32)


@njit(parallel=True, cache=True)
def _rowdot(ap, ai, ad, bp, bi, bd, ra, rb):
    out = np.zeros(ra.shape[0], dtype=np.float32)
    for k in prange(ra.shape[0]):
        i, j = ra[k], rb[k]
        p, pe = ap[i], ap[i + 1]
        q, qe = bp[j], bp[j + 1]
        s = 0.0
        while p < pe and q < qe:
            if ai[p] == bi[q]:
                s += ad[p] * bd[q]
                p += 1
                q += 1
            elif ai[p] < bi[q]:
                p += 1
            else:
                q += 1
        out[k] = s
    return out


def rowdot(A, B, ra, rb):
    return _rowdot(A.indptr, A.indices, A.data, B.indptr, B.indices, B.data,
                   ra.astype(np.int64), rb.astype(np.int64))


def topk(Q, IT, k):
    """Return (query_row, index_row) arrays of the top-k neighbours."""
    rows, cols = [], []
    for s in range(0, Q.shape[0], 100_000):
        C = sp_matmul_topn(Q[s:s + 100_000], IT, top_n=k, threshold=0.05,
                           n_threads=config.N_JOBS)
        C = C.tocoo()
        rows.append(C.row.astype(np.int64) + s)
        cols.append(C.col.astype(np.int64))
    return np.concatenate(rows), np.concatenate(cols)


def _both(Xn, Xa):
    m = _l2(sp.hstack([Xn, Xa], format="csr"))
    m.sort_indices()
    return m


def block_country(s1, q, pruner=None):
    s1, q = _add_derived(s1), _add_derived(q)
    t = time.time()
    vn, va = View(VIEWS["name"]), View(VIEWS["addr"])
    Xn_i, Xa_i = vn.fit(s1), va.fit(s1)
    Xb_i = _both(Xn_i, Xa_i)
    ITs = {"name": Xn_i.T.tocsr(), "addr": Xa_i.T.tocsr(), "both": Xb_i.T.tocsr()}
    ks = {"name": K_NAME, "addr": K_ADDR, "both": K_BOTH}
    out = []
    for s0 in range(0, q.height, QUERY_CHUNK):
        qc = q.slice(s0, QUERY_CHUNK)
        Xn_q, Xa_q = vn.transform(qc), va.transform(qc)
        Xb_q = _both(Xn_q, Xa_q)
        Qs = {"name": Xn_q, "addr": Xa_q, "both": Xb_q}
        pairs = []
        for v in ("name", "addr", "both"):
            r, c = topk(Qs[v], ITs[v], ks[v])
            pairs.append(pl.DataFrame({"qi": r, "si": c}).with_columns(pl.lit(1, pl.Int8).alias("v_" + v)))
        cand = (pl.concat(pairs, how="diagonal").group_by("qi", "si")
                .agg(pl.col("v_name").max(), pl.col("v_addr").max(), pl.col("v_both").max())
                .with_columns(pl.col("v_name", "v_addr", "v_both").fill_null(0)))
        qi, si = cand["qi"].to_numpy(), cand["si"].to_numpy()
        cand = cand.with_columns(
            pl.Series("cos_name", rowdot(Xn_q, Xn_i, qi, si)),
            pl.Series("cos_addr", rowdot(Xa_q, Xa_i, qi, si)),
            pl.Series("cos_both", rowdot(Xb_q, Xb_i, qi, si)),
            pl.Series("q_id", qc["entity_id"].to_numpy()[qi]),
            pl.Series("s1_id", s1["entity_id"].to_numpy()[si]),
        ).drop("qi", "si")
        cand = query_rank_features(cand)
        if pruner is not None:
            cand = prune(cand, pruner)
        out.append(cand)
    cand = pl.concat(out)
    print(f"    blocking {time.time() - t:.0f}s, "
          f"{cand.height:,} pairs for {q.height:,} queries", flush=True)
    return cand


def query_rank_features(cand):
    """Features describing how a pair ranks among the other S1 candidates of its query."""
    cand = cand.with_columns(
        pl.col("cos_both").rank("ordinal", descending=True).over("q_id").cast(pl.Int16).alias("q_rank"),
        (pl.col("cos_both") - pl.col("cos_both").max().over("q_id")).alias("q_gap_best"),
        pl.len().over("q_id").cast(pl.Int16).alias("q_ncand"),
        pl.col("cos_name").rank("ordinal", descending=True).over("q_id").cast(pl.Int16).alias("q_rank_name"),
        pl.col("cos_addr").rank("ordinal", descending=True).over("q_id").cast(pl.Int16).alias("q_rank_addr"),
        (pl.col("cos_name") - pl.col("cos_name").max().over("q_id")).alias("q_gap_name"),
        (pl.col("cos_addr") - pl.col("cos_addr").max().over("q_id")).alias("q_gap_addr"),
    )
    # margin of the best pair over the runner-up (for the others: gap to the best)
    second = pl.col("cos_both").sort(descending=True).slice(1, 1).first().over("q_id")
    return cand.with_columns(
        pl.when(pl.col("q_rank") == 1).then(pl.col("cos_both") - second)
        .otherwise(pl.col("q_gap_best")).fill_null(1.0).alias("q_margin"))


PRUNE_FEATS = ["v_name", "v_addr", "v_both", "cos_name", "cos_addr", "cos_both", "q_rank",
               "q_gap_best", "q_ncand", "q_rank_name", "q_rank_addr", "q_gap_name",
               "q_gap_addr", "q_margin"]
PRUNE_MIN_P, PRUNE_MAX_RANK = 0.001, 8


def prune(cand, pruner):
    """Stage-0 filter: keep pairs the cheap model scores >= PRUNE_MIN_P (top PRUNE_MAX_RANK per query)."""
    p0 = pruner.predict_proba(cand.select(PRUNE_FEATS).to_numpy().astype(np.float32))[:, 1]
    cand = cand.with_columns(pl.Series("p0", p0.astype(np.float32)))
    cand = cand.with_columns(pl.col("p0").rank("ordinal", descending=True).over("q_id").cast(pl.Int16).alias("p0_rank"))
    return cand.filter((pl.col("p0") >= PRUNE_MIN_P) & (pl.col("p0_rank") <= PRUNE_MAX_RANK))


def s1_rank_features(cand):
    """Competition among the queries that point at the same Source-1 entity."""
    return cand.with_columns(
        pl.len().over("s1_id").cast(pl.Int32).alias("s_ncand"),
        pl.col("p0").rank("ordinal", descending=True).over("s1_id").cast(pl.Int32).alias("s_rank"),
        pl.len().over("q_id").cast(pl.Int16).alias("q_ncand_kept"),
        (pl.col("p0") - pl.col("p0").max().over("q_id")).alias("p0_gap"),
    )


def fit_pruner(n_per_country=60_000):
    """Train the stage-0 pruner on a sample of *training* queries.

    Queries whose true Source-1 entity is in the validation split are excluded.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier

    from evaluate import load_gt_pairs, split_s1

    gt = load_gt_pairs()
    split = split_s1(gt["s1_id"].unique())
    valid_q = gt.join(split.filter(pl.col("is_valid")), on="s1_id", how="semi")["q_id"]
    s1, q = load("train")
    q = q.filter(~pl.col("entity_id").is_in(valid_q.implode()))
    parts = []
    for c in sorted(s1["country"].unique().to_list()):
        qc = q.filter(pl.col("country") == c)
        qc = qc.sample(min(n_per_country, qc.height), seed=config.SEED)
        parts.append(block_country(s1.filter(pl.col("country") == c), qc))
    cand = (pl.concat(parts).join(gt.with_columns(pl.lit(1, pl.Int8).alias("y")), on=["q_id", "s1_id"], how="left")
            .with_columns(pl.col("y").fill_null(0)))
    model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, random_state=config.SEED)
    model.fit(cand.select(PRUNE_FEATS).to_numpy().astype(np.float32), cand["y"].to_numpy())
    with open(PRUNER_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"[blocking] pruner trained on {cand.height:,} pairs", flush=True)
    return model


def main(splits, only_countries=None):
    """Blocks each (split, country) and checkpoints it to cand/<split>_<country>.parquet.

    Existing checkpoints are reused, so an interrupted run resumes, and different
    machines can run different countries (BER_COUNTRIES=US) and copy the files over.
    The final <split>_candidates.parquet is written once every country is present.
    """
    if PRUNER_PATH.exists():
        with open(PRUNER_PATH, "rb") as f:
            pruner = pickle.load(f)
    else:
        pruner = fit_pruner()
    for split in splits:
        t = time.time()
        s1, q = load(split)
        countries = sorted(set(s1["country"].unique().to_list()) & set(q["country"].unique().to_list()))
        parts = []
        for c in countries:
            part = config.WORK_DIR / "cand" / f"{split}_{c}.parquet"
            parts.append(part)
            if part.exists() or (only_countries and c not in only_countries):
                continue
            s1c = s1.filter(pl.col("country") == c)
            qc = q.filter(pl.col("country") == c)
            print(f"[blocking] {split} {c}: {s1c.height:,} S1 x {qc.height:,} queries", flush=True)
            block_country(s1c, qc, pruner).write_parquet(part)
        missing = [p.name for p in parts if not p.exists()]
        if missing:
            print(f"[blocking] {split}: waiting for {missing}", flush=True)
            continue
        cand = s1_rank_features(pl.concat([pl.read_parquet(p) for p in parts]))
        cand.write_parquet(config.cand_path(split))
        print(f"[blocking] {split}: {cand.height:,} candidate pairs, {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    import os
    only = [c for c in os.environ.get("BER_COUNTRIES", "").split(",") if c] or None
    main(sys.argv[1:] or list(config.SPLITS), only)
