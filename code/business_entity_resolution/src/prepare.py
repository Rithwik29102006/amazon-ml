"""Step 1: read the raw TSVs, normalise every record, write parquet.

Usage: python prepare.py [train|test ...]
"""
import sys
import time
from multiprocessing import Pool

import polars as pl

import config
from normalize import norm_record

OUT_COLS = ["name_c", "name_k", "name_s", "name_parts", "legal", "nonlatin",
            "is_domain", "addr_c", "addr_num", "addr_s"]


def read_tsv(path):
    return pl.read_csv(path, separator="\t", quote_char=None,
                       schema_overrides={c: pl.Utf8 for c in
                                         ["entity_id", "business_name", "business_address", "country"]},
                       infer_schema_length=0)


def _chunk(rows):
    return [norm_record(r) for r in rows]


def normalise_file(split, src, pool):
    t = time.time()
    df = read_tsv(config.raw_path(split, src))
    df = df.with_columns(pl.col("country").fill_null("UNK"),
                         pl.col("business_name").fill_null(""))
    rows = list(zip(df["business_name"].to_list(), df["business_address"].to_list(),
                    df["country"].to_list()))
    step = 20000
    chunks = [rows[i:i + step] for i in range(0, len(rows), step)]
    out = []
    for res in pool.imap(_chunk, chunks, chunksize=1):
        out.extend(res)
    del rows, chunks
    cols = {c: [d[c] for d in out] for c in OUT_COLS}
    del out
    norm = pl.DataFrame(cols, schema={c: (pl.Boolean if c in ("nonlatin", "is_domain") else pl.Utf8)
                                      for c in OUT_COLS})
    df = pl.concat([df, norm], how="horizontal").with_columns(
        pl.lit(src, dtype=pl.Int8).alias("src"))
    df.write_parquet(config.norm_path(split, src))
    print(f"[prepare] {split} s{src}: {df.height:,} rows in {time.time() - t:.0f}s", flush=True)


def main(splits):
    with Pool(config.N_JOBS) as pool:
        for split in splits:
            for src in config.SOURCES:
                normalise_file(split, src, pool)


if __name__ == "__main__":
    main(sys.argv[1:] or list(config.SPLITS))
