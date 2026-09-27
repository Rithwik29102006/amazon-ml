"""Step 1b: learn a romanised-Indic -> English token dictionary from training
pairs and re-normalise non-Latin names with it.

Source-1 names are English; many Source-2/3 names are the same name written in
Devanagari / Tamil / Bengali / ... script. After unidecode they look like
"sttaar knslttentts" for "Star Consultants". Positionally aligning the tokens of
matched (S1, S2/3) name pairs with equal token counts gives a high-precision
dictionary {"sttaar": "star", "knslttentts": "consultants", ...}.

Only the provided training ground truth is used (no external data).
For honest validation the dictionary applied to the *train* files is learned
from the training split of Source-1 entities only; the dictionary applied to the
*test* files is learned from all training pairs.

Usage: python translit.py
"""
import json
import time
from collections import Counter, defaultdict
from multiprocessing import Pool

import polars as pl

import config
from evaluate import load_gt_pairs, split_s1
from normalize import LEGAL, norm_name, norm_record

MIN_COUNT = 2
MIN_SHARE = 0.5
NAME_FIELDS = ["name_c", "name_k", "name_s", "name_parts", "legal", "nonlatin", "is_domain"]


def _plain(names):
    return [norm_name(n)["name_c"] for n in names]


def learn(pairs: pl.DataFrame):
    """pairs: s1_id, q_id restricted to the Source-1 ids allowed for learning."""
    s1 = pl.read_parquet(config.norm_path("train", 1), columns=["entity_id", "name_c"])
    q = pl.concat([pl.read_parquet(config.norm_path("train", s), columns=["entity_id", "business_name", "nonlatin"])
                   for s in (2, 3)]).filter(pl.col("nonlatin"))
    d = (pairs.join(q, left_on="q_id", right_on="entity_id")
         .join(s1, left_on="s1_id", right_on="entity_id"))
    with Pool(config.N_JOBS) as pool:
        names = d["business_name"].to_list()
        step = 50_000
        rom = sum(pool.map(_plain, [names[i:i + step] for i in range(0, len(names), step)]), [])
    counts = defaultdict(Counter)
    for r, e in zip(rom, d["name_c"].to_list()):
        rt, et = r.split(), e.split()
        if len(rt) != len(et):
            continue
        for a, b in zip(rt, et):
            if a != b and a not in LEGAL:
                counts[a][b] += 1
    tokmap = {}
    for a, c in counts.items():
        b, n = c.most_common(1)[0]
        if n >= MIN_COUNT and n / sum(c.values()) >= MIN_SHARE:
            tokmap[a] = b
    return tokmap


def _renorm(args):
    rows, tokmap = args
    return [norm_record(r, tokmap) for r in rows]


def apply(split, tokmap, pool):
    for src in (2, 3):
        path = config.norm_path(split, src)
        df = pl.read_parquet(path)
        mask = df["business_name"].str.contains(r"[^\x00-\x7F]").fill_null(False)
        sub = df.filter(mask)
        rows = list(zip(sub["business_name"].to_list(), sub["business_address"].to_list(),
                        sub["country"].to_list()))
        step = 20_000
        out = sum(pool.map(_renorm, [(rows[i:i + step], tokmap) for i in range(0, len(rows), step)]), [])
        new = pl.DataFrame({c: [o[c] for o in out] for c in NAME_FIELDS},
                           schema={c: (pl.Boolean if c in ("nonlatin", "is_domain") else pl.Utf8)
                                   for c in NAME_FIELDS})
        sub = sub.with_columns([new[c].alias(c) for c in NAME_FIELDS])
        df = pl.concat([df.filter(~mask), sub]).sort("entity_id")
        df.write_parquet(path)
        print(f"[translit] {split} s{src}: re-normalised {sub.height:,} non-Latin names", flush=True)


def main():
    t = time.time()
    gt = load_gt_pairs(all_s1=True)
    split = split_s1(gt["s1_id"].unique())
    train_ids = split.filter(~pl.col("is_valid"))["s1_id"]
    maps = {
        "train": learn(gt.filter(pl.col("s1_id").is_in(train_ids.implode()))),
        "test": learn(gt),
    }
    for k, m in maps.items():
        (config.WORK_DIR / "model" / f"translit_{k}.json").write_text(json.dumps(m, ensure_ascii=False))
        print(f"[translit] dictionary for {k}: {len(m):,} tokens", flush=True)
    with Pool(config.N_JOBS) as pool:
        for split_name in ("train", "test"):
            apply(split_name, maps[split_name], pool)
    print(f"[translit] done in {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
