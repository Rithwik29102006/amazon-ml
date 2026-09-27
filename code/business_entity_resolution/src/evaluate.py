"""Ground truth loading, train/validation split and the macro F0.5 metric."""
import polars as pl

import config


def load_gt_pairs(all_s1=False):
    """(s1_id, q_id) for every true match in the training ground truth.

    Unless all_s1, only pairs of the kept training Source-1 entities are returned
    (see config.TRAIN_S1_KEEP_PCT): records of dropped entities become unmatched.
    """
    gt = pl.read_csv(config.DATA_DIR / "train" / "train_ground_truth.tsv", separator="\t",
                     quote_char=None, infer_schema_length=0)
    if not all_s1:
        gt = gt.filter(config.keep_train_s1(pl.col("source1_entity_id")))
    return (gt.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(","))
            .explode("matched_entity_ids")
            .filter(pl.col("matched_entity_ids") != "")
            .select(pl.col("source1_entity_id").alias("s1_id"),
                    pl.col("matched_entity_ids").alias("q_id")))


def split_s1(s1_ids: pl.Series):
    """Deterministic hash split of Source-1 ids -> boolean 'is_valid' frame."""
    df = pl.DataFrame({"s1_id": s1_ids})
    return df.with_columns(
        ((pl.col("s1_id").hash(seed=config.SEED) % 10_000) < int(config.VALID_FRAC * 10_000)).alias("is_valid"))


def f05_macro(pred_pairs: pl.DataFrame, gt_pairs: pl.DataFrame, s1_ids) -> float:
    """Macro F0.5 over `s1_ids`; pairs frames have columns s1_id, q_id."""
    ids = pl.DataFrame({"s1_id": s1_ids}).unique()
    pred = pred_pairs.join(ids, on="s1_id", how="semi").unique()
    gt = gt_pairs.join(ids, on="s1_id", how="semi")
    tp = pred.join(gt, on=["s1_id", "q_id"], how="inner").group_by("s1_id").len("tp")
    npred = pred.group_by("s1_id").len("np")
    ngt = gt.group_by("s1_id").len("ng")
    d = (ids.join(tp, on="s1_id", how="left").join(npred, on="s1_id", how="left")
         .join(ngt, on="s1_id", how="left").fill_null(0))
    p = pl.col("tp") / pl.col("np")
    r = pl.col("tp") / pl.col("ng")
    f = (pl.when((pl.col("np") == 0) & (pl.col("ng") == 0)).then(1.0)
         .when(pl.col("tp") == 0).then(0.0)
         .otherwise(1.25 * p * r / (0.25 * p + r)))
    return d.select(f.mean()).item()
