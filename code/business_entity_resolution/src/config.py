"""Paths and global settings.

Override the defaults with environment variables:
  BER_DATA_DIR  - folder that contains train/ and test/ (default: <student_resource>/dataset)
  BER_WORK_DIR  - scratch folder for intermediate parquet files and models
  BER_OUT_DIR   - folder for the two submission TSVs
"""
import os
from pathlib import Path

_HERE = Path(__file__).resolve().parent
# src -> business_entity_resolution -> code -> student_resource
_ROOT = _HERE.parents[2]

DATA_DIR = Path(os.environ.get("BER_DATA_DIR", _ROOT / "dataset"))
WORK_DIR = Path(os.environ.get("BER_WORK_DIR", _ROOT / "work"))
OUT_DIR = Path(os.environ.get("BER_OUT_DIR", _ROOT / "output"))

N_JOBS = int(os.environ.get("BER_N_JOBS", os.cpu_count() or 4))
SEED = 42

# fraction of train Source-1 entities held out for validation
VALID_FRAC = 0.15

# Experiment (v4): the test set has ~5.75 Source-2/3 records per Source-1 entity
# versus 4.68 in training (test Source 1 omits many entities whose S2/S3 records are
# still present). Setting this below 100 keeps only that percentage of the training
# Source-1 entities, so the dropped entities' records become unmatched distractors
# (81 -> ~5.75 records per entity). With 81 the leaderboard did not improve
# (0.963 vs 0.9635), so the final submission trains on all entities (100).
TRAIN_S1_KEEP_PCT = int(os.environ.get("BER_TRAIN_S1_KEEP_PCT", 100))


def keep_train_s1(ids):
    """Boolean mask/expression: is this training Source-1 id kept (deterministic hash)?"""
    return (ids.hash(seed=2027) % 100) < TRAIN_S1_KEEP_PCT

SPLITS = ("train", "test")
SOURCES = (1, 2, 3)


def raw_path(split: str, src: int) -> Path:
    return DATA_DIR / split / f"{split}_source{src}.tsv"


def norm_path(split: str, src: int) -> Path:
    return WORK_DIR / "norm" / f"{split}_s{src}.parquet"


def cand_path(split: str) -> Path:
    return WORK_DIR / "cand" / f"{split}_candidates.parquet"


def feat_dir(split: str) -> Path:
    return WORK_DIR / "feat" / split


for d in (WORK_DIR / "norm", WORK_DIR / "cand", WORK_DIR / "feat", WORK_DIR / "model"):
    d.mkdir(parents=True, exist_ok=True)
