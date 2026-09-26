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
