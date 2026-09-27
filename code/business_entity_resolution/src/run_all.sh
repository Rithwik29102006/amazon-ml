#!/usr/bin/env bash
# End-to-end pipeline: raw TSVs -> output/matching_results.tsv + output/candidate_pairs.tsv
# Env overrides: BER_DATA_DIR (folder with train/ and test/), BER_WORK_DIR, BER_OUT_DIR, BER_N_JOBS
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python}
$PY -W ignore prepare.py train test        # 1. normalise names / addresses
$PY -W ignore translit.py                  # 1b. learn + apply Indic->English token dictionary
$PY -W ignore blocking.py train test       # 2. TF-IDF top-k blocking + stage-0 pruner
$PY -W ignore refine_candidates.py         # 2b. stage-0b filter (cheap string similarities) -> final candidate set
$PY -W ignore build_features.py train test # 3. pair features
$PY -W ignore train.py                     # 4. two-stage LightGBM, 4-fold CV, decision-rule search
$PY -W ignore predict.py                   # 5. write the two submission files
# optional: $PY -W ignore analysis.py       # ablation + country-transfer experiments for the write-up
