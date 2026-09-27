# Business Entity Resolution — reproducible pipeline

Links every Source-2 / Source-3 business record to the Source-1 (reference) entity
it describes, or to nothing. It produces the two submission files:

* `output/matching_results.tsv` — final matches (leaderboard file)
* `output/candidate_pairs.tsv` — the exact candidate set the matcher scores

Only the provided training data is used. There are no external lookups, APIs,
geocoders or pretrained language models. The classifiers are LightGBM gradient-boosted
trees (MIT license, far below 8B parameters). The stage-0 pruner is scikit-learn's
`HistGradientBoostingClassifier` (BSD-3).

## Environment

Python 3.11. Tested on macOS arm64 (10 cores, 16 GB RAM).

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# macOS only: LightGBM needs OpenMP -> brew install libomp
```

## Data layout

The pipeline expects the challenge `dataset/` folder:

```
dataset/train/train_source{1,2,3}.tsv, train_ground_truth.tsv
dataset/test/test_source{1,2,3}.tsv
```

By default it looks for `../../dataset` relative to this folder (the
`student_resource` layout). Point it elsewhere with environment variables:

| variable | default | meaning |
|---|---|---|
| `BER_DATA_DIR` | `<student_resource>/dataset` | folder with `train/` and `test/` |
| `BER_WORK_DIR` | `<student_resource>/work` | intermediate parquet files and models (~8 GB) |
| `BER_OUT_DIR`  | `<student_resource>/output` | where the two TSVs are written |
| `BER_N_JOBS`   | all cores | worker threads/processes |

## Run end to end

```bash
bash run_all.sh
```

or step by step, from `src/`:

| step | command | what it does | approx. time* |
|---|---|---|---|
| 1 | `python prepare.py train test` | normalise names/addresses into parquet | 3 min |
| 1b | `python translit.py` | learn the romanised-Indic → English token dictionary from training pairs, re-normalise non-Latin names | 2 min |
| 2 | `python blocking.py train test` | TF-IDF top-k retrieval (name / address / combined views) and the stage-0 pruner | ~2.5 h |
| 3 | `python build_features.py train test` | 60+ pairwise similarity features | ~7 min |
| 4 | `python train.py` | two-stage LightGBM (4-fold CV over Source-1 entities), out-of-fold token statistics, decision-rule search (global / per-country threshold, expected-F0.5 sets) | ~1.5 h |
| 5 | `python predict.py` | score the test candidates and write both TSVs | ~15 min |

\*On a 10-core laptop.

Validate the output format with the organisers' script:

```bash
python3 ../../utils/validate_submission.py --matching ../../output/matching_results.tsv --candidate ../../output/candidate_pairs.tsv --test-dir ../../dataset/test
```

## Source files

```
src/
  config.py          paths, split fraction, seed
  normalize.py       unidecode, legal-suffix stripping, alias (DBA/FKA) handling,
                     address abbreviations, state codes, consonant "skeleton" encoding
  prepare.py         step 1: raw TSV -> normalised parquet (multiprocessing)
  translit.py        step 1b: learned transliteration dictionary
  blocking.py        step 2: sparse TF-IDF views, sparse_dot_topn top-k, stage-0 pruner
  features.py        pairwise features (rapidfuzz cpdist, token-set overlaps, numbers)
  build_features.py  step 3: features for all candidate pairs, chunked
  evaluate.py        ground truth loading, S1 hash split, macro F0.5
  model.py           shared model code: token statistics, stage-2 features, decision rules
  train.py           step 4: stage-1 + stage-2 LightGBM with 4-fold CV, decision-rule search
  analysis.py        optional: feature ablation + country-transfer experiments
  predict.py         step 5: submission files
```

Every step is deterministic (fixed seeds, hash-based split). The validation
split holds out 15% of the Source-1 training entities.
