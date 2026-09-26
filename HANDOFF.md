# Amazon ML Challenge 2026: handoff status (2026-09-26, 15:55 IST)

## Done ✅

| # | Item | Where |
|---|---|---|
| 1 | Python 3.11 env with all packages (`uv venv`) | `student_resource/.venv` |
| 2 | Normaliser: unidecode, legal-suffix stripping, DBA/FKA aliases, address abbreviations, US/IN/FR state codes, consonant-skeleton key | `code/business_entity_resolution/src/normalize.py` |
| 3 | **Step 1 run**: all 6 source files normalised to parquet | `work/norm/*.parquet` |
| 4 | **Step 1b run**: learned Indic → English token dictionary (747 tokens) applied | `src/translit.py`, `work/model/translit_*.json` |
| 5 | Blocking code: 3 TF-IDF views (name / address / both), `sparse_dot_topn` top-10, stage-0 pruner | `src/blocking.py` |
| 6 | Stage-0 pruner **trained** | `work/model/pruner.pkl` |
| 7 | Feature code (67 features, rapidfuzz) smoke-tested | `src/features.py`, `src/build_features.py` |
| 8 | Training / threshold / prediction / metric code written (not run yet) | `src/train.py`, `src/predict.py`, `src/evaluate.py` |
| 9 | `README.md`, `requirements.txt`, `run_all.sh` | `code/business_entity_resolution/` |

Measured on 30–60k-query samples:
- **Blocking recall:** India 97.6%, US 98.9%.
- **Pruner:** keeps 99.75% of the true pairs blocking found, at about 2 candidates per query.

## Not done ❌

1. **Step 2, blocking.** It was stopped mid-run. Train India had finished (26 min) but was not saved, because the old code only saved at the end. The code now **checkpoints per country**, so reruns resume.
   - Estimated run times: train India about 26 min, train US about 60–70 min, test India / US / France about 40 min total.
2. **Step 3:** `python build_features.py train test` (about 25 min)
3. **Step 4:** `python train.py`. This prints the validation F0.5 and the chosen threshold (about 30 min).
4. **Step 5:** `python predict.py` writes `output/matching_results.tsv` and `output/candidate_pairs.tsv`.
5. **Validation:** run `python3 utils/validate_submission.py ...` and upload to the leaderboard.
6. **Documentation:** fill `Documentation_template.md` with real numbers.
7. **Zip:** build `<team>_submission.zip` (`output/`, `code/business_entity_resolution/`, `Documentation_template.md`).
8. **Optional improvements:**
   - Per-country threshold, especially for France.
   - Stage-2 features aggregated per S1 entity.
   - Error analysis.

## Commands (run from `code/business_entity_resolution/src`)

```bash
PY=../../../.venv/bin/python
$PY -W ignore blocking.py train test                  # resumes from checkpoints in work/cand/
BER_COUNTRIES=US $PY -W ignore blocking.py train       # only one country (for splitting work)
$PY -W ignore build_features.py train test
$PY -W ignore train.py
$PY -W ignore predict.py
cd ../../.. && python3 utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

To run on another machine, copy the `code/`, `dataset/` and `work/` folders (work/norm and work/model are about 3 GB). Then either create a venv from `requirements.txt` or rerun `prepare.py` and `translit.py`. If `work/model/pruner.pkl` is missing, `blocking.py` retrains it.

---

## Teammate setup (M2 / M3 / M4)

The repo excludes `dataset/`, `work/` (except the small `work/model/` files) and `.venv/`. Set them up like this:

```bash
git clone https://github.com/Rithwik29102006/amazon-ml.git && cd amazon-ml
# 1. put the challenge data at dataset/train/*.tsv and dataset/test/*.tsv (download from the portal)
# 2. environment (Python 3.11)
uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -r code/business_entity_resolution/requirements.txt
#    (or: python3.11 -m venv .venv && .venv/bin/pip install -r code/business_entity_resolution/requirements.txt)
# 3. regenerate the normalised data (~5 min, deterministic). Do NOT delete work/model/pruner.pkl
cd code/business_entity_resolution/src
../../../.venv/bin/python -W ignore prepare.py train test
../../../.venv/bin/python -W ignore translit.py
# 4. your blocking job, e.g. M2:
BER_COUNTRIES=India ../../../.venv/bin/python -W ignore blocking.py train
#    M3:
../../../.venv/bin/python -W ignore blocking.py test
# 5. upload the resulting work/cand/<split>_<country>.parquet to the shared Drive folder for M1
```

The output log line `waiting for [...]` is expected: it means the other countries' checkpoints come from teammates.

## Work split for 4 members

Parallelism: blocking is split by (split, country). Each person produces `work/cand/<split>_<country>.parquet` and copies it into one shared `work/cand/` folder. Every machine needs `work/norm/` and `work/model/` from this laptop (or reruns steps 1 and 1b, about 5 minutes).

| Member | Task | Output |
|---|---|---|
| **M1 (owner, this laptop)** | `BER_COUNTRIES=US python blocking.py train` (longest, about 70 min). Then collect everyone's checkpoints, run `blocking.py train test` (merge only), `build_features.py`, `train.py`, `predict.py`, validate, and upload to the leaderboard. | `work/cand/train_US.parquet`, final outputs |
| **M2** | `BER_COUNTRIES=India python blocking.py train`, then send the file to M1. Afterwards: error analysis on `work/model/valid_scores.parquet` (false merges and misses by country/source), and propose features. | `work/cand/train_India.parquet`, error-analysis notes |
| **M3** | `python blocking.py test` (all three test countries, about 40 min), then send the files to M1. Afterwards: France sanity check (look at test matches for France; tune a per-country threshold if needed). | `work/cand/test_{US,India,France}.parquet` |
| **M4** | Documentation and packaging: fill `Documentation_template.md` (methodology, blocking, features, model, results), check `code/…/README.md`, build the zip with the exact structure, and run the validator. | filled template, `<team>_submission.zip` |

---

## Prompt to continue in a new Claude chat

Paste this into a new Claude Code session opened in `/Users/rithwikreddy/developer/student_resource`:

```
I am working on the Amazon ML Challenge 2026 Business Entity Resolution task in this folder.
Read README.md (problem statement) and HANDOFF.md (status) first.

Pipeline code is in code/business_entity_resolution/src (prepare -> translit -> blocking ->
build_features -> train -> predict), the Python env is .venv (Python 3.11), and
intermediate data is in work/. Steps 1 and 1b (work/norm, work/model/translit_*.json) and the
stage-0 pruner (work/model/pruner.pkl) are already done. Do NOT rerun them.

Continue from step 2:
1. From code/business_entity_resolution/src, run `../../../.venv/bin/python -W ignore blocking.py train test`
   in the background with output going to work/blocking.log. It checkpoints per country in work/cand/ and resumes.
   Expect about 2.5 hours on this 10-core, 16 GB Mac. Watch memory.
2. Run build_features.py train test, then train.py. Report the validation macro F0.5, the chosen tau and the
   candidate-recall ceiling.
3. Run predict.py, then utils/validate_submission.py on output/, and fix any issue.
4. Fill Documentation_template.md with the real numbers and the method from HANDOFF.md / code docstrings.
5. Build <team_name>_submission.zip exactly as described in README.md "Final Submission Package"
   (do not include dataset/ or work/). Ask me for the team name.
Constraints: no external data/APIs, only MIT/Apache/BSD models of 8B parameters or fewer, TSV outputs with one row per test S1 id.
```
