# Amazon ML Challenge 2026: team Quantum handoff (updated 2026-09-26, 23:40 IST)

Team: Thoomu Rithwik Reddy (leader), Yama Ram Charan, Patina Mounika, Chinthakayala Dineshwar

## Status: first full submission is DONE ✅

| Item | Result |
|---|---|
| Validation macro F0.5 (15% held-out S1) | **0.9733** (India 0.9708, US 0.9750) at tau = 0.625 |
| Oracle F0.5 (perfect classifier on our candidates) | 0.9938, which is the headroom |
| Candidate recall | 98.1% (train India 97.2%, US 98.7%) |
| Validation pair precision / recall | 99.2% / 94.4% |
| Test predictions | 5,778,694 matches; 5.7% of S1 have no match (France 5.2%, India 5.8%, US 5.8%) |
| Official validator | **PASS** (including `--check-ids`) |

Files on Rithwik's laptop (too big for git, so share via Google Drive):
- `output/matching_results.tsv` (97 MB), the leaderboard upload
- `output/candidate_pairs.tsv` (293 MB)
- `Quantum_submission.zip` (164 MB), the final package: output/, code/business_entity_resolution/, Documentation_template.md
- `work/cand/train_candidates.parquet` (684 MB) and `work/cand/test_candidates.parquet` (759 MB), the blocking results (2.5 h to recompute)

In git (branch `final-submission`): all code, the filled `Documentation_template.md`, and `work/model/`
(`pruner.pkl`, `translit_*.json`, the trained `matcher.pkl`, and `matcher_meta.json` with tau and the validation curve).

## Pipeline recap (code: code/business_entity_resolution/src)
1. `prepare.py`: normalise names/addresses (unidecode, legal suffixes, DBA/FKA, abbreviations, state codes, consonant skeleton)
2. `translit.py`: Indic→English token dictionary learned from train pairs (747 tokens)
3. `blocking.py`: 3 TF-IDF views (name/addr/both) with sparse_dot_topn top-10, then a stage-0 HGB pruner (about 2 candidates per query). Checkpoints per country in work/cand/
4. `build_features.py`: 66 features (rapidfuzz cpdist, token/number overlaps, ranks)
5. `train.py`: HistGradientBoosting; each S2/S3 record keeps its argmax S1 if p ≥ tau; tau tuned on macro F0.5
6. `predict.py`: writes the two TSVs

Measured runtimes (10-core Mac, 16 GB): blocking 2.5 h, features 7 min, train 1.5 h, predict 15 min.

## Still to do
1. **Upload** `output/matching_results.tsv` to the leaderboard. Put the score in Documentation_template.md (Section 5, "Public leaderboard F0.5").
2. **Submit** `Quantum_submission.zip` (rebuild it if the docs change).
3. GitHub housekeeping: merge `final-submission` into `main`; close PR #1 (superseded).
4. **Optional improvements** (validation error analysis is in Documentation_template.md, Section 5):
   - Missed matches: 43k were retrieved but rejected (house-number perturbations, missing addresses); 21.6k were blocking misses (generic names without an address).
   - Wrong merges: near-identical names at slightly different house numbers (generated hard negatives).
   - Ideas: per-country tau; S1-level aggregate features (stage 2); house-number edit-distance features; train on all 16.5M train-split rows instead of a 12M sample.
   - Retraining needs only `work/cand/*_candidates.parquet` plus `work/norm/`, not a blocking rerun.

## Setting up another laptop
```bash
git clone https://github.com/Rithwik29102006/amazon-ml.git && cd amazon-ml && git checkout final-submission
# put the challenge data at dataset/train/*.tsv and dataset/test/*.tsv
uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -r code/business_entity_resolution/requirements.txt
cd code/business_entity_resolution/src
../../../.venv/bin/python -W ignore prepare.py train test   # ~3 min, deterministic
../../../.venv/bin/python -W ignore translit.py             # ~2 min
# from Drive: put train_candidates.parquet + test_candidates.parquet into work/cand/
# then build_features.py train test -> train.py -> predict.py (or only predict.py, using the committed matcher.pkl)
```
Note: `predict.py` needs the test features, so run `build_features.py test` first (a few minutes).
