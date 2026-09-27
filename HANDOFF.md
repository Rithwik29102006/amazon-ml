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

## M3 done (per-country tau)
`src/tune_country_tau.py` rebuilds the validation scores and tunes tau per country: US 0.625, India 0.650, France uses the global 0.625. Validation F0.5 goes from 0.97329 to 0.97332. The thresholds are saved as `tau_country` in `matcher_meta.json`, and `predict.py` applies them. New test output: 5,770,269 matches, 99,578 S1 without a match; validator PASS.

## v2 (branch `v2-two-stage-model`), training done, test prediction NOT yet run
Changes vs v1:
- **France normalisation:** departments→regions (Nord→Hauts-de-France …), "N°" dropped, French legal words. US/India normalisation is byte-identical to v1, so their blocking was reused and only France was re-blocked.
- **New pair features (features.py):**
  - words differing between the two names: typo similarity, and whether each is a real Source-1 word or an unseen typo;
  - house-number truncation and digit typos;
  - address similarity without admin words (CDP, City of …).
- **Model (model.py, train.py):**
  - LightGBM (MIT), 4-fold CV over S1 entities, all 19.4M training pairs;
  - out-of-fold token statistics for the differing words;
  - **stage 2:** each S2/S3 record keeps its best candidate and is re-scored with context (competition inside the query, how many records the S1 entity already has, similarity to those records);
  - decision rule chosen by CV among global tau, per-country tau (M3's idea, now built into train.py; tune_country_tau.py removed) and expected-F0.5 set selection.
- **Cross-validated macro F0.5 on all 2.2M training S1:**

| model | F0.5 |
|---|---|
| v1 (HGB, 15% holdout) | 0.9733 |
| v2 stage 1 | 0.9786 |
| v2 stage 2 + global tau | 0.9830 |
| v2 stage 2 + per-country tau | 0.9830 |
| **v2 stage 2 + expected-F0.5 sets (chosen)** | **0.9832** (India 0.9806, US 0.9850; old 15% holdout 0.9832) |

Models are in `work/model/` (s1_fold*.txt, s2_fold*.txt, te_*.parquet, matcher_meta.json).
**Next:** `python predict.py` (~20 min; needs test features from `build_features.py test`) → validator → update docs → rebuild zip.
Until then the v1 outputs (global/per-country tau) remain the submitted version.

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
