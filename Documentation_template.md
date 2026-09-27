# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** Quantum  
**Team Members:** Thoomu Rithwik Reddy (Team Leader), Yama Ram Charan, Patina Mounika, Chinthakayala Dineshwar  
**Submission Date:** 27 September 2026

---

## 1. Executive Summary
We treat entity resolution as an **assignment problem**. Every Source-2/3 record belongs to at most one Source-1 entity, so we search for its best Source-1 match and keep that match only if a classifier is confident enough.
- **Candidate generation:** three sparse TF-IDF "views" (name, address, both) searched with multi-threaded sparse top-k matrix multiplication, followed by two small learned filters that use only cheap signals. The final candidate set has **5.9 candidates per Source-1 entity** (1.03 per S2/S3 record), a reduction ratio of 99.99994%, and keeps 98.05% of the true matches.
- **Matching:** a **two-stage LightGBM** model trained with 4-fold cross-validation over Source-1 entities.
  - Stage 1 scores every candidate pair with 89 string, number and ranking features.
  - Stage 2 re-scores each record's best candidate using collective evidence from the other records of the same entity.
  - Each entity then keeps the set of records that maximises the **expected F0.5**.
- **Key ideas:**
  - a **learned transliteration dictionary** plus a **consonant-skeleton encoding**, which link Indic-script names ("स्टार कंसल्टेंट्स") to their English form ("Star Consultants");
  - **"typo or a different word?" features**, which separate true matches from the generator's hard distractors;
  - a model that ignores the country label, so it transfers to the unseen France data.
- **Result:** cross-validated macro F0.5 = **0.9837** on all 2.2M training entities. **Public leaderboard 0.972** after recalibrating the final threshold to the test distribution (Section 5.1).

---

## 2. Methodology

### 2.1 Problem Analysis
Findings from exploring the training data (2.2M S1, 5.0M S2 and 5.3M S3 records):
- **Each S2/S3 record matches at most one S1 entity.** There are 7.64M ground-truth pairs and 7.64M distinct targets. About 74% of S2/S3 records match something, and the other 26% are distractors, many of them near-duplicates.
- **Matches per S1:** 0 matches 5.6% (singletons), 1 match 5.4%, 2–5 matches 78%, 6 or more 11%.
- **Country always agrees** between matched records, so blocking can be done within each country. The test set adds France (259k S1), which has no training labels.
- **Name noise:**
  - typos and leetspeak ("Diam0nd", "SHlVAM");
  - legal-suffix changes (Pvt/Private, LLC, "[LIMITED]");
  - word reordering and extra words ("Limited Bharat Surya Infrastructure Center");
  - alias prefixes ("Fayepyralyra F/K/A Surgical Partners", "Nylaevomira DBA: …");
  - domains and handles ("maurewilliamscolombier.com", "@panchratnaenterprises");
  - honorifics ("Mr").
  - Some true matches carry a completely unrelated name and share only the address.
- **Script:** 23% of India S2 names and 13% of India S3 names are in native script (Devanagari, Tamil, Bengali, Gujarati, Telugu, Kannada …), and state names often are too ("தமிழ்நாடு").
- **Address noise:**
  - reordered components ("IA, Iowa City, 1064 Newton Rd");
  - abbreviations (St/Street, Ave, R./Rue);
  - wrong ordinals ("45nd", "40st") and leading zeros ("001872");
  - literal "null" / "N/A" values;
  - missing street parts, and about 3% missing addresses altogether;
  - state name vs. code (New York/NY, Tamil Nadu/TN);
  - typos ("Wanye", "Townshiip").

### 2.2 Solution Strategy
**Approach Type:** Hybrid. Blocking (TF-IDF top-k + learned pruner) → stage-1 pair classifier → stage-2 collective re-scoring (graph-style evidence per entity) → expected-F0.5 set selection  
**Core Innovations:**
1. Matching across scripts without external data: a token dictionary mined from aligned training pairs plus a consonant-skeleton code map romanised Indic names onto English.
2. Features that tell a typo ("brotenrs") apart from a substituted real word ("technology" → "technologies"): the pattern that separates true matches from the generator's hard distractors.
3. Collective stage 2 plus decision-theoretic set selection that directly optimises the per-entity macro F0.5.

Pipeline:
```
raw TSV → normalise (names, addresses, skeletons) → transliteration dictionary
       → blocking per country: NAME / ADDR / BOTH TF-IDF top-10 each → union (~21 pairs/query)
       → stage-0 pruner (HGB on cosine & rank features) → ~2 pairs/query
       → stage-0b filter (LightGBM + cheap name/address/number similarities) → ~1 pair/query = candidate_pairs.tsv
       → 89 pair features (incl. out-of-fold token statistics) → stage-1 LightGBM → p1
       → per S2/S3 record keep its best S1 candidate → stage-2 LightGBM (+ collective features) → p2
       → per S1 entity: subset of records maximising expected F0.5 → matching_results.tsv
```

**Normalisation** (`normalize.py`):
- Transliterate with unidecode and lowercase.
- Replace "&" with "and", merge dotted initialisms (L.L.C. → llc), strip domain suffixes and handles.
- Split on alias markers (DBA, F/K/A, AKA, "|") and keep each part.
- Remove legal and stop tokens to get the core name.
- Addresses:
  - map abbreviations (road → rd, avenue → ave, rue …) and state/region names to codes (US, India, France);
  - **France only:** map departments to their region (S2/S3 write "Dunkerque, Nord" where S1 writes "Dunkerque, Hauts-de-France"), drop "N°", and treat "groupe", "holding", "ets" as legal words. These rules are country-specific, so US/India normalisation stays byte-identical. On test France, the share of records with a confident best candidate (stage-0 p > 0.5) rose from 69.7% to 80.7%;
  - drop ordinal suffixes and leading zeros;
  - separate number tokens.
- **Consonant skeleton:** ph→f, soft c→s, c/q→k, voiced→unvoiced (b→p, d→t, g→k), remove vowels, h and y, collapse repeats. The Hindi romanisation "praaivett limittedd" and "private limited" both become `prvt lmt`. The same trick maps native-script state names ("tmilllnaattu") to state codes.

**Transliteration dictionary** (`translit.py`):
- For matched training pairs where the S2/S3 name is in native script and has the same token count as the S1 name, align the tokens by position.
- Keep a mapping when it was seen at least twice and is the majority translation. This yields 747 mappings (e.g. sttaar→star, knslttentts→consultants, praa→pvt, li→ltd).
- For honest validation, the dictionary applied to the training files is learned only from the training split of S1 entities.

---

## 3. Candidate Generation (Blocking)

We block within each country (`blocking.py`). S2/S3 records are the queries and S1 records the index.

- **Blocking keys used:** three L2-normalised TF-IDF views (idf fitted on the S1 index):
  - **NAME:** character 4-grams of the space-free core name, character 4-grams of the name skeleton, and whole name words. Space-free grams catch concatenated domains such as "maurewilliamscolombier".
  - **ADDR:** address words, address-word skeletons, and address character 4-grams (for typos).
  - **BOTH:** NAME and ADDR concatenated, which catches records that are only weakly similar on each.
  - Features appearing in more than 0.5% of index documents are dropped (0.2% for address 4-grams). This cut search cost about 50× at a small recall cost.
- **Search:** the top 10 S1 neighbours per view come from `sparse_dot_topn` (a multi-threaded sparse matrix product that keeps only the top-k). The three lists are unioned (about 21 pairs per query), and exact cosines for all three views are computed for every pair.
- **Stage-0 pruner:** a small HistGradientBoosting model on the cheap blocking features:
  - cosines, per-query ranks, gaps to the best candidate, margin over the runner-up, and which views retrieved the pair.
  - It is trained on 120k training queries, excluding validation-split entities.
  - Pairs with p0 ≥ 0.001 (at most 8 per query) are kept.
- **Stage-0b filter** (`refine_candidates.py`, the last blocking stage, added because a smaller candidate set now counts in the ranking):
  - a small LightGBM (63 leaves, 300 rounds) on the stage-0 signals plus nine cheap similarities:
    - token-set, ratio and Jaro-Winkler on the core name;
    - token-set on the name skeleton;
    - token-set and ratio on the address;
    - house-number Jaccard and shared-number count;
    - a missing-address flag.
  - Trained out-of-fold on the same 4 entity folds as the matcher, on 30% of entities; test uses the average of the 4 fold models. Pairs with p ≥ 0.01 are kept.
  - It removed 44% of training candidates (−51% on test) while candidate recall only moved from 98.12% to 98.05%. The downstream score **improved** (CV 0.98323 → 0.98365), because the matcher sees fewer confusing competitors.
- **Candidate pairs generated** (final set = `candidate_pairs.tsv`):

| | train | test |
|---|---|---|
| candidate pairs | 10,915,489 | **10,258,242** |
| candidates per Source-1 entity (mean / median / p95) | 4.95 / 5 / 8 | **5.92 / 6 / 9** |
| candidates per S2/S3 record | 1.06 | 1.03 |
| reduction ratio vs all S1 × S2/S3 pairs | 99.99995% | 99.99994% |
| pairs completeness (true matches kept) | 98.05% (India 97.2%, US 98.7%) | n/a |

  Per country on test: France 6.13, India 5.88, US 5.88 candidates per entity.
- **How we ensured true matches were not lost:**
  - The three complementary views catch name-only matches (missing address), address-only matches (unrelated name) and weak-on-both matches.
  - Skeleton and transliteration features let native-script names match English ones.
  - Character 4-grams tolerate typos.
  - Recall was measured at every stage:

| stage | India | US |
|---|---|---|
| union of 3 views (sample of 30k queries) | 97.6% | 98.9% |
| after stage-0 pruner (share of the above kept) | 99.7% | 99.7% |
| after stage-0 pruner, full training data (98.12% overall) | 97.2% | 98.7% |
| **after stage-0b filter = final candidate set** (98.05% overall, 7,489,554 / 7,638,365) | **97.16%** | **98.65%** |

The remaining misses are mostly generic names ("Family Center", "Housing Trust") with **no address**, where dozens of S1 entities are equally plausible. They are very hard to resolve even with perfect features.

---

## 4. Matching Model

**Features used** (`features.py`; string similarities computed in C++ with `rapidfuzz.process.cpdist`):
- **Name features:**
  - on the core name: ratio, token-set, token-sort, partial ratio and WRatio;
  - Jaro-Winkler, ratio and partial ratio on the space-free name;
  - ratio, token-set and Jaro-Winkler on the consonant skeleton;
  - token-set ratio on the name including legal words;
  - Jaccard of word sets and of skeleton sets, first-token equality;
  - space-free containment (domains and handles);
  - best token-set score over alias parts (DBA / FKA);
  - legal-form agreement (−1 / 0 / 1);
  - token counts and lengths, non-Latin flags, domain flag, alias flag.
- **Address features:**
  - ratio, token-set, token-sort and partial ratio, plus token-set on the address skeleton;
  - word Jaccard and skeleton Jaccard;
  - number-token Jaccard and count of shared numbers, first-number equality, "query numbers ⊆ S1 numbers";
  - missing-address flags and token counts.
- **Difference-token features (new in v2):** computed on the words present in only one of the two names:
  - their number on each side;
  - typo similarity between them (ratio, best Jaro-Winkler, minimum Levenshtein distance);
  - whether each is a **real word** (log document frequency in the Source-1 file of the same split, known / unseen counts) or an unseen **typo**.
  - This targets the generator's distractors, which swap in a real word: "Heartland Bioworks" vs "Heartland Plumbing", or "technology" vs "technologies".
- **Out-of-fold token statistics:** the smoothed rate at which a given differing word coincides with a true match (an inserted "services" is a match 47% of the time, "enterprises" 0.9%, "public" 0%). Computed from the other folds only; min/max/log-count per pair.
- **House-number features (new in v2):** minimum Levenshtein distance between number tokens, truncation / prefix relation ("3900" vs "390"), digits-equal ignoring letter suffixes ("1056c"), and address token-set similarity after removing admin words (CDP, City of, Township …).
- **Other (context / competition):**
  - the three blocking cosines and which views retrieved the pair;
  - the pair's rank among the query's candidates by combined, name and address cosine, and its gaps to the best;
  - margin over the runner-up S1;
  - stage-0 probability and its gap to the query's best;
  - number of candidates per query and per S1, and the pair's rank among the S1's candidates;
  - source (S2 or S3).
  - The **country is deliberately not a feature**, so the model applies unchanged to France.

**Model type:** LightGBM gradient-boosted trees (MIT license; about 30 MB per model, far below the 8B-parameter limit).
- Settings: 255 leaves, learning rate 0.1, min 200 rows per leaf, 80% feature and 70% row bagging, up to 1,500 rounds with early stopping on 2% of training entities.
- **Cross-validation:** 4 folds by a hash of the S1 id. Every S1 entity, with all its candidate pairs, is scored by models that never saw it. Blocking ran against the full training index, so competition between candidates is realistic.
- **Stage 1:** all 10.9M training candidate pairs, 90 features (89 + the stage-0b score), 660–1,030 rounds per fold.
- **Stage 2:** one row per S2/S3 record (its best stage-1 candidate, 10.3M rows). Features: stage-1 probability p1, gap to the query's runner-up, confident candidates in the query, the entity's other records (count, number confident, sum/max of p1, rank), and **collective evidence**, the max/mean token-set similarity of this record's name and address to the entity's other confident records. Also the top 30 stage-1 features by gain. 9.9M rows, 240–350 rounds per fold.
- Test scores are the average of the 4 fold models of each stage.

**Decision rule / threshold selection:** each S2/S3 record is assigned to its best S1 entity only (one-to-one from the S2/S3 side). Three rules were compared on the out-of-fold stage-2 scores, using the exact macro F0.5 over all 2.2M training entities:

| decision rule | CV macro F0.5 |
|---|---|
| global threshold (best τ = 0.70) | 0.98345 |
| per-country threshold (team member M3's idea) | 0.98345 |
| **expected-F0.5 set selection (chosen)** | **0.98365** |

- **Expected-F0.5 set selection:** for each entity, with its records' probabilities sorted p₁ ≥ p₂ ≥ …, keep the top k that maximises E[F₀.₅] ≈ 1.25·Σᵢ≤ₖ pᵢ / (0.25·Σ p + k). Keep nothing when P(no match) = Π(1−pᵢ) is larger, which is worth a full 1.0 on singletons. This is the plug-in approximation of the F-measure-optimal decision (Jansche 2007; Nan Ye et al. 2012).
- **France:** the chosen rule has no country-specific parameters, so it applies unchanged.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro, 4-fold CV over all 2,206,821 training S1 entities):** **0.98365** (India 0.9811, US 0.9854). On the 15% hash holdout used for the first version it is 0.9837, against 0.9733 for v1.
- **Progression:**

| version | model | macro F0.5 |
|---|---|---|
| v1 | HistGradientBoosting, 66 features, global τ (15% holdout) | 0.9733 |
| v1 + M3 | per-country τ | 0.9733 |
| v2 stage 1 | LightGBM on all rows, + difference-token, house-number and out-of-fold token-statistic features (CV) | 0.9786 |
| v2 stage 2 | + collective re-scoring, global τ (CV) | 0.9830 |
| v2 | + expected-F0.5 set selection (CV) | 0.9832 |
| **v3 final** | **+ stage-0b candidate filter: 12.05 → 5.92 candidates per entity (CV)** | **0.9837** |

- **Most important features** (stage 1, by gain):
  - the stage-0b filter score;
  - address similarity without admin words;
  - the **out-of-fold token statistic** of the differing words;
  - legal-form agreement, number Jaccard, **digits-equal** and **admin-free address similarity** (the last two are new in v2).
- **Most important features** (stage 2): the gap to the query's runner-up and p1, then the query's number of confident candidates and the entity's other records.
- **Public leaderboard F0.5:** **0.972** (final file). See Section 5.1 for all leaderboard results.
- **Common false positives (wrong merges):**
  - Out-of-fold pair-level precision is **99.67%** (24,235 wrong pairs out of 7.38M predicted; v1 99.2%).
  - Singletons wrongly given a match fell from 3.2% (v1) to **2.5%** (3,094 / 123,247).
  - The remaining wrong merges are hard distractors: near-identical names at a slightly different house number ("3902 Hay Point Landing Rd" vs "390 …").
  - 5,830 of them are records that truly belong to a *different* S1 entity, which won the assignment.
- **Common false negatives (missed matches):** pair-level recall is **96.2%** (v1 94.4%). Of the 287k missed pairs:
  - **149k** were not in the final candidate set. 53% of these records have no address, and 69% have a near-identical name to their true entity: generic names ("Future Marketing", "Family Center") shared by many entities in the same city, so they cannot be resolved without an address. An exact name+house-number key recovered only 11% of these at +1 candidate per entity, so we left it out.
  - **53k** were retrieved, but another S1 entity scored higher for that record.
  - **85k** were the record's best candidate but were not selected. These are typically true matches with a perturbed house number and no other strong evidence; F0.5 weights precision 2×, so giving them up is the better trade.

### 5.1 Leaderboard results: the test set differs from training

Our first upload (v3, CV 0.9837) scored **0.9635** on the public leaderboard. We compared test and training data without labels:
- **Records per entity:** test has **5.75** S2/S3 records per Source-1 entity in every country, against 4.68 in training. Test Source-1 is 21% smaller than training, while S2/S3 have the same size.
- **Confident matches:** the model still finds ~3.45 confident matches per entity, the same as the true count in training.
- **Borderline records:** test has **2–3× more** of them (stage-2 probability 0.1–0.9) than training.
- **Our reading:** test Source 1 omits many entities whose S2/S3 records are still present ("orphans"). Our strongest features are relative ("clearly the best candidate"), so an orphan's best candidate, a similar entity, can look like a match.

Experiments (same candidates; leaderboard = public F0.5):

| experiment | matches on test | leaderboard |
|---|---|---|
| v3, decision rule chosen by CV (≈ τ 0.62) | 5,897,393 | 0.9635 |
| v4: retrained with 19% of training entities removed to simulate orphans (`TRAIN_S1_KEEP_PCT=81`) | – | 0.963 |
| v3 scores, τ = 0.30 (more lenient) | 6,204,743 | 0.940 |
| **v3 scores, τ = 0.90 (final)** | **5,753,999** | **0.972** |
| v3 scores, τ = 0.97 | 5,620,558 | 0.970 |

- **What this shows:** on test, borderline records are far more often false merges than the training calibration implies. Being more lenient costs a lot, and a stricter threshold gains +0.0085. The curve peaks around τ = 0.9.
- **Final submission:** the same v3 model and candidate set, with a single global threshold of **0.90** (`predict.py`, `FINAL_TAU`). It has no country-specific parameters, so France is treated the same way. The v4 simulation code is kept (`config.TRAIN_S1_KEEP_PCT`, `src/check_v3_sim.py`) but disabled (100%).
- **Lesson:** under this shift, validation on the training distribution overestimates the score by ~0.01–0.02. Next steps would be features that measure *absolute* evidence (how generic a name is, exact house-number agreement), so orphans are rejected without relying on the threshold.

## 6. Conclusion
A carefully normalised sparse-retrieval blocker, a learned pruner and a gradient-boosted pair classifier with a one-to-one assignment rule give a fast, fully reproducible entity-resolution pipeline. It runs on a 16 GB laptop and uses no external data. The biggest gains came from:
- noticing that each S2/S3 record belongs to at most one S1 entity;
- linking native-script and English names through a dictionary learned from the training pairs;
- studying *how* the generator builds distractors: substituted real words and truncated house numbers, answered with difference-token and house-number features (+0.005);
- collective re-scoring using the entity's other records, plus decision-theoretic set selection for the per-entity macro F0.5 (+0.005).

---

## Appendix

### A. Code Artefacts
The code is in `code/business_entity_resolution/`: all source in `src/` (including the end-to-end script `src/run_all.sh`), plus `README.md` and `requirements.txt`.

| step | entry point | output |
|---|---|---|
| 1 | `src/prepare.py train test` | `work/norm/*.parquet` (normalised records) |
| 1b | `src/translit.py` | transliteration dictionaries; non-Latin names re-normalised |
| 2 | `src/blocking.py train test` | `work/cand/<split>_candidates.parquet` (pruner in `work/model/pruner.pkl`) |
| 2b | `src/refine_candidates.py` | final candidate set (stage-0b filter, `work/model/refine.pkl`) |
| 3 | `src/build_features.py train test` | `work/feat/<split>/part_*.parquet` |
| 4 | `src/train.py` (uses `src/model.py`) | `work/model/s1_fold*.txt`, `s2_fold*.txt`, `te_*.parquet`, `matcher_meta.json` (features, decision rule, CV scores) |
| 5 | `src/predict.py` | `output/matching_results.tsv`, `output/candidate_pairs.tsv` |

`bash src/run_all.sh` runs everything end to end, taking about 3.5 hours on a 10-core, 16 GB laptop. Blocking dominates the runtime. `src/analysis.py` (optional) reproduces the ablation and country-transfer experiments.

### B. Additional Results
Out-of-fold macro F0.5 of stage 2 with a single global threshold τ (the chosen expected-F rule reaches 0.98365):

| τ | 0.40 | 0.50 | 0.55 | 0.60 | 0.65 | 0.70 | 0.75 | 0.80 | 0.90 |
|---|---|---|---|---|---|---|---|---|---|
| macro F0.5 | 0.98143 | 0.98264 | 0.98307 | 0.98331 | 0.98342 | **0.98344** | 0.98333 | 0.98305 | 0.98148 |

**Test-set prediction statistics** (final submission: v3 model, τ = 0.90):

| | value |
|---|---|
| S1 entities | 1,732,544 |
| candidates per S1 (candidate_pairs.tsv) | 5.92 (France 6.13, India 5.88, US 5.88) |
| matches | **5,753,999** (3.32 per S1) |
| S1 with no match predicted | 108,249 (6.2%) |

France behaves like the two training countries (the training singleton rate is 5.6%), which suggests the country-agnostic model transfers. Every predicted match is in `candidate_pairs.tsv` (validator PASS with the ID check).

**Out-of-fold pair-level metrics:** precision 99.67%, recall 96.2%.

**Runtime** (10-core Apple laptop, 16 GB):

| step | time |
|---|---|
| normalisation | 3 min |
| transliteration | 2 min |
| blocking (train) | 1 h 49 min |
| blocking (test) | 39 min |
| stage-0b candidate filter | 9 min |
| features | 2 min |
| training (both stages, 4 folds each) | 34 min |
| prediction | 12 min |
