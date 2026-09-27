# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** Quantum  
**Team Members:** Thoomu Rithwik Reddy (Team Leader), Yama Ram Charan, Patina Mounika, Chinthakayala Dineshwar  
**Submission Date:** 26 September 2026

---

## 1. Executive Summary
We treat entity resolution as an **assignment problem**. Every Source-2/3 record belongs to at most one Source-1 entity, so we search for its best Source-1 match and keep that match only if a classifier is confident enough.
- **Candidate generation:** three sparse TF-IDF "views" (name, address, both) searched with multi-threaded sparse top-k matrix multiplication, followed by a small learned pruner.
- **Matching:** a gradient-boosted classifier on 60+ string, number and ranking features. It keeps at most one Source-1 entity per record, with a threshold tuned directly for macro F0.5.
- **Key ideas:** a **learned transliteration dictionary** plus a **consonant-skeleton encoding**, which link Indic-script names ("स्टार कंसल्टेंट्स") to their English form ("Star Consultants"), and a model that ignores the country label, so it transfers to the unseen France data.

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
**Approach Type:** Blocking (TF-IDF top-k + learned pruner) → gradient-boosted pair classifier → one-to-one assignment with an F0.5-tuned threshold  
**Core Innovation:** Matching across scripts without external data. A token dictionary mined from aligned training pairs, plus a consonant-skeleton code, maps romanised Indic names onto English. A cheap stage-0 pruner shrinks the candidate set about 10× while losing almost no recall.

Pipeline:
```
raw TSV → normalise (names, addresses, skeletons) → transliteration dictionary
       → blocking per country: NAME / ADDR / BOTH TF-IDF top-10 each → union (~21 pairs/query)
       → stage-0 pruner (HGB on cosine & rank features) → ~2 pairs/query  = candidate_pairs.tsv
       → 60+ pair features → HistGradientBoosting classifier → p(match)
       → per S2/S3 record: keep argmax S1 if p ≥ τ → matching_results.tsv
```

**Normalisation** (`normalize.py`):
- Transliterate with unidecode and lowercase.
- Replace "&" with "and", merge dotted initialisms (L.L.C. → llc), strip domain suffixes and handles.
- Split on alias markers (DBA, F/K/A, AKA, "|") and keep each part.
- Remove legal and stop tokens to get the core name.
- Addresses:
  - map abbreviations (road → rd, avenue → ave, rue …) and state/region names to codes (US, India, France);
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
- **Candidate pairs generated:** train 19,411,765, test 21,036,635 (about 2 per S2/S3 record).
- **How we ensured true matches were not lost:**
  - The three complementary views catch name-only matches (missing address), address-only matches (unrelated name) and weak-on-both matches.
  - Skeleton and transliteration features let native-script names match English ones.
  - Character 4-grams tolerate typos.
  - Recall was measured at every stage:

| stage | India | US |
|---|---|---|
| union of 3 views (sample of 30k queries) | 97.6% | 98.9% |
| after stage-0 pruner (share of the above kept) | 99.7% | 99.7% |
| **final candidate set, full training data** (overall 98.1% = 7,494,392 / 7,638,365) | **97.2%** | **98.7%** |

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
- **Other (context / competition):**
  - the three blocking cosines and which views retrieved the pair;
  - the pair's rank among the query's candidates by combined, name and address cosine, and its gaps to the best;
  - margin over the runner-up S1;
  - stage-0 probability and its gap to the query's best;
  - number of candidates per query and per S1, and the pair's rank among the S1's candidates;
  - source (S2 or S3).
  - The **country is deliberately not a feature**, so the model applies unchanged to France.

**Model type:** scikit-learn `HistGradientBoostingClassifier` (BSD-3-Clause, about 1M parameters, far below the 8B limit). Settings: 127 leaves, learning rate 0.08, up to 600 iterations with early stopping, L2 = 1.0.

**Decision rule:** every S2/S3 record keeps only its highest-probability S1 candidate, since the ground truth is one-to-one from the S2/S3 side. That candidate becomes a match only if p ≥ τ.

**Threshold selection method:**
- Validation holds out 15% of S1 training entities by a deterministic hash split. Blocking still runs against the full training S1 index, so the density of competing candidates is realistic.
- τ is chosen by grid search to maximise the exact **macro F0.5 over all validation S1 entities**, singletons included. This gives τ = 0.625.
- The final model is then refit on all training pairs with the same τ.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro, validation split):** **0.9733 (India 0.9708, US 0.9750)**. The upper bound given perfect classification of our candidates is 0.9938.
- **Per-country threshold (M3):** `tune_country_tau.py` tunes τ separately per training country (US 0.625, India 0.650); France has no labels and keeps the global τ = 0.625. Validation macro F0.5 rises from 0.97329 to 0.97332, and the final test output uses these thresholds (5,770,269 matches, 99,578 S1 without a match; validator PASS).
- **Public leaderboard F0.5:** to be added after upload
- **Common false positives (wrong merges):** Pair-level precision on validation is 99.2% (8,908 wrong pairs out of 1.09M predicted). Only 594 of the 18,625 validation singletons received any match. Most wrong merges are generated hard negatives: the same or a near-identical name at a slightly different house number ("3902 Hay Point Landing Rd" vs "390 …", "6912 40th Ave" vs "691 40th Ave"), or the same name stem with a different trailing word ("Heartland Bioworks" vs "Heartland Plumbing"). 1,528 of them are records that truly belong to another S1 entity where the wrong entity won the argmax.
- **Common false negatives (missed matches):** Pair-level recall on validation is 94.4%. Of the 64.6k missed pairs, 21.6k were never retrieved by blocking: mostly generic names ("Family Center", "Housing Trust") with no address, where many S1 entities are equally plausible. The other 43.0k were retrieved but scored below τ. These are typically true matches whose house number was perturbed ("3900 Olympic Blvd" vs "390 …", "5327" vs "5325 Abbeywood Ct"), which look exactly like the hard negatives above, or records with no address and a typo in the name. Because F0.5 favours precision, the tuned threshold deliberately gives these up.

---

## 6. Conclusion
A carefully normalised sparse-retrieval blocker, a learned pruner and a gradient-boosted pair classifier with a one-to-one assignment rule give a fast, fully reproducible entity-resolution pipeline. It runs on a 16 GB laptop and uses no external data. The biggest gains came from:
- noticing that each S2/S3 record belongs to at most one S1 entity;
- linking native-script and English names through a dictionary learned from the training pairs;
- tuning the decision threshold directly on the precision-weighted macro F0.5.

---

## Appendix

### A. Code Artefacts
The code is in `code/business_entity_resolution/`: all source in `src/`, plus `README.md`, `requirements.txt` and `run_all.sh`.

| step | entry point | output |
|---|---|---|
| 1 | `src/prepare.py train test` | `work/norm/*.parquet` (normalised records) |
| 1b | `src/translit.py` | transliteration dictionaries; non-Latin names re-normalised |
| 2 | `src/blocking.py train test` | `work/cand/<split>_candidates.parquet` (pruner in `work/model/pruner.pkl`) |
| 3 | `src/build_features.py train test` | `work/feat/<split>/part_*.parquet` |
| 4 | `src/train.py` | `work/model/matcher.pkl`, `matcher_meta.json` (τ, validation scores) |
| 5 | `src/predict.py` | `output/matching_results.tsv`, `output/candidate_pairs.tsv` |

`bash run_all.sh` runs everything end to end, taking about 3–4 hours on a 10-core, 16 GB laptop. Blocking dominates the runtime.

### B. Additional Results
Validation macro F0.5 vs. threshold τ:

| τ | 0.200 | 0.250 | 0.300 | 0.400 | 0.500 | 0.600 | 0.625 | 0.700 | 0.750 | 0.800 | 0.900 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| macro F0.5 | 0.9549 | 0.9600 | 0.9639 | 0.9690 | 0.9719 | 0.9732 | **0.9733** | 0.9732 | 0.9728 | 0.9721 | 0.9685 |

The curve is flat between τ = 0.60 and 0.70 (0.9732–0.9733), so the choice is robust.

**Test-set prediction statistics** (final model, τ = 0.625):

| country | S1 entities | no match predicted | avg. matches per S1 |
|---|---|---|---|
| France (unseen in training) | 259,452 | 5.2% | 3.38 |
| India | 809,986 | 5.8% | 3.32 |
| US | 663,106 | 5.8% | 3.34 |
| **total** | **1,732,544** | **5.7% (99,241)** | **5,778,694 matches** |

France behaves like the two training countries (the training singleton rate is 5.6%), which suggests the country-agnostic model transfers.

**Validation pair-level metrics:** precision 99.2%, recall 94.4%; singletons that wrongly received a match: 594 / 18,625 (3.2%).

**Runtime** (10-core Apple laptop, 16 GB): normalisation 3 min, transliteration 2 min, blocking train 1 h 49 min and test 39 min, features 7 min, training (validation and final) 1 h 35 min, prediction 15 min.
