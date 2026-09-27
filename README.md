# Business Entity Resolution — Amazon ML Challenge 2026

Match Source 2/3 business records to Source 1 (deduplicated reference), scored on macro F0.5.

## Pipeline overview

1. **Blocking**: sparse hashed-feature blocking (top-K=60) UNION dense embedding retrieval (top-M=20), per country.
2. **Pair features**: 46 features per candidate pair — name/address similarity (Levenshtein, Jaro-Winkler, token overlap), numeric-code matching, blocking-score statistics, embedding similarity/rank, cross-encoder score.
3. **Cross-encoder reranking**: a fine-tuned cross-encoder scores each entity's top-25 candidates (by embedding similarity), feeding a `ce_score` feature into the classifier.
4. **Classifier**: LightGBM (400 trees) on the 46 features → match probability.
5. **Decision**: threshold + one-to-one assignment (each S2/S3 record → one S1 entity) + per-source caps.

## Two versions, both included

### Version A — submitted, scored 0.969 on the leaderboard
- Bi-encoder: **off-the-shelf** `sentence-transformers/LaBSE` (downloads from Hugging Face, no local file)
- Cross-encoder: fine-tuned locally → `ce_model_100k/` (base: `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`, fine-tuned on 3.67M pairs — positives + blocking-derived hard negatives)
- Classifier: `artifacts_ce100k/config.json` (threshold t=0.8, K=60, M=20, ce_topn=25)
- Retrieval: flat top-K/M (no adaptive selection)

### Version B — offline-validated at 0.9853 (clean, disjoint sample), full-scale run pending
- Bi-encoder: **also fine-tuned locally** → `biencoder_model/` (same base LaBSE, fine-tuned via MultipleNegativesRankingLoss on the same 3.67M-pair positives)
- Cross-encoder: same as Version A (`ce_model_100k/`)
- Classifier: `artifacts_clean_check/config.json` (threshold t=0.75, K=60, M=20, ce_topn=25, adaptive-K blocking: kmin=35, gap=0.15)

**Check the submission portal for which `matching_results.tsv` was actually the last one submitted before the deadline — that's the one to report as the final result. The corresponding `artifacts_*/config.json` above is the source of truth for what settings produced it.**

## What's NOT in this repo (too large for git)

- Model weights (`*.safetensors`) for both `ce_model_100k/` and `biencoder_model/`
- Trained LightGBM model (`artifacts_*/model.txt`)
- Training/validation data (`*.pkl`, `*.tsv`)
- The base LaBSE model (downloads automatically from Hugging Face when run)

These are needed to actually *run* the pipeline but not to understand or read the code. If you need to reproduce the trained models, contact Vedant for the files, or retrain using `src/finetune_crossencoder.py` and `src/finetune_biencoder.py` on the training data.

## Key files in src/

- `train_eval.py` — trains the classifier + runs offline validation (small-sample, with LOCO cross-country generalization check)
- `predict.py` — runs the full pipeline on the real test set, produces submission files
- `finalize.py` — applies threshold + assignment, writes `matching_results.tsv`
- `finetune_crossencoder.py` / `finetune_biencoder.py` — fine-tuning scripts for each model
- `ce_rerank.py` — shortlist selection + cross-encoder scoring logic
- `blocking.py`, `normalize.py`, `pairfeats.py`, `dense_retrieve.py`, `embedder.py` — core pipeline components

Everything else in `src/` (`check_*.py`, `diagnose_*.py`, `eval_*.py`, `bench_*.py`) are diagnostic/analysis scripts used during development, not needed to run the pipeline.

## Results timeline

| Stage | F0.5 (offline) | F0.5 (leaderboard) |
|---|---|---|
| Sparse blocking + LightGBM only | 0.9537 | 0.942 |
| + LaBSE bi-encoder embeddings | 0.9563 | — |
| + fine-tuned cross-encoder reranker | 0.9750–0.9764 | **0.969** |
| + fine-tuned bi-encoder + adaptive-K (Version B) | **0.9853** (clean holdout) | *pending full-scale run* |

Leaderboard context: top teams scored ~0.989–0.991 at competition close.

## What was tried and rejected (worth knowing, don't repeat)

- **Split name/address embeddings** (separate vectors instead of one combined "name | address" string) — worse than combined.
- **Blind widening of K/M candidates** (K=100, M=30) on the base embeddings — recall went up, but F0.5 went down; too much noise reached the classifier. Retrying this *after* fine-tuning the bi-encoder also didn't help.
- **Adaptive-K blocking with too-aggressive kmin (15)** — cut real recall. kmin=35 was better but still didn't clearly beat flat K=60 at full scale.
- **Reverse (pool→S1) blocking** — theoretically sound (find candidates a crowded S1 entity ranks low, but which have few competing S1 matches from their own side), but our implementation reverse-queried the *entire* country pool against S1, which is far too slow at real scale. Needs a smarter design (e.g., restrict to pool records already near *some* forward candidate) — flagged as the top idea for future work.
- **n=100k training sample + trees=800** — more data + bigger trees hurt cross-country (LOCO) generalization; smaller trees=400 was better.

## How to run

Requires: Python 3.10, `torch`, `sentence-transformers`, `lightgbm`, `pandas`, `numpy`, `rapidfuzz`, `unidecode`.

```bash
# train + offline-validate (small sample, fast)
NORM=2 python train_eval.py --n 20000 --M 20 --K 60 \
  --emb_model <path-to-biencoder-or-labse> --ce_model ../ce_model_100k --ce_topn 25 \
  --loco --outdir ../artifacts_run

# full test-set prediction (slow — hours)
NORM=2 python predict.py --split test --art ../artifacts_run --work ../work_test

# produce final submission files
python finalize.py --split test --art ../artifacts_run --work ../work_test \
  --t <threshold-from-config> --out ../../../output
```

`NORM` must match whatever the model was trained with (2, throughout this project).
