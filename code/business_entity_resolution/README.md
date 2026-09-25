# Business Entity Resolution — pipeline

A four-stage entity resolution pipeline: **standardise → block → match → decide**.

Full methodology, data analysis and design rationale are in
[`docs/entity_resolution_report.pdf`](../../docs/entity_resolution_report.tex).

## Install

```bash
uv venv && uv pip install -r requirements.txt
# or: python -m venv .venv && pip install -r requirements.txt
```

## Reproduce end-to-end

Run both commands from the repository root (the directory containing `dataset/`).

```bash
# 1. Train the matcher and tune the decision threshold on a held-out split.
#    Writes artifacts/model.pkl and artifacts/validation.json.
python code/business_entity_resolution/src/pipeline.py train \
    --data dataset --out artifacts --sample 200000

# 2. Score the test set. Writes output/candidate_pairs.tsv and
#    output/matching_results.tsv.
python code/business_entity_resolution/src/pipeline.py predict \
    --data dataset --artifacts artifacts --out output

# 3. Check the submission format before uploading.
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

`--sample 0` trains on all 2.2M Source-1 entities; the default of 200,000 is
enough to saturate the model and keeps a training run to minutes rather than
hours.

## Source layout

| File | Stage | Responsibility |
| --- | --- | --- |
| `src/normalize.py` | 1 | Brahmic→Latin transliteration; the three canonical keys (`name_key`, `phon_key`, `addr_key`); structured extraction (numeric tokens, postal codes) |
| `src/blocking.py` | 2 | Country partitioning, hashed TF-IDF over character n-grams, sharded sparse top-K, three-channel union |
| `src/features.py` | 3 | 40 pairwise features across retrieval / name / address / shape / context families |
| `src/decide.py` | 4 | F-beta, per-source caps, global exclusivity assignment, threshold tuning |
| `src/pipeline.py` | — | `train` and `predict` entry points, TSV I/O |

## Design notes

Three measured properties of the data drive the architecture. Each is derived in
the report with the supporting numbers.

- **Country is an exact partition.** Zero disagreements in 276,251 sampled true
  pairs. It is applied as a `groupby` over whatever labels are present, never a
  hard-coded set, so the test set's unseen `France` label forms its own
  partition with no code change.
- **No single blocking key suffices.** 14.9% of true pairs share no name token
  with their Source-1 record. The pipeline runs three independent channels and
  unions them; a name-only blocker is capped at ~85% recall.
- **The true assignment is a partition of the pool.** No Source-2/3 record is
  ever claimed by two Source-1 entities across 7,638,365 links, so the decision
  stage resolves candidates globally rather than thresholding pairs
  independently.

## Memory

Blocking uses `HashingVectorizer` rather than `TfidfVectorizer`: the latter
materialises a dictionary of every distinct n-gram before pruning, which
exhausts 17 GB on ~500k address strings. The pool is additionally processed in
400,000-record shards with per-shard top-K merging, so peak memory is set by the
shard size rather than the 10M-record pool. Both are configurable at the top of
`src/blocking.py`.

## Fair play

The pipeline reads only the files under `dataset/`. It performs no network
access, and uses no external gazetteer, geocoder or entity database. The
transliteration tables are derived from Unicode block layout, not from any
external corpus.
