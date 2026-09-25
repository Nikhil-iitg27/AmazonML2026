# Business Entity Resolution — ML Challenge 2026

Matching business records across three noisy, multilingual, keyless data sources.
Given 1.7M reference records from Source 1, find every Source 2 / Source 3 record
that refers to the same real-world business.

The official problem statement is in **[DESCRIPTION.md](DESCRIPTION.md)**.
The methodology, data analysis and measured results are in
**[docs/entity_resolution_report.pdf](docs/entity_resolution_report.pdf)**.

---

## Contents

- [Quick start](#quick-start)
- [Entry points](#entry-points)
- [Repository layout](#repository-layout)
- [How the pipeline works](#how-the-pipeline-works)
- [Running at full test scale](#running-at-full-test-scale) ← **read before a long run**
- [Rebuilding the report](#rebuilding-the-report)
- [Submission package](#submission-package)

---

## Quick start

```bash
# 1. Environment (Python 3.12)
uv venv
uv pip install -r code/business_entity_resolution/requirements.txt

# 2. Smoke-test the pipeline end-to-end on a tiny sample (minutes, not days)
python code/business_entity_resolution/src/pipeline.py train \
    --data dataset --out artifacts --sample 2000

# 3. Score the test set (see the scale warning below before running this for real)
python code/business_entity_resolution/src/pipeline.py predict \
    --data dataset --artifacts artifacts --out output

# 4. Validate the submission files
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

Step 2 prints the blocking recall ceiling, the tuned threshold and the
validation macro-F₀.₅, and writes `artifacts/model.pkl` +
`artifacts/validation.json`.

> **Do not use `uv sync` here.** `pyproject.toml` predates the pipeline and does
> not list `rapidfuzz`, `lightgbm` or `sparse-dot-topn`. Because `uv sync` prunes
> the environment to match the lockfile, it will *remove* those three and break
> the pipeline. Either install from `requirements.txt` as above, or add the three
> to `pyproject.toml` and re-lock first.

---

## Entry points

There are exactly **two** executable entry points, plus the notebook.

### 1. `code/business_entity_resolution/src/pipeline.py`

The whole pipeline. Two subcommands.

**`train`** — block, featurise, fit the classifier, tune the decision threshold.

| Flag | Default | Meaning |
| --- | --- | --- |
| `--data` | `dataset` | Root containing `train/` and `test/` |
| `--out` | `artifacts` | Where `model.pkl` and `validation.json` are written |
| `--sample` | `200000` | Source-1 entities to train on; `0` = all 2.2M |
| `--val-frac` | `0.25` | Held-out fraction, split **by entity** |
| `--trees` | `400` | LightGBM estimators |
| `--threads` | `8` | Threads for blocking and LightGBM |
| `--seed` | `13` | RNG seed for sampling and the split |

**`predict`** — block the test set, score it, write the two submission TSVs.

| Flag | Default | Meaning |
| --- | --- | --- |
| `--data` | `dataset` | Root containing `test/` |
| `--artifacts` | `artifacts` | Where to load `model.pkl` from |
| `--out` | `output` | Where the two TSVs are written |
| `--threads` | `8` | Threads for blocking |

Outputs, both tab-separated with one row per Source-1 entity:

- `output/candidate_pairs.tsv` — the blocking output, i.e. exactly what the
  classifier ran inference over
- `output/matching_results.tsv` — the final matches (this is the leaderboard file)

### 2. `utils/validate_submission.py`

Stdlib-only format checker. Supplied by the organisers; run it before every
upload.

| Flag | Meaning |
| --- | --- |
| `-m, --matching` | Path to `matching_results.tsv` |
| `-c, --candidate` | Path to `candidate_pairs.tsv` (optional; enables the subset check) |
| `-t, --test-dir` | Folder with `test_source1/2/3.tsv` |
| `--check-ids` | Also verify every ID exists in the test set. Off by default — loads all S2/S3 IDs (~2 GB) |

Exit `0` = `PASS`. Exit `1` = a numbered list of problems.

### 3. `PrototypeNotebook/AmazonML2026Prototyping.ipynb`

Exploratory Colab notebook (mounts Google Drive, unzips the dataset, eyeballs
matched records). Scratch work — not part of the pipeline, and nothing depends
on it.

---

## Repository layout

```
├── DESCRIPTION.md                  Official problem statement
├── README.md                       This file
├── dataset/                        (gitignored — not in the repo)
│   ├── train/  train_source{1,2,3}.tsv, train_ground_truth.tsv
│   └── test/   test_source{1,2,3}.tsv
├── code/business_entity_resolution/
│   ├── README.md                   Pipeline-specific notes
│   ├── requirements.txt            Pinned dependencies
│   └── src/
│       ├── normalize.py            Stage 1 — transliteration, canonical keys
│       ├── blocking.py             Stage 2 — candidate generation
│       ├── features.py             Stage 3 — pairwise features
│       ├── decide.py               Stage 4 — thresholds, caps, assignment
│       └── pipeline.py             CLI: train / predict
├── utils/validate_submission.py    Submission format checker
├── docs/                           (gitignored — see note below)
│   ├── entity_resolution_report.tex/.pdf
│   ├── results_table.tex           Generated from the measured run
│   └── Documentation_template.md   Organisers' methodology template
├── ProblemStatement/               Original PDFs
└── PrototypeNotebook/              Colab EDA notebook
```

> **Note:** `.gitignore` currently excludes `docs/`, so the report is not
> tracked. If you want it in the submission repo, drop that line.

`main.py` at the root is a leftover `uv init` stub and is unused.

---

## How the pipeline works

Four stages. Each exists to solve a problem the previous one cannot. Full
derivation — with the measurements behind every choice — is in the report.

| Stage | Module | Does |
| --- | --- | --- |
| **1. Standardise** | `normalize.py` | Folds 9 Indic scripts to Latin via Unicode block-offset alignment; strips web/legal/punctuation noise; emits three keys: `name_key`, `phon_key`, `addr_key` |
| **2. Block** | `blocking.py` | Partitions on country, then runs three independent TF-IDF character-n-gram channels and **unions** their top-K |
| **3. Match** | `features.py` | 40 pairwise features across retrieval / name / address / shape / **context** families, scored by LightGBM |
| **4. Decide** | `decide.py` | F₀.₅-tuned threshold, per-source caps, global exclusivity assignment |

Three measured properties of the data drive the design:

1. **Country is an exact partition** — zero disagreements in 276,251 true pairs.
   Applied as a `groupby`, never a hard-coded set, so the test set's unseen
   `France` label needs no code change.
2. **14.9% of true pairs share no name token** — hence three channels unioned,
   not one key. A name-only blocker is capped at ~85% recall.
3. **The address is the strongest single channel** (0.909 recall vs 0.749 for
   the name), because the noise in this dataset attacks names specifically.

Measured on a deliberately pessimistic held-out benchmark (20k entities vs an
828k pool, a 1:41 query-to-pool ratio against ~1:6 in production):

| | |
| --- | --- |
| Blocking union recall | **0.9817** at 65.6 candidates/entity |
| Blocking-only baseline | 0.4403 |
| **Final macro-F₀.₅** | **0.9672** @ threshold 0.775 |

---

## Running at full test scale

**Read this before starting a long run.** The algorithm is validated; the
execution harness is not yet built for full scale.

### Status

- `normalize` / `blocking` / `features` / `decide` — exercised end-to-end at
  20k × 828k, results above.
- `pipeline.py train` / `predict` — **never executed end-to-end.** Only
  import-checked. Run the `--sample 2000` smoke test first.

### Time

Blocking cost is the sum of per-partition query × pool products, which is fixed
by the data (lowering `K` does **not** reduce it — the matmul scores everything
before taking the top-K):

| Partition | S1 queries | Pool | Products | Est. time |
| --- | ---: | ---: | ---: | ---: |
| India | 809,986 | 4,717,565 | 3.82×10¹² | ~87 h |
| US | 663,106 | 3,817,031 | 2.53×10¹² | ~30 h |
| France | 259,452 | 1,434,993 | 0.37×10¹² | ~4.5 h |
| **Total** | **1,732,544** | **9,969,589** | **6.72×10¹²** | **~122 h** |

That is ~5 days at measured burst throughput, and realistically 6–8 days
sustained on a laptop CPU. Featurisation adds ~10 h for the ~114M candidate
pairs; LightGBM scoring is negligible.

Training is not free either: `--sample 200000` against the full 10.3M train pool
is ~34 h. Training needs *realistic negatives*, not the whole pool — a ~1M-record
pool subset that retains all true targets gets there in ~3 h.

### Memory

At full scale `predict` will exhaust 32 GB at four independent points:

1. `build_candidates` accumulates every partition before returning (~23 GB).
2. `featurise` `np.vstack`s one 114M × 40 design matrix (~36 GB peak).
3. `view_cache` in `featurise` is unbounded (~20 GB for 10M pool records).
   Correct at benchmark scale, a liability at full scale.
4. `_retrieve`'s `sparse.vstack` over all pool shards to fit IDF (~6 GB/channel).

### What would fix it

The change is structural, not algorithmic — stream instead of batch:

- Process one country partition at a time.
- Chunk queries (~50k): block → featurise → score → append to both TSVs → free.
  This bounds all four walls at once and lets the ~1.5 GB `candidate_pairs.tsv`
  stream to disk.
- Fit IDF once on a pool sample and persist it.
- Bound `view_cache` to the current chunk.

Once chunked, the work is embarrassingly parallel by partition. To cut the
5 days meaningfully you need sub-blocking *inside* each country (a coarse city or
postal key — the data supports it, worth 10–50×, at a recall cost that must be
measured) or LSH/ANN instead of exact top-K.

### Validation notes

- Expected sizes: `matching_results.tsv` ~90 MB, `candidate_pairs.tsv` ~1.5 GB.
- Run `--check-ids` on the matching file **without** `--candidate` — combining
  them holds ~114M candidate IDs in memory at once.

---

## Rebuilding the report

Requires XeLaTeX (MiKTeX/TeX Live) and a font covering Indic scripts
(`Nirmala UI` on Windows; change `\newfontfamily\indic` in the preamble
otherwise). Run twice so the table of contents resolves:

```bash
cd docs
xelatex entity_resolution_report.tex
xelatex entity_resolution_report.tex
rm -f *.aux *.log *.out *.toc
```

`results_table.tex` is generated from the measured run and is `\input`-ed by the
main document; edit the generator, not the output, if the numbers change.

---

## Submission package

The organisers expect a single zip:

```
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/business_entity_resolution/
│   ├── src/
│   ├── README.md
│   └── requirements.txt
└── Documentation_template.md        filled in (docs/ has the blank template)
```

`code/business_entity_resolution/` in this repo is already laid out to be copied
in as-is. `Documentation_template.md` still needs filling — the content for it is
in the report.

---

## Fair play

The pipeline reads only files under `dataset/`. No network access, no external
gazetteer, geocoder or entity database. The transliteration tables are derived
from Unicode block layout, not from any external corpus. External data lookup is
grounds for disqualification under the challenge rules.
