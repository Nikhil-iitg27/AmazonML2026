# -*- coding: utf-8 -*-
"""End-to-end runner: data -> blocking -> features -> model -> output.

    # train a model and report validation macro-F0.5
    python src/pipeline.py train --data dataset --out artifacts --sample 200000

    # score the test set into the two required TSVs
    python src/pipeline.py predict --data dataset --artifacts artifacts --out output

Both commands write ``candidate_pairs.tsv`` (the blocking output, exactly what
the classifier ran inference over) and ``matching_results.tsv`` (the final
decisions), as required by the challenge.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import random
import sys
import time

import numpy as np
from tqdm.auto import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from blocking import CHANNEL_ORDER, block_partition, group_by_country
from decide import macro_fbeta, resolve_exclusive, tune_threshold
from features import N_FEATURES, RecordView, add_context, pair_features

csv.field_size_limit(1 << 30)
CHUNK = 20_000   # Source-1 entities scored per streamed batch
LOG = lambda *a: print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


# ---------------------------------------------------------------- data I/O
def read_source(path):
    with open(path, encoding="utf-8", newline="") as f:
        r = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        next(r, None)
        return [(row[0], row[1], row[2], row[3]) for row in r if len(row) >= 4]


def read_truth(path):
    truth = {}
    with open(path, encoding="utf-8", newline="") as f:
        r = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        next(r, None)
        for row in r:
            if not row:
                continue
            ids = row[1].split(",") if len(row) > 1 and row[1].strip() else []
            truth[row[0]] = [i for i in ids if i]
    return truth


def write_id_lists(path, header, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("\t".join(header) + "\n")
        for s1, ids in rows:
            f.write(f"{s1}\t{','.join(ids)}\n")


# ------------------------------------------------- blocking + featurisation
def build_candidates(s1_records, pool_records, n_threads=8):
    """Run blocking per country partition. Returns {s1_id: {cand_id: scores}}."""
    q_by_country = group_by_country(s1_records)
    p_by_country = group_by_country(pool_records)
    out = {}
    for country, queries in q_by_country.items():
        pool = p_by_country.get(country, [])
        if not pool:
            out.update({q[0]: {} for q in queries})
            LOG(f"  {country}: {len(queries)} queries, empty pool -> no candidates")
            continue
        t0 = time.time()
        merged = block_partition(queries, pool, n_threads)
        for qi, q in enumerate(queries):
            out[q[0]] = {pool[pi][0]: sc for pi, sc in merged.get(qi, {}).items()}
        LOG(f"  {country}: {len(queries)} queries x {len(pool)} pool "
            f"-> {sum(len(out[q[0]]) for q in queries)} candidates "
            f"({time.time() - t0:.0f}s)")
    return out


def featurise(candidates, s1_index, pool_index, truth=None,
              view_cache=None, max_cache=300_000):
    """Build the design matrix. Returns (X, y, pair_keys).

    Pool records are cached: a popular record appears in many candidate lists,
    and re-running transliteration and the regex cascade for each occurrence
    dominated the runtime before this cache was added.
    """
    blocks, ys, keys = [], [], []
    if view_cache is None:
        view_cache = {}

    def view(rec_id, index):
        v = view_cache.get(rec_id)
        if v is None:
            # Bounded, because at full scale an unbounded cache over a 5M-record
            # pool is tens of gigabytes. Dropping it wholesale is fine: the cost
            # is a recomputation, not a wrong answer.
            if len(view_cache) >= max_cache:
                view_cache.clear()
            v = view_cache[rec_id] = RecordView(*index[rec_id][1:3])
        return v

    for s1_id, cands in candidates.items():
        if not cands:
            continue
        q = view(s1_id, s1_index)
        gold = set(truth.get(s1_id, ())) if truth is not None else set()
        rows = []
        for cid, sc in cands.items():
            c = view(cid, pool_index)
            rows.append(pair_features(q, c, sc[:3], sc[3:], cid.startswith("S3")))
            keys.append((s1_id, cid))
            if truth is not None:
                ys.append(1 if cid in gold else 0)
        blocks.append(add_context(np.asarray(rows, dtype=np.float32)))
    if not blocks:
        return np.zeros((0, N_FEATURES), np.float32), np.zeros(0, np.int8), []
    X = np.vstack(blocks)
    y = np.asarray(ys, dtype=np.int8) if truth is not None else np.zeros(0, np.int8)
    return X, y, keys


def group_scores(keys, probs):
    scored = {}
    for (s1, cid), p in zip(keys, probs):
        scored.setdefault(s1, []).append((cid, float(p)))
    return scored


# ------------------------------------------------------------------- train
def cmd_train(args):
    d = os.path.join(args.data, "train")
    LOG("loading training data")
    s1 = read_source(os.path.join(d, "train_source1.tsv"))
    s2 = read_source(os.path.join(d, "train_source2.tsv"))
    s3 = read_source(os.path.join(d, "train_source3.tsv"))
    truth = read_truth(os.path.join(d, "train_ground_truth.tsv"))
    LOG(f"S1={len(s1)} S2={len(s2)} S3={len(s3)}")

    rng = random.Random(args.seed)
    if args.sample and args.sample < len(s1):
        s1 = rng.sample(s1, args.sample)
        LOG(f"sub-sampled Source 1 to {len(s1)} entities")

    # Split by entity so no Source-1 record appears in both halves.
    rng.shuffle(s1)
    cut = int(len(s1) * (1 - args.val_frac))
    s1_tr, s1_va = s1[:cut], s1[cut:]

    pool = s2 + s3
    pool_index = {r[0]: r for r in pool}
    s1_index = {r[0]: r for r in s1}

    LOG("blocking (train split)")
    cand_tr = build_candidates(s1_tr, pool, args.threads)
    LOG("blocking (val split)")
    cand_va = build_candidates(s1_va, pool, args.threads)

    LOG("featurising")
    Xtr, ytr, _ = featurise(cand_tr, s1_index, pool_index, truth)
    Xva, yva, kva = featurise(cand_va, s1_index, pool_index, truth)
    LOG(f"train pairs={len(ytr)} pos={int(ytr.sum())} | val pairs={len(yva)}")

    import lightgbm as lgb
    LOG("training LightGBM")
    model = lgb.LGBMClassifier(
        n_estimators=args.trees, learning_rate=0.06, num_leaves=63,
        min_child_samples=50, subsample=0.85, subsample_freq=1,
        colsample_bytree=0.85, reg_lambda=1.0, n_jobs=args.threads,
        random_state=args.seed,
    )
    model.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="average_precision",
              callbacks=[lgb.log_evaluation(50)])

    probs = model.predict_proba(Xva)[:, 1]
    scored = group_scores(kva, probs)
    truth_va = {r[0]: truth.get(r[0], []) for r in s1_va}

    thr, best, curve = tune_threshold(scored, truth_va, exclusive=True)
    _, best_naive, _ = tune_threshold(scored, truth_va, exclusive=False)
    recall_ceiling = _blocking_recall(cand_va, truth_va)

    LOG(f"blocking recall ceiling : {recall_ceiling:.4f}")
    LOG(f"macro-F0.5 (per-entity) : {best_naive:.4f}")
    LOG(f"macro-F0.5 (+exclusive) : {best:.4f}  @ threshold {thr:.3f}")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "model.pkl"), "wb") as f:
        pickle.dump({"model": model, "threshold": thr}, f)
    with open(os.path.join(args.out, "validation.json"), "w") as f:
        json.dump({"threshold": thr, "macro_f05": best,
                   "macro_f05_no_exclusivity": best_naive,
                   "blocking_recall": recall_ceiling,
                   "avg_candidates": float(np.mean([len(v) for v in cand_va.values()])),
                   "threshold_curve": curve,
                   "feature_importance": dict(zip(
                       __import__("features").FEATURE_NAMES,
                       model.feature_importances_.tolist()))}, f, indent=1)
    LOG(f"wrote {args.out}/model.pkl and validation.json")


def _blocking_recall(candidates, truth):
    hit = tot = 0
    for s1, gold in truth.items():
        if not gold:
            continue
        tot += len(gold)
        hit += len(set(gold) & set(candidates.get(s1, {})))
    return hit / tot if tot else 0.0


# ----------------------------------------------------------------- predict
def cmd_predict(args):
    d = os.path.join(args.data, "test")
    LOG("loading test data")
    s1 = read_source(os.path.join(d, "test_source1.tsv"))
    pool = (read_source(os.path.join(d, "test_source2.tsv"))
            + read_source(os.path.join(d, "test_source3.tsv")))
    LOG(f"S1={len(s1)} pool={len(pool)}")

    with open(os.path.join(args.artifacts, "model.pkl"), "rb") as f:
        art = pickle.load(f)
    model, thr = art["model"], art["threshold"]

    s1_index = {r[0]: r for r in s1}
    pool_index = {r[0]: r for r in pool}

    # Streamed, one country partition at a time and one entity chunk at a time.
    # Holding the whole run in memory means a ~114M-row design matrix (~18 GB)
    # plus a comparable candidate dict, which does not fit. Candidates stream
    # straight to disk; only the final match lists (small) are kept.
    os.makedirs(args.out, exist_ok=True)
    cand_path = os.path.join(args.out, "candidate_pairs.tsv")
    final = {}
    n_cand = 0

    q_by_country = group_by_country(s1)
    p_by_country = group_by_country(pool)

    with open(cand_path, "w", encoding="utf-8", newline="") as cf:
        cf.write("source1_entity_id\tcandidate_entity_ids\n")

        for country, queries in q_by_country.items():
            p = p_by_country.get(country, [])
            if not p:
                LOG(f"  {country}: {len(queries)} queries, empty pool")
                for q in queries:
                    cf.write(f"{q[0]}\t\n")
                continue

            t0 = time.time()
            merged = block_partition(queries, p, args.threads, tag=country)
            LOG(f"  {country}: blocked {len(queries)}q x {len(p)}pool "
                f"({time.time() - t0:.0f}s)")

            scored_part, view_cache = {}, {}
            for lo in tqdm(range(0, len(queries), CHUNK),
                           desc=f"  {country} score", unit="chunk"):
                block = {}
                for qi in range(lo, min(lo + CHUNK, len(queries))):
                    cands = {p[pi][0]: sc for pi, sc in merged.get(qi, {}).items()}
                    block[queries[qi][0]] = cands
                    cf.write(f"{queries[qi][0]}\t{','.join(sorted(cands))}\n")
                    n_cand += len(cands)

                X, _, keys = featurise(block, s1_index, pool_index, truth=None,
                                       view_cache=view_cache)
                if not len(X):
                    continue
                probs = model.predict_proba(X)[:, 1]
                for (s1_id, cid), pr in zip(keys, probs):
                    if pr >= thr:                     # pre-filter: keeps this small
                        scored_part.setdefault(s1_id, []).append((cid, float(pr)))

            # Exclusivity only ever applies within a partition, because blocking
            # never proposes a candidate across country boundaries.
            for q in queries:
                scored_part.setdefault(q[0], [])
            final.update(resolve_exclusive(scored_part, thr))
            del merged, scored_part, view_cache

    LOG(f"wrote candidate_pairs.tsv ({n_cand} candidate pairs)")
    write_id_lists(os.path.join(args.out, "matching_results.tsv"),
                   ["source1_entity_id", "matched_entity_ids"],
                   [(r[0], final.get(r[0], [])) for r in s1])
    n_pred = sum(len(v) for v in final.values())
    n_single = sum(1 for r in s1 if not final.get(r[0]))
    LOG(f"wrote matching_results.tsv: {n_pred} links, "
        f"{n_single} predicted singletons ({n_single / len(s1):.2%})")


# -------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Business entity resolution pipeline")
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--data", default="dataset")
    t.add_argument("--out", default="artifacts")
    t.add_argument("--sample", type=int, default=200_000,
                   help="Source-1 entities to train on (0 = all)")
    t.add_argument("--val-frac", type=float, default=0.25)
    t.add_argument("--trees", type=int, default=400)
    t.add_argument("--threads", type=int, default=8)
    t.add_argument("--seed", type=int, default=13)
    t.set_defaults(func=cmd_train)

    p = sub.add_parser("predict")
    p.add_argument("--data", default="dataset")
    p.add_argument("--artifacts", default="artifacts")
    p.add_argument("--out", default="output")
    p.add_argument("--threads", type=int, default=8)
    p.set_defaults(func=cmd_predict)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
