# -*- coding: utf-8 -*-
"""Stage 4 — turning pair probabilities into a submission.

Scoring a pair independently is not the same as answering the question the
leaderboard asks. Three properties of the ground truth change the decision
rule, and all three were measured on the training labels rather than assumed:

1. **The assignment is a partition.** Across 7,638,365 links in
   ``train_ground_truth.tsv``, no Source-2/3 record is ever claimed by two
   different Source-1 entities (0 collisions). A candidate is therefore a
   *contested resource*: if two Source-1 records both want it, at most one is
   right. `resolve_exclusive` enforces that globally.
2. **Match lists are small and bounded.** The largest true list has 11 members;
   at most 5 come from Source 2 and 6 from Source 3.
3. **The metric is macro-F0.5 with singletons included.** Per entity, a false
   positive costs more than a false negative, and 5.58% of entities are
   singletons where the correct answer is the empty list -- worth a full 1.0
   each, and 0.0 if we predict anything at all.
"""
from __future__ import annotations

from collections import defaultdict

MAX_PER_SOURCE = {"S2": 5, "S3": 6}
MAX_TOTAL = 11


def fbeta(pred: set, gold: set, beta: float = 0.5) -> float:
    """Per-entity F_beta, with the empty/empty case scoring 1.0."""
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    tp = len(pred & gold)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(gold)
    b2 = beta * beta
    return (1 + b2) * p * r / (b2 * p + r)


def macro_fbeta(predictions: dict, truth: dict, beta: float = 0.5) -> float:
    """Macro-average over every Source-1 entity in ``truth``."""
    if not truth:
        return 0.0
    return sum(fbeta(set(predictions.get(k, ())), set(v), beta)
               for k, v in truth.items()) / len(truth)


def select_per_entity(scored, threshold):
    """Keep candidates above ``threshold``, respecting the per-source caps.

    ``scored`` maps ``source1_id -> [(candidate_id, probability), ...]``.
    """
    out = {}
    for s1, cands in scored.items():
        kept = sorted((c for c in cands if c[1] >= threshold),
                      key=lambda x: -x[1])
        chosen, per_src = [], defaultdict(int)
        for cid, prob in kept:
            src = cid[:2]
            if per_src[src] >= MAX_PER_SOURCE.get(src, MAX_TOTAL):
                continue
            if len(chosen) >= MAX_TOTAL:
                break
            chosen.append(cid)
            per_src[src] += 1
        out[s1] = chosen
    return out


def resolve_exclusive(scored, threshold):
    """Globally assign each candidate to at most one Source-1 entity.

    A greedy pass in descending probability. Because the ground-truth
    assignment is a strict partition, a candidate claimed by a
    higher-confidence entity cannot also belong to a lower-confidence one; the
    greedy order is the natural relaxation of the optimal assignment and is the
    precision-safe choice under F0.5.
    """
    flat = [(prob, s1, cid)
            for s1, cands in scored.items()
            for cid, prob in cands if prob >= threshold]
    flat.sort(key=lambda t: -t[0])

    claimed = set()
    taken = defaultdict(lambda: defaultdict(int))
    out = {s1: [] for s1 in scored}
    for prob, s1, cid in flat:
        if cid in claimed:
            continue
        src = cid[:2]
        if taken[s1][src] >= MAX_PER_SOURCE.get(src, MAX_TOTAL):
            continue
        if len(out[s1]) >= MAX_TOTAL:
            continue
        out[s1].append(cid)
        taken[s1][src] += 1
        claimed.add(cid)
    return out


def tune_threshold(scored, truth, lo=0.05, hi=0.95, steps=37, exclusive=True):
    """Grid-search the probability threshold that maximises macro-F0.5."""
    select = resolve_exclusive if exclusive else select_per_entity
    best = (-1.0, 0.5)
    curve = []
    for i in range(steps):
        t = lo + (hi - lo) * i / (steps - 1)
        score = macro_fbeta(select(scored, t), truth)
        curve.append((round(t, 4), round(score, 5)))
        if score > best[0]:
            best = (score, t)
    return best[1], best[0], curve
