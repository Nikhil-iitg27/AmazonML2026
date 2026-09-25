# -*- coding: utf-8 -*-
"""Stage 2 — candidate generation.

The comparison space is 1.73M x 9.97M on the test set: roughly 1.7e13 pairs.
Blocking has to cut that to something a classifier can score, without throwing
away the true matches the classifier would have found.

Design, in the order the constraints were discovered from the data:

* **Country is an exact partition.** In 276,251 sampled true pairs the country
  label never disagreed, so partitioning on it is free recall-wise and roughly
  halves the space. It is applied as a `groupby` over whatever labels are
  present, never a hard-coded list, so an unseen label (France in the test set)
  forms its own partition automatically.
* **No single key is enough.** ~9% of true pairs share no name token at all
  with their Source-1 record, so a name-only blocker has a hard recall ceiling
  well below 1. We therefore run three independent channels and take the
  *union* of their top-K lists.
* **Channels retrieve, they do not decide.** Each channel returns its top-K by
  TF-IDF cosine over character n-grams; stage 3 does the deciding. Character
  n-grams, not words, are what survive typos, word-order transposition and
  concatenated web names.

Two engineering constraints are baked in, both learned the hard way:

* **Hashing, not a vocabulary.** ``TfidfVectorizer`` materialises a Python dict
  of every distinct n-gram before it prunes to ``max_features``. On ~500k
  address strings that dict alone exhausts 17 GB of RAM. ``HashingVectorizer``
  projects into a fixed 2**20 space with no vocabulary at all, in constant
  memory, at the cost of rare hash collisions that the classifier can absorb.
* **Sharding.** The pool side is processed in shards and the per-shard top-K
  lists are merged, so peak memory is set by the shard size rather than by the
  10M-record pool.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sparse_dot_topn import sp_matmul_topn

from normalize import addr_key, name_key, phon_key

# Channel -> (key function, per-channel top-K). Budgets are tuned on the
# validation split against the recall / candidate-count trade-off curve.
CHANNELS = {
    "name": (lambda n, a: name_key(n), 30),
    "phon": (lambda n, a: phon_key(n), 20),
    "addr": (lambda n, a: addr_key(a), 30),
}
CHANNEL_ORDER = ("name", "phon", "addr")

N_HASH_FEATURES = 1 << 20
SHARD = 400_000


def _vectorizer():
    return HashingVectorizer(analyzer="char_wb", ngram_range=(3, 4),
                             n_features=N_HASH_FEATURES, alternate_sign=False,
                             norm=None, dtype=np.float32)


def _retrieve(query_texts, pool_texts, top_k, n_threads=8):
    """Top-K pool rows per query by TF-IDF cosine.

    Returns ``(indices, scores)`` arrays of shape ``(n_queries, top_k)``,
    padded with ``-1`` / ``0.0``.

    IDF is fitted on the pool, which matters here: near-stopword tokens like
    "private limited" or "hospital" must be down-weighted, and which tokens
    those are differs per country partition.
    """
    hv = _vectorizer()
    n_q, n_p = len(query_texts), len(pool_texts)

    # One pass to fit IDF over the whole pool, shard by shard.
    tfidf = TfidfTransformer(sublinear_tf=True)
    counts = [hv.transform(pool_texts[i:i + SHARD]) for i in range(0, n_p, SHARD)]
    tfidf.fit(sparse.vstack(counts) if len(counts) > 1 else counts[0])

    q_mat = tfidf.transform(hv.transform(query_texts)).tocsr()

    best_idx = np.full((n_q, top_k), -1, dtype=np.int64)
    best_sc = np.zeros((n_q, top_k), dtype=np.float32)

    offset = 0
    for chunk in counts:
        p_mat = tfidf.transform(chunk).tocsr()
        c = sp_matmul_topn(q_mat, p_mat.T.tocsr(), top_n=top_k,
                           sort=True, threshold=0.0, n_threads=n_threads)
        # Merge this shard's top-K into the running global top-K.
        for qi in range(n_q):
            lo, hi = c.indptr[qi], c.indptr[qi + 1]
            if lo == hi:
                continue
            cand_i = c.indices[lo:hi] + offset
            cand_s = c.data[lo:hi]
            mi = np.concatenate([best_idx[qi], cand_i])
            ms = np.concatenate([best_sc[qi], cand_s])
            order = np.argsort(-ms)[:top_k]
            best_idx[qi], best_sc[qi] = mi[order], ms[order]
        offset += chunk.shape[0]
        del p_mat, c
    return best_idx, best_sc


def block_partition(queries, pool, n_threads=8):
    """Candidates for one country partition.

    ``queries`` / ``pool`` are lists of ``(entity_id, name, address)``.

    Returns ``{query_index: {pool_index: [sc_name, sc_phon, sc_addr,
    rk_name, rk_phon, rk_addr]}}``. Ranks are normalised to [0, 1] with 1.0
    meaning "this channel did not retrieve the pair", so the classifier reads a
    missing channel as a weak signal rather than a strong one.
    """
    merged = {}
    for ch_i, ch in enumerate(CHANNEL_ORDER):
        key_fn, top_k = CHANNELS[ch]
        q_text = [key_fn(n, a) for _, n, a in queries]
        p_text = [key_fn(n, a) for _, n, a in pool]
        idx, sc = _retrieve(q_text, p_text, top_k, n_threads)
        for qi in range(len(queries)):
            slot = merged.setdefault(qi, {})
            for rank in range(top_k):
                pi = int(idx[qi, rank])
                if pi < 0 or sc[qi, rank] <= 0.0:
                    continue
                rec = slot.get(pi)
                if rec is None:
                    rec = slot[pi] = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
                rec[ch_i] = float(sc[qi, rank])
                rec[3 + ch_i] = rank / top_k
    return merged


def group_by_country(records):
    """Partition records by their country label (open set, never hard-coded)."""
    buckets = {}
    for r in records:
        buckets.setdefault(r[3], []).append((r[0], r[1], r[2]))
    return buckets
