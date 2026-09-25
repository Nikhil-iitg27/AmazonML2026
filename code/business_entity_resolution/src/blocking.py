# -*- coding: utf-8 -*-
"""Stage 2 — candidate generation.

The comparison space is 1.73M x 9.97M on the test set: roughly 1.7e13 pairs.
Blocking has to cut that to something a classifier can score, without throwing
away the true matches the classifier would have found.

Design, in the order the constraints were discovered from the data:

* **Country is an exact partition.** In 276,251 sampled true pairs the country
  label never disagreed, so partitioning on it is free recall-wise and roughly
  halves the space. Applied as a `groupby` over whatever labels are present,
  never a hard-coded list, so an unseen label (France in the test set) forms its
  own partition automatically.
* **No single key is enough.** ~9% of true pairs share no name token at all, so
  a name-only blocker has a hard recall ceiling well below 1. We run three
  independent channels and take the *union* of their top-K lists.
* **Channels retrieve, they do not decide.** Each channel returns its top-K by
  TF-IDF cosine over character n-grams; stage 3 does the deciding.

Three engineering constraints are baked in, all learned by measurement:

* **Hashing, not a vocabulary.** ``TfidfVectorizer`` materialises a dict of
  every distinct n-gram before pruning to ``max_features``. On ~500k address
  strings that dict alone exhausts 17 GB. ``HashingVectorizer`` projects into a
  fixed 2**20 space in constant memory.
* **DF capping is what makes this tractable.** Measured on 200k real records: a
  query's character 3-grams touch **32.8% of the entire pool**, because the
  char n-gram vocabulary is tiny (~28k grams, mean DF 123) and therefore
  near-stopwords by construction. Exhaustive cosine over them is a full scan
  dressed up as retrieval. Dropping grams above ``max_df`` removes the hot
  postings lists that carry nearly all the cost and almost none of the signal
  after IDF weighting.
* **IDF is computed from the DF pass, not fitted.** That removes the
  ``sparse.vstack`` over every shard that the previous version needed, which was
  a multi-GB copy per channel.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize
from sparse_dot_topn import sp_matmul_topn
from tqdm.auto import tqdm

from normalize import addr_key, name_key, phon_key

# Channel -> (key function, per-channel top-K).
CHANNELS = {
    "name": (lambda n, a: name_key(n), 30),
    "phon": (lambda n, a: phon_key(n), 20),
    "addr": (lambda n, a: addr_key(a), 30),
}
CHANNEL_ORDER = ("name", "phon", "addr")

N_HASH_FEATURES = 1 << 20
SHARD = 400_000

# Drop any n-gram occurring in more than this fraction of the pool. 1.0
# disables the cap and restores the old exhaustive behaviour.
MAX_DF = 0.02


def _vectorizer():
    return HashingVectorizer(analyzer="char_wb", ngram_range=(3, 4),
                             n_features=N_HASH_FEATURES, alternate_sign=False,
                             norm=None, dtype=np.float32)


def _drop_hot(mat, hot):
    """Zero every column flagged in ``hot``, in place."""
    if hot is None or not hot.any():
        return mat
    mask = hot[mat.indices]
    if mask.any():
        mat.data[mask] = 0
        mat.eliminate_zeros()
    return mat


def _weight(counts, idf, hot):
    """Sub-linear TF x IDF, L2-normalised, with hot columns removed."""
    m = counts.tocsr(copy=True)
    _drop_hot(m, hot)
    if m.nnz:
        m.data = (1.0 + np.log(m.data)) * idf[m.indices]
    return normalize(m, norm="l2", copy=False)


def _retrieve(query_texts, pool_texts, top_k, n_threads=8, max_df=MAX_DF,
              progress=None):
    """Top-K pool rows per query by TF-IDF cosine.

    Returns ``(indices, scores)`` arrays of shape ``(n_queries, top_k)``,
    padded with ``-1`` / ``0.0``.
    """
    hv = _vectorizer()
    n_q, n_p = len(query_texts), len(pool_texts)
    shards = range(0, n_p, SHARD)

    # --- Pass 1: vectorise the pool and accumulate document frequency --------
    counts, df = [], np.zeros(N_HASH_FEATURES, dtype=np.int64)
    for i in tqdm(shards, desc=f"{progress} df", leave=False, unit="shard"):
        c = hv.transform(pool_texts[i:i + SHARD]).tocsr()
        counts.append(c)
        # CSR stores one entry per (row, column) pair, so a plain bincount over
        # the column indices *is* the document frequency.
        df += np.bincount(c.indices, minlength=N_HASH_FEATURES)

    hot = (df > max_df * n_p) if max_df < 1.0 else None
    idf = np.log((1.0 + n_p) / (1.0 + df)).astype(np.float32) + 1.0

    # --- Pass 2: weight, match, merge ---------------------------------------
    q_mat = _weight(hv.transform(query_texts).tocsr(), idf, hot)

    best_idx = np.full((n_q, top_k), -1, dtype=np.int64)
    best_sc = np.zeros((n_q, top_k), dtype=np.float32)

    offset = 0
    for chunk in tqdm(counts, desc=f"{progress} match", leave=False, unit="shard"):
        p_mat = _weight(chunk, idf, hot)
        c = sp_matmul_topn(q_mat, p_mat.T.tocsr(), top_n=top_k,
                           sort=True, threshold=0.0, n_threads=n_threads)
        for qi in range(n_q):
            lo, hi = c.indptr[qi], c.indptr[qi + 1]
            if lo == hi:
                continue
            mi = np.concatenate([best_idx[qi], c.indices[lo:hi] + offset])
            ms = np.concatenate([best_sc[qi], c.data[lo:hi]])
            order = np.argsort(-ms)[:top_k]
            best_idx[qi], best_sc[qi] = mi[order], ms[order]
        offset += chunk.shape[0]
        del p_mat, c
    return best_idx, best_sc


def block_partition(queries, pool, n_threads=8, max_df=MAX_DF, tag=""):
    """Candidates for one country partition.

    ``queries`` / ``pool`` are lists of ``(entity_id, name, address)``.

    Returns ``{query_index: {pool_index: [sc_name, sc_phon, sc_addr,
    rk_name, rk_phon, rk_addr]}}``. Ranks are normalised to [0, 1] with 1.0
    meaning "this channel did not retrieve the pair".
    """
    merged = {}
    for ch_i, ch in enumerate(CHANNEL_ORDER):
        key_fn, top_k = CHANNELS[ch]
        q_text = [key_fn(n, a) for _, n, a in queries]
        p_text = [key_fn(n, a) for _, n, a in pool]
        idx, sc = _retrieve(q_text, p_text, top_k, n_threads, max_df,
                            progress=f"{tag}/{ch}")
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
