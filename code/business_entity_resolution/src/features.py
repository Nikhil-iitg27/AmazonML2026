# -*- coding: utf-8 -*-
"""Stage 3 — pairwise feature engineering.

Three families of features, in increasing order of what they cost to compute:

1. **Retrieval features** — the cosine score and rank each blocking channel
   gave the pair. Free: blocking already computed them.
2. **String-similarity features** — edit / token / n-gram similarities on the
   canonical name and address.
3. **Context features** — how a candidate compares with the *other* candidates
   of the same Source-1 record. These carry most of the precision: an absolute
   name similarity of 0.82 means something very different when the runner-up
   scored 0.81 than when it scored 0.40.
"""
from __future__ import annotations

import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

from normalize import (addr_key, has_web_name, is_non_latin, name_key,
                       numeric_tokens, phon_key, postal_codes)

FEATURE_NAMES = [
    # retrieval
    "sc_name", "sc_phon", "sc_addr", "n_channels",
    "rk_name", "rk_phon", "rk_addr",
    # name similarity
    "nm_token_set", "nm_token_sort", "nm_partial", "nm_jw", "nm_ratio",
    "nm_jaccard", "nm_containment", "nm_3gram_jac", "nm_3gram_cont",
    "ph_ratio", "ph_jw", "ph_3gram_jac",
    # address similarity
    "ad_token_set", "ad_jaccard", "ad_containment", "ad_3gram_jac",
    "ad_num_jaccard", "ad_num_containment", "ad_num_exact", "ad_postal_match",
    # shape / provenance
    "len_ratio_name", "len_ratio_addr", "tgt_addr_empty", "tgt_non_latin",
    "tgt_web_name", "is_s3",
    # context
    "ctx_cand_count", "ctx_name_margin", "ctx_name_zscore", "ctx_name_rank",
    "ctx_combo_margin", "ctx_combo_rank", "ctx_is_best",
]
N_FEATURES = len(FEATURE_NAMES)


def _grams(s, n=3):
    return {s[i:i + n] for i in range(max(1, len(s) - n + 1))} if s else set()


def _jac(a, b):
    if not a and not b:
        return 0.0
    u = len(a | b)
    return len(a & b) / u if u else 0.0


def _cont(a, b):
    m = min(len(a), len(b))
    return len(a & b) / m if m else 0.0


class RecordView:
    """Pre-computed normalisation for one record; built once, reused per pair."""
    __slots__ = ("name", "addr", "nk", "ak", "pk", "ntok", "atok",
                 "n3", "a3", "p3", "nums", "pins", "nonlatin", "web")

    def __init__(self, name, addr):
        self.name, self.addr = name or "", addr or ""
        self.nk = name_key(self.name)
        self.ak = addr_key(self.addr)
        self.pk = phon_key(self.name)
        self.ntok = set(self.nk.split())
        self.atok = set(self.ak.split())
        self.n3 = _grams(self.nk.replace(" ", ""))
        self.a3 = _grams(self.ak.replace(" ", ""))
        self.p3 = _grams(self.pk)
        self.nums = numeric_tokens(self.addr)
        self.pins = postal_codes(self.addr)
        self.nonlatin = is_non_latin(self.name)
        self.web = has_web_name(self.name)


def pair_features(q: RecordView, c: RecordView, scores, ranks, is_s3):
    """Feature vector for one (Source-1, candidate) pair, minus context."""
    sc_name, sc_phon, sc_addr = scores
    rk_name, rk_phon, rk_addr = ranks

    nm_ts = fuzz.token_set_ratio(q.nk, c.nk) / 100.0
    nm_so = fuzz.token_sort_ratio(q.nk, c.nk) / 100.0
    nm_pa = fuzz.partial_ratio(q.nk, c.nk) / 100.0
    nm_jw = JaroWinkler.similarity(q.nk, c.nk)
    nm_ra = fuzz.ratio(q.nk, c.nk) / 100.0

    ad_ts = fuzz.token_set_ratio(q.ak, c.ak) / 100.0 if c.ak else 0.0

    return [
        sc_name, sc_phon, sc_addr,
        float((sc_name > 0) + (sc_phon > 0) + (sc_addr > 0)),
        rk_name, rk_phon, rk_addr,

        nm_ts, nm_so, nm_pa, nm_jw, nm_ra,
        _jac(q.ntok, c.ntok), _cont(q.ntok, c.ntok),
        _jac(q.n3, c.n3), _cont(q.n3, c.n3),
        fuzz.ratio(q.pk, c.pk) / 100.0, JaroWinkler.similarity(q.pk, c.pk),
        _jac(q.p3, c.p3),

        ad_ts, _jac(q.atok, c.atok), _cont(q.atok, c.atok), _jac(q.a3, c.a3),
        _jac(q.nums, c.nums), _cont(q.nums, c.nums),
        float(bool(q.nums) and q.nums == c.nums),
        float(bool(q.pins & c.pins)),

        len(c.nk) / max(1, len(q.nk)), len(c.ak) / max(1, len(q.ak)),
        float(not c.ak), float(c.nonlatin), float(c.web), float(is_s3),
    ]


_CTX_START = FEATURE_NAMES.index("ctx_cand_count")
_NM_TS = FEATURE_NAMES.index("nm_token_set")
_AD_TS = FEATURE_NAMES.index("ad_token_set")


def add_context(block: np.ndarray) -> np.ndarray:
    """Append the seven context features to one Source-1 record's candidates.

    ``block`` holds the non-context features for every candidate of a single
    Source-1 record, so the statistics below are computed *within* that record.
    """
    n = block.shape[0]
    out = np.empty((n, N_FEATURES), dtype=np.float32)
    out[:, :_CTX_START] = block

    nm = block[:, _NM_TS]
    combo = 0.65 * nm + 0.35 * block[:, _AD_TS]

    def margin_rank(v):
        order = np.argsort(-v)
        rank = np.empty(n, dtype=np.float32)
        rank[order] = np.arange(n, dtype=np.float32)
        best = v.max()
        second = np.partition(v, -2)[-2] if n > 1 else 0.0
        # A candidate's margin is measured against the best *other* candidate.
        margin = np.where(v >= best, v - second, v - best)
        return margin.astype(np.float32), rank

    nm_margin, nm_rank = margin_rank(nm)
    cb_margin, cb_rank = margin_rank(combo)
    sd = nm.std()

    out[:, _CTX_START + 0] = n
    out[:, _CTX_START + 1] = nm_margin
    out[:, _CTX_START + 2] = (nm - nm.mean()) / sd if sd > 1e-6 else 0.0
    out[:, _CTX_START + 3] = nm_rank
    out[:, _CTX_START + 4] = cb_margin
    out[:, _CTX_START + 5] = cb_rank
    out[:, _CTX_START + 6] = (cb_rank == 0).astype(np.float32)
    return out
