# -*- coding: utf-8 -*-
"""Stage 1 — standardisation.

Everything in this module is deterministic, offline and derived only from the
competition data: no gazetteers, no geocoders, no external lookups.

The three exported keys are the *only* text representations the rest of the
pipeline ever sees:

    name_key(n)   canonical Latin business name, legal suffixes removed
    phon_key(n)   aggressive phonetic fold, whitespace squashed
    addr_key(a)   canonical address, phone numbers and null-markers removed
"""
from __future__ import annotations

import re

# --------------------------------------------------------------------------
# 1. Script folding: Brahmic -> Latin
# --------------------------------------------------------------------------
# Every Indic block in Unicode is laid out parallel to Devanagari (U+0900) on a
# fixed 0x80 stride, so we fold each script onto Devanagari by subtracting its
# block offset and then apply a single Devanagari -> Latin table. This covers
# Bengali, Gurmukhi, Gujarati, Oriya, Tamil, Telugu, Kannada and Malayalam with
# one table instead of eight.
_BLOCK_STARTS = (0x0980, 0x0A00, 0x0A80, 0x0B00, 0x0B80, 0x0C00, 0x0C80, 0x0D00)

_CONS = {
    'क': 'k', 'ख': 'kh', 'ग': 'g', 'घ': 'gh', 'ङ': 'n',
    'च': 'ch', 'छ': 'chh', 'ज': 'j', 'झ': 'jh', 'ञ': 'n',
    'ट': 't', 'ठ': 'th', 'ड': 'd', 'ढ': 'dh', 'ण': 'n',
    'त': 't', 'थ': 'th', 'द': 'd', 'ध': 'dh', 'न': 'n',
    'प': 'p', 'फ': 'ph', 'ब': 'b', 'भ': 'bh', 'म': 'm',
    'य': 'y', 'र': 'r', 'ल': 'l', 'ळ': 'l', 'व': 'v',
    'श': 'sh', 'ष': 'sh', 'स': 's', 'ह': 'h',
    'ऴ': 'l', 'ऱ': 'r', 'ऩ': 'n',
}
_VOWELS = {'अ': 'a', 'आ': 'aa', 'इ': 'i', 'ई': 'ee', 'उ': 'u', 'ऊ': 'oo',
           'ऋ': 'ri', 'ए': 'e', 'ऐ': 'ai', 'ओ': 'o', 'औ': 'au',
           'ऑ': 'o', 'ऎ': 'e', 'ऒ': 'o'}
_MATRA = {'ा': 'a', 'ि': 'i', 'ी': 'ee', 'ु': 'u', 'ू': 'oo', 'ृ': 'ri',
          'े': 'e', 'ै': 'ai', 'ो': 'o', 'ौ': 'au', 'ॉ': 'o', 'ॆ': 'e', 'ॊ': 'o'}
_ANUSVARA = {'ं': 'n', 'ँ': 'n', 'ः': 'h'}
_VIRAMA = '्'
_NUKTA = '़'
_SKIP = {'‌', '‍', _NUKTA}


def _fold_to_devanagari(ch: str) -> str:
    cp = ord(ch)
    for start in _BLOCK_STARTS:
        if start <= cp < start + 0x80:
            return chr(cp - (start - 0x0900))
    return ch


def romanize(text: str) -> str:
    """Phonetic Brahmic -> Latin romanisation. Latin input passes through."""
    if not text or text.isascii():
        return text
    chars = [_fold_to_devanagari(c) for c in text]
    out, i, n = [], 0, len(chars)
    while i < n:
        c = chars[i]
        if c in _SKIP:
            i += 1
            continue
        if c in _CONS:
            out.append(_CONS[c])
            nxt = chars[i + 1] if i + 1 < n else ''
            if nxt == _VIRAMA:                      # dead consonant, no vowel
                i += 2
                continue
            if nxt in _MATRA:
                out.append(_MATRA[nxt]); i += 2; continue
            if nxt in _ANUSVARA:
                out.append('a' + _ANUSVARA[nxt]); i += 2; continue
            out.append('a')                          # inherent schwa
            i += 1
            continue
        if c in _VOWELS:   out.append(_VOWELS[c]);   i += 1; continue
        if c in _ANUSVARA: out.append(_ANUSVARA[c]); i += 1; continue
        if c in _MATRA:    out.append(_MATRA[c]);    i += 1; continue
        if c == _VIRAMA:   i += 1; continue
        if '०' <= c <= '९':                # Devanagari digits
            out.append(chr(ord('0') + ord(c) - 0x0966)); i += 1; continue
        out.append(c)
        i += 1
    return ''.join(out)


# --------------------------------------------------------------------------
# 2. Surface cleanup
# --------------------------------------------------------------------------
_LEGAL_WORDS = (r"llc|l\.l\.c|inc|incorporated|corp|corporation|co|company|"
                r"ltd|limited|pvt|private|llp|plc|sarl|sas|sa|eurl|gmbh|"
                r"proprietor|prop|and sons")
_LEGAL_RE = re.compile(r"(?<![a-z0-9])(?:" + _LEGAL_WORDS + r")(?![a-z0-9])", re.I)

# `c0m` / `corn` are OCR-style corruptions of `com` observed in the data.
_TLD = re.compile(r"\.(?:com|net|org|in|io|fr|biz|info|c0m|corn)(?![a-z])", re.I)
_WWW = re.compile(r"\bwww\.", re.I)
_NULL = re.compile(r"(?<![a-z0-9])(?:null|none|nan|n/?a|not available|unknown)(?![a-z0-9])", re.I)
# A phone number: an optional lead-in label plus >=9 digits. Deliberately
# conservative so that house numbers and PIN codes survive.
_PHONE = re.compile(r"(?:\b(?:ph|phone|mob|mobile|tel|contact)\b[.: ]*)?\+?\d[\d\s\-]{8,}\d",
                    re.I)
_PUNCT = re.compile(r"[^\w\s]+")
_WS = re.compile(r"\s+")


def _base(s: str) -> str:
    if not s:
        return ""
    s = romanize(s)
    s = _WWW.sub(" ", s)
    s = _TLD.sub(" ", s)
    s = s.lower().replace("&", " and ")
    s = _NULL.sub(" ", s)
    s = _PUNCT.sub(" ", s)
    return _WS.sub(" ", s).strip()


def name_key(s: str) -> str:
    """Canonical name: romanised, de-webbed, legal suffixes stripped."""
    return _WS.sub(" ", _LEGAL_RE.sub(" ", _base(s))).strip()


def addr_key(s: str) -> str:
    """Canonical address: romanised, phone numbers and null-markers removed."""
    if not s:
        return ""
    s = romanize(s)
    s = _PHONE.sub(" ", s)
    s = _NULL.sub(" ", s.lower())
    s = _PUNCT.sub(" ", s)
    return _WS.sub(" ", s).strip()


# --------------------------------------------------------------------------
# 3. Phonetic folding
# --------------------------------------------------------------------------
# Applied to *both* sides of every comparison, so it does not need to be
# linguistically correct -- only consistent. It exists to make a romanised
# Indic spelling ("kansalting") collide with the native Latin one
# ("consulting"). '\x01' parks the `ch` digraph so the later c->k rule
# cannot consume it.
_VOWEL_FOLD = [("aa", "a"), ("ee", "i"), ("oo", "u"), ("ai", "e"),
               ("au", "o"), ("ou", "u"), ("ea", "i")]
_CONS_FOLD = [("chh", "\x01"), ("ch", "\x01"), ("sh", "s"), ("ph", "f"),
              ("kh", "k"), ("gh", "g"), ("bh", "b"), ("dh", "d"), ("th", "t"),
              ("ck", "k"), ("qu", "k"), ("q", "k"), ("c", "k"), ("\x01", "c"),
              ("x", "ks"), ("z", "s"), ("w", "v"), ("j", "z"), ("y", "i")]
# Legal suffixes as they survive romanisation *and* the folds above.
_FOLDED_LEGAL = re.compile(
    r"(?<![a-z0-9])(?:pr|pra|prai|privet|prevet|privat|pvt|li|lim|limit|"
    r"limited|limitet|lt|ltd|elelpi|elelsi|ink|inkorporated|korp|"
    r"korporation|kompani|ko|elsi)(?![a-z0-9])")
_DOUBLE = re.compile(r"(.)\1+")
_TRAILING_A = re.compile(r"a(?![a-z0-9])")


def phon_key(s: str) -> str:
    """Aggressive phonetic fold, whitespace removed.

    Squashing whitespace matters more than it looks: the blocking vectoriser
    uses ``char_wb`` n-grams, which do not cross word boundaries. Removing the
    boundaries lets a concatenated web name ("hrsinvestments") and a spaced one
    ("Hrs Investments") share n-grams, and makes the representation invariant
    to word-order transposition.
    """
    s = name_key(s)
    for a, b in _VOWEL_FOLD:
        s = s.replace(a, b)
    for a, b in _CONS_FOLD:
        s = s.replace(a, b)
    s = _DOUBLE.sub(r"\1", s)
    s = _TRAILING_A.sub("", s)            # schwa deletion at token end
    s = _FOLDED_LEGAL.sub(" ", s)
    return re.sub(r"\s+", "", s)


# --------------------------------------------------------------------------
# 4. Structured extraction (used as matching features, not as blocking keys)
# --------------------------------------------------------------------------
_NUMTOK = re.compile(r"\d+")
_US_ZIP = re.compile(r"(?<!\d)(\d{5})(?:-\d{4})?(?!\d)")
_IN_PIN = re.compile(r"(?<!\d)(\d{6})(?!\d)")


def numeric_tokens(addr: str) -> set:
    """House / plot / survey numbers -- the highest-precision address signal."""
    return set(_NUMTOK.findall(romanize(addr or "")))


def postal_codes(addr: str) -> set:
    a = romanize(addr or "")
    return set(_US_ZIP.findall(a)) | set(_IN_PIN.findall(a))


def has_web_name(name: str) -> bool:
    return bool(_TLD.search(name or "") or _WWW.search(name or ""))


def is_non_latin(s: str) -> bool:
    return not (s or "").isascii()
