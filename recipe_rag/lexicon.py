"""Corpus-derived spelling-variant lexicon (replaces v1's hard-coded expansions).

Problem
-------
Moroccan dish names are Darija/Arabic words transliterated through French or
English spelling conventions, so the same dish is written many ways:
tanjia / tangia, bastilla / pastilla / bstilla, msemen / msemmen,
tajine / tagine, chebakia / chebakya. The corpus only contains *some* of these
spellings (it has "tangia" 5 times and "tanjia" 0 times), so a user typing
"tanjia" gets no lexical match and the embedding model - which has never
seen either word - maps it to an arbitrary point in vector space.

v1 fixed this with a hand-written dict {'tanjia': ['marrakchi', 'stew',
'lamb'], ...}. That doesn't scale (every new dish needs a code change) and the
generic terms it injected ("stew", "lamb") actually pulled in American stews.

Approach
--------
1. Mine the vocabulary of recipe *titles* with document frequencies - titles
   are where dish names live.
2. For every query token, look for title tokens that are likely spellings of
   the same word, using three signals:
     * a transliteration key that merges the letters Maghrebi
       transliterations swap (g/j/dj, p/b, ch/sh, ou/w/u, q/k/c, doubled
       letters...),
     * its consonant skeleton (Arabic doesn't write short vowels, so
       transliterations disagree most on vowels: beghrir / baghrir),
     * character-level similarity (normalised Indel ratio).
3. Be conservative with words the corpus already knows: they are only mapped
   to an exact transliteration variant that is clearly more frequent
   (tajine -> tagine), so real words are never "corrected" (paste != pasta).

Nothing here is specific to a dish: it works for any word whose variant
spelling exists in the corpus, and improves automatically as data is added.
It is purely orthographic, so it cannot know that "mhammar" is not
"muhammara": that is left to the reranker and the LLM query-understanding
step.
"""
from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from rapidfuzz import fuzz, process

_TOKEN = re.compile(r"[a-z]+")
_WORD = re.compile(r"[^\W\d_]+")
MIN_LEN = 4
COMMON_ZIPF = 2.5  # >= ~3 occurrences per million words in French or English

try:  # optional dependency: without it the guard is simply disabled
    from wordfreq import zipf_frequency
except ImportError:  # pragma: no cover
    zipf_frequency = None


def strip_accents(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text if not unicodedata.combining(c))


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(strip_accents(text.lower()))


def is_common_word(word: str) -> bool:
    """True for everyday French/English words (poisson, facile, recette...).
    Those are not misspelled dish names and must never be fuzzy-'corrected'
    towards an English title word (poisson -> poison, facile -> facial)."""
    if zipf_frequency is None:
        return False
    return max(zipf_frequency(word, "fr"), zipf_frequency(word, "en")) >= COMMON_ZIPF


_TRANSLIT_RULES = [
    (r"dj", "j"), (r"g(?=[eiy])", "j"), (r"ch|sh", "x"), (r"kh|gh", "g"),
    (r"ou|w", "u"), (r"ck|q|c(?=[aou])|k", "k"), (r"ph", "f"), (r"p", "b"),
    (r"y", "i"), (r"(.)\1+", r"\1"),
]


def translit_key(token: str) -> str:
    """Normalise letters that Maghrebi transliterations use interchangeably.
    Two spellings with the same key are the *same word* written differently:
    tanjia/tangia -> 'tanjia', tajine/tagine -> 'tajine',
    bastilla/pastilla -> 'bastila', chebakia/chebakya -> 'xebakia'."""
    t = token
    for pattern, repl in _TRANSLIT_RULES:
        t = re.sub(pattern, repl, t)
    return t


def skeleton(token: str) -> str:
    """Consonant skeleton of the key: beghrir/baghrir -> 'bgrr'."""
    k = translit_key(token)
    return k[:1] + re.sub(r"[aeiou]", "", k[1:])


@dataclass
class VariantLexicon:
    df: dict[str, int]
    stopwords: frozenset[str] = frozenset()
    min_ratio: float = 85.0           # fuzzy-only match threshold
    min_ratio_translit: float = 70.0  # threshold when consonant skeletons agree
    min_freq_gain: float = 3.0        # common words: target must be 3x more frequent
    rare_df: int = 3                  # below this, a query word counts as rare/unknown

    @classmethod
    def from_titles(cls, titles, stopwords=()) -> "VariantLexicon":
        df = Counter(t for title in titles for t in set(tokenize(title)) if len(t) >= MIN_LEN)
        return cls(df=dict(df), stopwords=frozenset(stopwords))

    def __post_init__(self):
        self._vocab = list(self.df)
        self._keys = {t: translit_key(t) for t in self._vocab}
        self._skel = {t: skeleton(t) for t in self._vocab}

    def variants(self, token: str, limit: int = 2, common: bool = False) -> list[str]:
        """Corpus spellings of `token`, best first.

        * Exact transliteration match (same key): accepted even for common
          words, but only towards a clearly more frequent spelling
          (tajine -> tagine, never tagine -> tajine).
        * Looser match (same consonant skeleton, or high fuzzy ratio): only
          for rare / unknown query words - a frequent word like "paste" is a
          real word and must not be "corrected" to "pasta" - and never for
          everyday words of the user's language (`common=True`).
        """
        token = token.lower()
        if len(token) < MIN_LEN or token in self.stopwords:
            return []
        own_df = self.df.get(token, 0)
        rare = own_df < self.rare_df and not common
        key, skel = translit_key(token), skeleton(token)
        scored = []
        for cand, ratio, _ in process.extract(
            token, self._vocab, scorer=fuzz.ratio,
            score_cutoff=self.min_ratio_translit, limit=50,
        ):
            if cand == token or abs(len(cand) - len(token)) > 3:
                continue
            if self._keys[cand] == key:
                # Two everyday words are two different words (poisson != poison),
                # even when their transliteration keys collide.
                if common and is_common_word(cand):
                    continue
                level = 2
            # Short words / skeletons collide too easily (trid ~ tried, meloui ~ meli).
            elif rare and len(token) >= 5 and (
                ratio >= self.min_ratio
                or (self._skel[cand] == skel and len(skel) >= 3 and ratio >= self.min_ratio_translit)
            ):
                level = 1
            else:
                continue
            if not rare and self.df[cand] < self.min_freq_gain * own_df:
                continue
            scored.append((level, ratio, self.df[cand], cand))
        if any(s[0] == 2 for s in scored):
            scored = [s for s in scored if s[0] == 2]
        scored.sort(reverse=True)
        return [c for *_, c in scored[:limit]]

    def expand(self, query: str) -> dict[str, list[str]]:
        """Map each query token to the corpus spellings it likely refers to."""
        out = {}
        for surface in dict.fromkeys(_WORD.findall(query.lower())):
            tok = strip_accents(surface)
            if not _TOKEN.fullmatch(tok) or tok in out:
                continue
            v = self.variants(tok, common=is_common_word(surface))
            if v:
                out[tok] = v
        return out

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"df": self.df, "stopwords": sorted(self.stopwords)}),
                        encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "VariantLexicon":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(df=data["df"], stopwords=frozenset(data["stopwords"]))
