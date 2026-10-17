"""Sparse (lexical) retrieval with BM25.

Dense embeddings are good at meaning ("a slow-cooked lamb dish" ~ "tangia")
but bad at rare proper nouns they never saw in training (rfissa, msemen,
seffa). BM25 is the opposite: it matches exact tokens and gives rare tokens
a high IDF weight. Fusing both (see retriever.py) gets the best of each.

One BM25 document per recipe (not per chunk): BM25 has no context window,
and its length normalisation already handles long documents.
"""
from __future__ import annotations

from pathlib import Path

import bm25s
import numpy as np
import Stemmer

# The corpus is English but most user queries are French, so French function
# words are removed too. QUERY_NOISE are "how do I make a recipe" words that
# say nothing about *which* recipe is wanted (query side only).
QUERY_NOISE = """recipe recipes make making cook cooking prepare how want would like
give show please tell traditional homemade comment faire préparer preparer recette
recettes cuisiner plat plats veux voudrais aimerais peux pouvez donne donner quoi
idée idee""".split()


class BM25Index:
    def __init__(self, retriever: bm25s.BM25, language: str = "english"):
        self.retriever = retriever
        self.stemmer = Stemmer.Stemmer(language)
        self.stopwords = list(bm25s.stopwords.STOPWORDS_EN) + list(bm25s.stopwords.STOPWORDS_FRENCH)

    def _tokenize(self, texts: list[str], query: bool = False):
        stop = self.stopwords + (QUERY_NOISE if query else [])
        return bm25s.tokenize(texts, stopwords=stop, stemmer=self.stemmer,
                              return_ids=False, show_progress=False)

    @classmethod
    def build(cls, corpus: list[str]) -> "BM25Index":
        index = cls(bm25s.BM25(k1=1.2, b=0.75))
        index.retriever.index(bm25s.tokenize(
            corpus, stopwords=index.stopwords, stemmer=index.stemmer, show_progress=False),
            show_progress=False)
        return index

    def search(self, query: str, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Returns (recipe indices, scores), best first."""
        tokens = [t for t in self._tokenize([query], query=True)[0]
                  if t in self.retriever.vocab_dict]
        if not tokens:
            return np.array([], dtype=int), np.array([], dtype="float32")
        docs, scores = self.retriever.retrieve([tokens], k=k, show_progress=False,
                                               n_threads=1)
        keep = scores[0] > 0
        return docs[0][keep], scores[0][keep]

    def save(self, path: Path) -> None:
        self.retriever.save(str(path))

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        return cls(bm25s.BM25.load(str(path), mmap=False))
