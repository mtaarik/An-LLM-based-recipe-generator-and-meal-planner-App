"""Cross-encoder reranking.

Bi-encoders (dense retrieval) embed query and document *separately*, which
makes search fast but loses fine-grained interactions. A cross-encoder reads
(query, document) *together* and is much more accurate, but too slow to run on
the whole corpus. Standard two-stage design: cheap recall with hybrid search,
then precise re-ordering of the top ~30 candidates.

Default model: mMiniLM-L12 fine-tuned on mMARCO (multilingual MS MARCO), small
enough for CPU and able to score a French query against an English recipe.
"""
from __future__ import annotations

import numpy as np
from sentence_transformers import CrossEncoder


class Reranker:
    def __init__(self, model_name: str, device: str = "cpu", max_length: int = 384):
        self.model = CrossEncoder(model_name, device=device, max_length=max_length)

    def score(self, query: str, docs: list[str]) -> np.ndarray:
        return self.score_multi([query], docs)

    def score_multi(self, queries: list[str], docs: list[str], agg: str = "mean") -> np.ndarray:
        """Score docs against several formulations of the same need (one batched
        call) and aggregate: "mean" = a doc must fit all formulations."""
        if not docs:
            return np.array([])
        pairs = [(q, d) for q in queries for d in docs]
        scores = np.asarray(self.model.predict(pairs, batch_size=32, show_progress_bar=False))
        scores = scores.reshape(len(queries), len(docs))
        return scores.max(axis=0) if agg == "max" else scores.mean(axis=0)
