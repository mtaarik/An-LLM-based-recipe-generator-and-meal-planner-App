"""Dense (embedding) retrieval: sentence-transformer encoder + FAISS index."""
from __future__ import annotations

import logging
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

log = logging.getLogger(__name__)


class Embedder:
    """Wraps a sentence-transformer and applies model-specific prefixes.

    E5 models are trained with asymmetric prefixes ("query: " / "passage: ");
    omitting them costs several points of retrieval quality.
    """

    def __init__(self, model_name: str, device: str = "cpu"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self._e5 = "e5" in model_name.lower()
        self.dim = self.model.get_embedding_dimension()

    @property
    def max_seq_length(self) -> int:
        return self.model.max_seq_length

    def _prefix(self, texts: list[str], kind: str) -> list[str]:
        return [f"{kind}: {t}" for t in texts] if self._e5 else list(texts)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(self._prefix(texts, "query"), normalize_embeddings=True,
                                 convert_to_numpy=True).astype("float32")

    def encode_passages(self, texts: list[str], batch_size: int = 64,
                        show_progress: bool = True) -> np.ndarray:
        return self.model.encode(self._prefix(texts, "passage"), batch_size=batch_size,
                                 normalize_embeddings=True, convert_to_numpy=True,
                                 show_progress_bar=show_progress).astype("float32")


class DenseIndex:
    """Cosine-similarity search over chunk embeddings.

    Vectors are L2-normalised, so inner product == cosine. They are stored with
    8-bit scalar quantization (SQ8): 4x less memory than float32 for a
    negligible recall loss (measured in eval/), while keeping exact
    (exhaustive) search – at a few hundred thousand vectors a flat scan is
    ~tens of ms on CPU, so an approximate graph index (HNSW) isn't needed yet.
    """

    def __init__(self, index: faiss.Index):
        self.index = index

    @classmethod
    def build(cls, embeddings: np.ndarray, quantize: bool = True) -> "DenseIndex":
        d = embeddings.shape[1]
        if quantize:
            index = faiss.IndexScalarQuantizer(d, faiss.ScalarQuantizer.QT_8bit,
                                               faiss.METRIC_INNER_PRODUCT)
            index.train(embeddings)
        else:
            index = faiss.IndexFlatIP(d)
        index.add(embeddings)
        return cls(index)

    def search(self, query_vecs: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        scores, ids = self.index.search(query_vecs, k)
        return scores, ids

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path))

    @classmethod
    def load(cls, path: Path) -> "DenseIndex":
        return cls(faiss.read_index(str(path)))
