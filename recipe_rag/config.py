"""Central configuration.

Every tunable of the pipeline lives here so that experiments (eval/) and the
web app share exactly the same settings. Values can be overridden through
environment variables (see `.env.example`).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


def model_slug(name: str) -> str:
    return name.replace("/", "__")


@dataclass(frozen=True)
class Paths:
    raw_dir: Path = PROJECT_ROOT / "data" / "raw"
    processed_dir: Path = PROJECT_ROOT / "data" / "processed"
    index_dir: Path = PROJECT_ROOT / "data" / "index"
    legacy_index_dir: Path = PROJECT_ROOT / "data" / "embeddings"

    @property
    def foodcom_csv(self) -> Path:
        return self.raw_dir / "RAW_recipes.csv"

    @property
    def moroccan_csv(self) -> Path:
        return self.raw_dir / "moroccan_recipes_dataset.csv"

    @property
    def recipes_parquet(self) -> Path:
        return self.processed_dir / "recipes.parquet"

    def dense_dir(self, embedding_model: str) -> Path:
        """Each embedding model gets its own FAISS index + chunk table, so
        several models can be built and benchmarked side by side."""
        return self.index_dir / model_slug(embedding_model)

    def embedding_cache(self, embedding_model: str, chunk_type: str) -> Path:
        return self.index_dir / "cache" / model_slug(embedding_model) / chunk_type

    @property
    def bm25_dir(self) -> Path:
        return self.index_dir / "bm25"

    @property
    def lexicon_json(self) -> Path:
        return self.index_dir / "lexicon.json"


@dataclass(frozen=True)
class ChunkingConfig:
    # Max words of a "method" chunk. ~180 words ≈ 240 e5 tokens: comfortably
    # inside the 512-token window, small enough to stay topically focused.
    max_method_words: int = 180
    # Description is truncated in the overview chunk: long blog-style intros
    # dilute the embedding without adding retrievable facts.
    max_description_words: int = 60


@dataclass(frozen=True)
class RetrievalConfig:
    embedding_model: str = _env("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
    reranker_model: str = _env("RERANKER_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
    # Candidates pulled from each retriever before fusion.
    dense_candidates: int = 100
    sparse_candidates: int = 100
    # Reciprocal Rank Fusion constant (Cormack et al., 2009 use k=60).
    rrf_k: int = 60
    # How many fused candidates are re-scored by the cross-encoder, and how many
    # tokens of each it reads (latency is ~linear in both on CPU).
    rerank_candidates: int = int(_env("RERANK_CANDIDATES", "20"))
    rerank_max_length: int = int(_env("RERANK_MAX_LENGTH", "256"))
    # How scores for the user's query and the LLM rewrite are combined.
    rerank_agg: str = _env("RERANK_AGG", "mean")
    # Recipes passed to the LLM.
    top_k: int = 4
    # Below this cross-encoder score the answer is flagged as "low confidence"
    # (calibrated on the eval set, see eval/README.md).
    confidence_threshold: float = float(_env("CONFIDENCE_THRESHOLD", "2.0"))
    use_reranker: bool = _env("USE_RERANKER", "1") == "1"
    use_query_understanding: bool = _env("USE_QUERY_UNDERSTANDING", "1") == "1"


@dataclass(frozen=True)
class LLMConfig:
    api_key: str | None = os.getenv("GROQ_API_KEY")
    # Groq retired the Llama-4 model v1 used; gpt-oss-120b is the strongest open-weight
    # model currently served there.
    model: str = _env("LLM_MODEL", "openai/gpt-oss-120b")
    # A small, fast model is enough to turn a question into structured JSON.
    query_model: str = _env("LLM_QUERY_MODEL", "openai/gpt-oss-20b")
    # Judge from a different model family than the generator (limits self-preference bias).
    judge_model: str = _env("LLM_JUDGE_MODEL", "qwen/qwen3.8-27b")
    # gpt-oss models "think" before answering; low effort keeps latency acceptable.
    reasoning_effort: str = _env("LLM_REASONING_EFFORT", "low")
    # Tried in order when a model is rate-limited or retired.
    fallbacks: dict = field(default_factory=lambda: {
        "openai/gpt-oss-20b": ("openai/gpt-oss-120b",),
        "openai/gpt-oss-120b": ("openai/gpt-oss-20b",),
        "qwen/qwen3.8-27b": ("openai/gpt-oss-120b",),
    })
    temperature: float = 0.3
    max_tokens: int = 900
    timeout_s: float = 30.0


@dataclass(frozen=True)
class Settings:
    paths: Paths = field(default_factory=Paths)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)


settings = Settings()
