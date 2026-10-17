"""Hybrid retrieval pipeline.

    query ─┬─► lexicon variants (tanjia → tangia)
           ├─► dense search over chunks ──► max-pool to recipes ─┐
           └─► BM25 over recipes ────────────────────────────────┤
                                                                 ▼
                      metadata filters (diet / time / calories / excluded ingredients)
                                                                 ▼
                                      Reciprocal Rank Fusion (RRF)
                                                                 ▼
                                   cross-encoder rerank of top-N
                                                                 ▼
                                    top-k recipes + confidence score

Each stage can be switched off (`SearchOptions`) so that eval/ can measure
its contribution in isolation (ablation study).
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from recipe_rag.config import RetrievalConfig, settings
from recipe_rag.dense import DenseIndex, Embedder
from recipe_rag.lexicon import VariantLexicon
from recipe_rag.query_understanding import QueryAnalysis
from recipe_rag.reranker import Reranker
from recipe_rag.sparse import BM25Index

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SearchOptions:
    dense: bool = True
    sparse: bool = True
    lexicon: bool = True
    rerank: bool = True
    filters: bool = True
    # Overview chunks only: adding method chunks lowered nDCG@10 (0.792 vs 0.828).
    chunk_types: tuple[str, ...] = ("overview",)
    top_k: int = 4

    def but(self, **changes) -> "SearchOptions":
        return replace(self, **changes)


@dataclass
class Hit:
    recipe_idx: int
    score: float
    dense_rank: int | None = None
    sparse_rank: int | None = None
    rerank_score: float | None = None
    matched_chunk: str | None = None


@dataclass
class SearchResult:
    hits: list[Hit]
    expanded_terms: dict[str, list[str]] = field(default_factory=dict)
    confidence: float | None = None
    timings_ms: dict[str, float] = field(default_factory=dict)


def rrf_fuse(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion: score(d) = Σ 1 / (k + rank_d).

    Uses only ranks, so it needs no score calibration between BM25 scores
    (unbounded) and cosine similarities ([-1, 1]) – the main reason it is
    preferred over a weighted sum of raw scores.
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])


class RecipeSearchEngine:
    def __init__(self, recipes: pd.DataFrame, chunks: pd.DataFrame, embedder: Embedder | None,
                 dense: DenseIndex | None, bm25: BM25Index, lexicon: VariantLexicon,
                 reranker: Reranker | None = None, cfg: RetrievalConfig | None = None):
        self.recipes = recipes
        self.chunk_recipe = chunks["recipe_idx"].to_numpy()
        self.chunk_type = chunks["chunk_type"].to_numpy()
        self.chunk_text = chunks["text"].to_numpy()
        self.overview_by_recipe = (chunks[chunks["chunk_type"] == "overview"]
                                   .set_index("recipe_idx")["text"].to_dict())
        self.embedder, self.dense, self.bm25 = embedder, dense, bm25
        self.lexicon, self.reranker = lexicon, reranker
        self.cfg = cfg or settings.retrieval
        self._ingredients_low = None

    # ------------------------------------------------------------------ load
    @classmethod
    def load(cls, with_dense: bool = True, with_reranker: bool = True,
             embedding_model: str | None = None) -> "RecipeSearchEngine":
        from recipe_rag.chunking import build_chunks
        from recipe_rag.ingest import load_recipes
        p, cfg = settings.paths, settings.retrieval
        model = embedding_model or cfg.embedding_model
        dense_dir = p.dense_dir(model)
        t0 = time.perf_counter()
        recipes = load_recipes()
        with_dense = with_dense and (dense_dir / "dense.faiss").exists()
        if not with_dense:
            log.warning("No dense index for %s: BM25-only retrieval", model)
        engine = cls(
            recipes=recipes,
            chunks=(pd.read_parquet(dense_dir / "chunks.parquet") if with_dense
                    else build_chunks(recipes, with_method=False)),
            embedder=Embedder(model) if with_dense else None,
            dense=DenseIndex.load(dense_dir / "dense.faiss") if with_dense else None,
            bm25=BM25Index.load(p.bm25_dir),
            lexicon=VariantLexicon.load(p.lexicon_json),
            reranker=(Reranker(cfg.reranker_model, max_length=cfg.rerank_max_length)
                      if with_reranker and cfg.use_reranker else None),
            cfg=cfg,
        )
        log.info("Search engine loaded in %.1fs", time.perf_counter() - t0)
        return engine

    # --------------------------------------------------------------- search
    def search(self, query: str, analysis: QueryAnalysis | None = None,
               opts: SearchOptions | None = None) -> SearchResult:
        opts = opts or SearchOptions(top_k=self.cfg.top_k)
        timings: dict[str, float] = {}
        t = time.perf_counter()

        def lap(name):
            nonlocal t
            now = time.perf_counter()
            timings[name] = round((now - t) * 1000, 1)
            t = now

        # 1. Query views ------------------------------------------------------
        expanded = self.lexicon.expand(query) if opts.lexicon else {}
        variants = [v for vs in expanded.values() for v in vs]
        rewritten = analysis.search_query if analysis and analysis.source == "llm" else ""
        dish_terms = " ".join(analysis.dish_names) if analysis else ""
        base = " ".join([query, *variants])
        # Spelling variants go to BM25 and the reranker (exact tokens matter there) but
        # not to the embedding: a string of unfamiliar words ("ghoriba ghriba gorayba")
        # embeds near unrelated recipes (garam masala, sambhar).
        dense_queries = [query] + ([rewritten] if rewritten else [])
        lexical_query = " ".join(filter(None, [base, rewritten, dish_terms]))
        lap("query_expansion")

        mask = self._filter_mask(analysis) if (opts.filters and analysis) else None
        widen = 5 if mask is not None else 1

        # 2. Retrieval ----------------------------------------------------------
        rankings: list[list[int]] = []
        dense_rank, best_chunk = {}, {}
        if opts.dense and self.dense is not None:
            qv = self.embedder.encode_queries(dense_queries)
            n = self.cfg.dense_candidates * widen * (3 if len(opts.chunk_types) < 2 else 1)
            scores, ids = self.dense.search(qv, n)
            lap("dense")
            for row_scores, row_ids in zip(scores, ids):
                ranking = self._pool_chunks(row_ids, row_scores, opts.chunk_types, mask, best_chunk)
                rankings.append(ranking)
                for r, doc in enumerate(ranking, 1):
                    dense_rank[doc] = min(r, dense_rank.get(doc, r))
        sparse_rank = {}
        if opts.sparse:
            docs, _ = self.bm25.search(lexical_query, self.cfg.sparse_candidates * widen)
            ranking = [int(d) for d in docs if mask is None or mask[d]]
            rankings.append(ranking)
            sparse_rank = {d: r for r, d in enumerate(ranking, 1)}
            lap("sparse")

        # 3. Fusion -------------------------------------------------------------
        fused = rrf_fuse(rankings, self.cfg.rrf_k)
        hits = [Hit(recipe_idx=d, score=s, dense_rank=dense_rank.get(d),
                    sparse_rank=sparse_rank.get(d), matched_chunk=best_chunk.get(d))
                for d, s in fused]
        lap("fusion")

        # 4. Rerank -------------------------------------------------------------
        confidence = None
        if opts.rerank and self.reranker is not None and hits:
            pool = hits[: self.cfg.rerank_candidates]
            docs = [self._rerank_text(h) for h in pool]
            # Each candidate is scored against the user's own words (+ corpus spellings)
            # AND against the LLM's English rewrite (+ spellings); the best score wins.
            # The rewrite helps French queries, but must not be able to override the
            # user: the LLM once mistook "harsha" (a flatbread) for "harira" (a soup).
            queries = [base] + ([f"{rewritten} {' '.join(variants)}".strip()] if rewritten else [])
            ce = self.reranker.score_multi(queries, docs, agg=self.cfg.rerank_agg)
            for h, s in zip(pool, ce):
                h.rerank_score = float(s)
            hits = sorted(pool, key=lambda h: -h.rerank_score)
            confidence = hits[0].rerank_score
            lap("rerank")

        return SearchResult(hits=hits[: opts.top_k], expanded_terms=expanded,
                            confidence=confidence, timings_ms=timings)

    # -------------------------------------------------------------- helpers
    def _pool_chunks(self, ids, scores, chunk_types, mask, best_chunk) -> list[int]:
        """Max-pool chunk scores into a recipe ranking (parent-document retrieval)."""
        seen, ranking = set(), []
        for cid, s in zip(ids, scores):
            if cid < 0 or self.chunk_type[cid] not in chunk_types:
                continue
            doc = int(self.chunk_recipe[cid])
            if doc in seen or (mask is not None and not mask[doc]):
                continue
            seen.add(doc)
            ranking.append(doc)
            best_chunk.setdefault(doc, self.chunk_text[cid])
        return ranking

    def _rerank_text(self, hit: Hit) -> str:
        overview = self.overview_by_recipe[hit.recipe_idx]
        if hit.matched_chunk and hit.matched_chunk != overview:
            return overview + "\n" + hit.matched_chunk
        return overview

    def _filter_mask(self, a: QueryAnalysis) -> np.ndarray | None:
        if not a.has_filters:
            return None
        r = self.recipes
        mask = np.ones(len(r), dtype=bool)
        for diet, col in [("vegetarian", "is_vegetarian"), ("vegan", "is_vegan"),
                          ("gluten-free", "is_gluten_free")]:
            if diet in a.diet:
                mask &= r[col].to_numpy()
        # Missing time / calories (e.g. Moroccan dataset) is not treated as a
        # violation: we don't know, so we don't exclude.
        if a.max_minutes:
            m = r["minutes"].to_numpy()
            mask &= np.isnan(m) | (m <= a.max_minutes)
        if a.max_calories:
            c = r["calories"].to_numpy()
            mask &= np.isnan(c) | (c <= a.max_calories)
        if a.exclude_ingredients:
            if self._ingredients_low is None:
                self._ingredients_low = r["ingredients"].map(lambda xs: " | ".join(xs).lower()).to_numpy()
            for ing in a.exclude_ingredients:
                term = ing.lower().rstrip("s")
                mask &= np.array([term not in s for s in self._ingredients_low])
        return mask
