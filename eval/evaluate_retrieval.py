"""Retrieval benchmark / ablation study.

    python -m eval.evaluate_retrieval                       # default systems
    python -m eval.evaluate_retrieval --systems v1 bm25 hybrid_rerank

Each "system" is one configuration of the pipeline. Adding the components one
at a time shows what each of them contributes. Results are written to
eval/results/ (per-query JSON for error analysis + a Markdown summary).
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from datetime import datetime

import numpy as np

from eval.judge import EVAL_DIR, aggregate, grade, load_queries, query_metrics
from recipe_rag.query_understanding import QueryAnalyzer, rule_based_analysis
from recipe_rag.retriever import SearchOptions

K = 10
BM25 = SearchOptions(dense=False, lexicon=False, rerank=False, filters=False, top_k=K)
DENSE = SearchOptions(sparse=False, lexicon=False, rerank=False, filters=False,
                      chunk_types=("overview",), top_k=K)
HYBRID = SearchOptions(lexicon=False, rerank=False, filters=False, chunk_types=("overview",), top_k=K)

# name -> (kind, options, query-analysis mode, description)
SYSTEMS = {
    "v1": ("legacy", True, None, "v1: MiniLM, whole recipe, hard-coded expansion"),
    "v1_noexp": ("legacy", False, None, "v1 without the hard-coded expansion"),
    "bm25": ("v2", BM25, None, "BM25 only"),
    "bm25_lex": ("v2", BM25.but(lexicon=True), None, "BM25 + variant lexicon"),
    "dense": ("v2", DENSE, None, "e5 dense, overview chunks"),
    "hybrid": ("v2", HYBRID, None, "BM25 + dense (RRF)"),
    "hybrid_lex": ("v2", HYBRID.but(lexicon=True), None, "hybrid + lexicon"),
    "hybrid_lex_filters": ("v2", HYBRID.but(lexicon=True, filters=True), "rules",
                           "+ metadata filters (rule-based query parsing)"),
    "hybrid_rerank": ("v2", HYBRID.but(lexicon=True, filters=True, rerank=True), "rules",
                      "+ cross-encoder reranking"),
    "full_llm": ("v2", HYBRID.but(lexicon=True, filters=True, rerank=True), "llm",
                 "+ LLM query understanding"),
    "full_llm_method": ("v2", HYBRID.but(lexicon=True, filters=True, rerank=True,
                                         chunk_types=("overview", "method")), "llm",
                        "+ method chunks"),
}
# The table of docs/TECHNICAL_REPORT.md §9.1. (full_llm needs GROQ_API_KEY;
# full_llm_method needs an index built with --chunk-types overview method.)
DEFAULT = ["v1", "v1_noexp", "bm25", "bm25_lex", "dense", "hybrid", "hybrid_lex",
           "hybrid_lex_filters", "hybrid_rerank", "full_llm"]


class Runner:
    def __init__(self, embedding_model: str | None = None):
        self.embedding_model = embedding_model
        self._engine = self._legacy = self._analyzer = None

    def engine(self, opts: SearchOptions):
        from recipe_rag.retriever import RecipeSearchEngine
        if self._engine is None:
            self._engine = RecipeSearchEngine.load(embedding_model=self.embedding_model)
        if opts.dense and self._engine.dense is None:
            raise SystemExit("Dense index not built yet (python -m recipe_rag.build_index)")
        return self._engine

    def legacy(self, use_expansion: bool):
        from recipe_rag.legacy import LegacyV1Retriever
        if self._legacy is None:
            self._legacy = LegacyV1Retriever()
        self._legacy.use_expansion = use_expansion
        return self._legacy

    def analyzer(self):
        from recipe_rag.config import settings
        from recipe_rag.llm import LLMClient
        if self._analyzer is None:
            llm = LLMClient(settings.llm)
            if not llm.available:
                raise SystemExit("full_llm needs GROQ_API_KEY in .env")
            self._analyzer = QueryAnalyzer(llm, settings.llm.query_model)
        return self._analyzer

    def cached_analysis(self, query: str):
        """LLM parses are cached on disk so that comparisons between retrieval
        configurations use exactly the same parses (and cost nothing to rerun)."""
        from recipe_rag.config import settings
        from recipe_rag.query_understanding import PROMPT_VERSION, QueryAnalysis
        path = EVAL_DIR / "results" / "llm_query_cache.json"
        if not hasattr(self, "_qcache"):
            self._qcache = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        key = f"{settings.llm.query_model}::{PROMPT_VERSION}::{query}"
        spellings = self._engine.lexicon.expand(query) if self._engine else {}
        if key not in self._qcache:
            a = self.analyzer().analyze(query, corpus_spellings=spellings)
            if a.source != "llm":  # failed call: retry once, never cache a fallback
                a = self.analyzer().analyze(query, corpus_spellings=spellings)
            if a.source != "llm":
                return a
            self._qcache[key] = a.model_dump()
            path.parent.mkdir(exist_ok=True)
            path.write_text(json.dumps(self._qcache, ensure_ascii=False, indent=1), encoding="utf-8")
        return QueryAnalysis.model_validate(self._qcache[key])

    def run(self, name: str, query: str) -> tuple[list[dict], dict]:
        kind, opts, analysis_mode, _ = SYSTEMS[name]
        if kind == "legacy":
            return self.legacy(opts).search(query, K), {}
        engine = self.engine(opts)
        analysis = None  # (engine is loaded first: its lexicon feeds the LLM parser)
        if analysis_mode == "rules":
            analysis = rule_based_analysis(query)
        elif analysis_mode == "llm":
            analysis = self.cached_analysis(query)
        res = engine.search(query, analysis, opts)
        recipes = [recipe_dict(engine.recipes, h.recipe_idx) for h in res.hits]
        extra = {"expanded": res.expanded_terms,
                 "analysis": analysis.model_dump() if analysis else None}
        return recipes, extra


def recipe_dict(recipes, idx: int) -> dict:
    r = recipes.iloc[idx]
    return {"recipe_id": r["recipe_id"], "name": r["name"], "ingredients": list(r["ingredients"]),
            "is_moroccan": r["is_moroccan"], "is_vegetarian": r["is_vegetarian"],
            "is_vegan": r["is_vegan"], "is_gluten_free": r["is_gluten_free"],
            "minutes": r["minutes"], "calories": r["calories"]}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="+", default=DEFAULT, choices=list(SYSTEMS))
    ap.add_argument("--tag", default="")
    ap.add_argument("--embedding-model", default=None,
                    help="dense index to use (default: EMBEDDING_MODEL from config)")
    args = ap.parse_args(argv)

    queries = [q for q in load_queries() if q.get("relevant")]
    qrels = json.loads((EVAL_DIR / "qrels.json").read_text(encoding="utf-8"))
    runner = Runner(args.embedding_model)
    rows = []
    for name in args.systems:
        runner.run(name, "warm up")  # load models outside the timed loop
        for q in queries:
            t0 = time.perf_counter()
            results, extra = runner.run(name, q["query"])
            latency = (time.perf_counter() - t0) * 1000
            grades = [grade(q, r) for r in results]
            rel = qrels[q["id"]]
            n2 = sum(g == 2 for g in rel.values())
            rows.append({
                "system": name, "qid": q["id"], "category": q["category"], "query": q["query"],
                **query_metrics(grades, n2, len(rel) - n2, K),
                "latency_ms": latency, "grades": grades,
                "top3": [r["name"] for r in results[:3]], **extra,
            })
        summary = aggregate([r for r in rows if r["system"] == name])
        print(f"{name:20} nDCG@10={summary['ndcg@10']:.3f}  MRR={summary['mrr@10']:.3f}  "
              f"Hit@5={summary['hit@5']:.3f}", flush=True)

    out_dir = EVAL_DIR / "results"
    out_dir.mkdir(exist_ok=True)
    stamp = args.tag or datetime.now().strftime("%Y%m%d_%H%M")
    (out_dir / f"retrieval_{stamp}.json").write_text(
        json.dumps(rows, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    md = report(rows, args.systems)
    (out_dir / f"retrieval_{stamp}.md").write_text(md, encoding="utf-8")
    print("\n" + md)


def report(rows: list[dict], systems: list[str]) -> str:
    cats = sorted({r["category"] for r in rows}, key=lambda c: [
        "moroccan_fr", "darija", "intl_en", "intl_fr", "ingredients", "constraints"].index(c))
    by = defaultdict(list)
    for r in rows:
        by[r["system"]].append(r)
    n = len(by[systems[0]])
    lines = [f"### Overall ({n} queries)", "",
             "| System | nDCG@10 | MRR@10 | Hit@1 | Hit@5 | Recall@10 | p50 latency |",
             "|---|---|---|---|---|---|---|"]
    for s in systems:
        a = aggregate(by[s])
        lat = np.median([r["latency_ms"] for r in by[s]])
        lines.append(f"| {SYSTEMS[s][3]} | {a['ndcg@10']:.3f} | {a['mrr@10']:.3f} | {a['hit@1']:.3f} "
                     f"| {a['hit@5']:.3f} | {a['recall@10']:.3f} | {lat:.0f} ms |")
    lines += ["", "### nDCG@10 by query category", "",
              "| System | " + " | ".join(cats) + " |", "|---|" + "---|" * len(cats)]
    for s in systems:
        cells = [f"{aggregate([r for r in by[s] if r['category'] == c])['ndcg@10']:.3f}" for c in cats]
        lines.append(f"| {SYSTEMS[s][3]} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
