"""Calibrate the "low retrieval confidence" threshold on the cross-encoder score.

    python -m eval.calibrate_confidence

Positives: answerable queries whose top-1 result is fully relevant.
Negatives: in-domain unanswerable queries (dish absent from the corpus) and
answerable queries whose top-1 result is not relevant. The threshold maximising
balanced accuracy is reported; it is then set as CONFIDENCE_THRESHOLD.
Off-topic queries are excluded: they never reach retrieval (intent detection).
"""
from __future__ import annotations

import json

import numpy as np

from eval.evaluate_retrieval import recipe_dict
from eval.judge import EVAL_DIR, grade, load_queries
from recipe_rag.config import settings
from recipe_rag.llm import LLMClient
from recipe_rag.query_understanding import QueryAnalyzer
from recipe_rag.retriever import RecipeSearchEngine, SearchOptions

IN_DOMAIN_UNANSWERABLE = {"una01", "una02", "una03", "una07"}


def main() -> None:
    engine = RecipeSearchEngine.load()
    analyzer = QueryAnalyzer(LLMClient(settings.llm), settings.llm.query_model)
    rows = []
    for q in load_queries():
        if q["category"] == "unanswerable" and q["id"] not in IN_DOMAIN_UNANSWERABLE:
            continue
        a = analyzer.analyze(q["query"], corpus_spellings=engine.lexicon.expand(q["query"]))
        res = engine.search(a.standalone_question, a, SearchOptions(top_k=1))
        top = res.hits[0] if res.hits else None
        relevant = bool(top) and q.get("relevant") is not None and \
            grade(q, recipe_dict(engine.recipes, top.recipe_idx)) == 2
        rows.append({"qid": q["id"], "score": res.confidence, "positive": relevant,
                     "top": engine.recipes.iloc[top.recipe_idx]["name"] if top else None})
        print(f"{q['id']:6} score={res.confidence:6.2f} positive={relevant}  {rows[-1]['top']}")

    scores = np.array([r["score"] for r in rows])
    pos = np.array([r["positive"] for r in rows])
    best = max(
        (((scores >= t) & pos).sum() / pos.sum() / 2 + ((scores < t) & ~pos).sum() / (~pos).sum() / 2, t)
        for t in np.unique(scores)
    )
    print(f"\npositives={pos.sum()} negatives={(~pos).sum()}  best threshold={best[1]:.2f} "
          f"balanced accuracy={best[0]:.3f}")
    for qid in sorted(IN_DOMAIN_UNANSWERABLE):
        r = next(r for r in rows if r["qid"] == qid)
        print(f"  {qid}: score {r['score']:.2f} -> {'LOW' if r['score'] < best[1] else 'high'}")
    (EVAL_DIR / "results" / "confidence_calibration.json").write_text(
        json.dumps({"threshold": float(best[1]), "balanced_accuracy": float(best[0]), "rows": rows},
                   indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
