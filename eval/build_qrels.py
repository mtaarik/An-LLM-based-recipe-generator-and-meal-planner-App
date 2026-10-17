"""Resolve the rules of queries.yaml into explicit qrels over the v2 corpus.

    python -m eval.build_qrels

Writes eval/qrels.json ({query_id: {recipe_id: grade}}) and prints how many
relevant recipes each query has, so that empty or absurdly large relevance
sets are caught before running any experiment.
"""
from __future__ import annotations

import json

from eval.judge import EVAL_DIR, grade, load_queries
from recipe_rag.ingest import load_recipes


def recipe_dicts(recipes):
    cols = ["recipe_id", "name", "ingredients", "is_moroccan", "is_vegetarian",
            "is_vegan", "is_gluten_free", "minutes", "calories"]
    for row in recipes[cols].itertuples(index=False):
        d = row._asdict()
        d["ingredients"] = list(d["ingredients"])
        yield d


def main() -> None:
    queries = [q for q in load_queries() if q.get("relevant")]
    recipes = list(recipe_dicts(load_recipes()))
    qrels = {}
    for q in queries:
        rel = {r["recipe_id"]: g for r in recipes if (g := grade(q, r))}
        qrels[q["id"]] = rel
        n2 = sum(g == 2 for g in rel.values())
        n1 = len(rel) - n2
        flag = "  <-- EMPTY" if n2 == 0 else ""
        print(f"{q['id']:6} {q['category']:12} relevant={n2:6d} partial={n1:5d}  {q['query']}{flag}")
    (EVAL_DIR / "qrels.json").write_text(json.dumps(qrels, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
