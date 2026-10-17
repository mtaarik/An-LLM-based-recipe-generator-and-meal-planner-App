"""Rule-based relevance judge + IR metrics.

The same `grade()` function scores the results of *every* system, including
v1 (which used a different metadata file), so all systems are judged by the
exact same criteria.
"""
from __future__ import annotations

import math
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml

EVAL_DIR = Path(__file__).resolve().parent


def load_queries(path: Path = EVAL_DIR / "queries.yaml") -> list[dict]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))["queries"]


@lru_cache(maxsize=None)
def _rx(pattern: str) -> re.Pattern:
    return re.compile(pattern, re.IGNORECASE)


def matches(rule: dict | None, recipe: dict) -> bool:
    """`recipe` needs: name, ingredients (list[str]), is_moroccan, is_vegetarian,
    is_vegan, is_gluten_free, minutes, calories."""
    if not rule:
        return False
    if "name" in rule and not _rx(rule["name"]).search(recipe["name"]):
        return False
    if "ing" in rule:
        ings = recipe["ingredients"]
        if not all(any(_rx(p).search(i) for i in ings) for p in rule["ing"]):
            return False
    for flag, col in [("moroccan", "is_moroccan"), ("vegetarian", "is_vegetarian"),
                      ("vegan", "is_vegan"), ("gluten_free", "is_gluten_free")]:
        if flag in rule and bool(recipe[col]) != rule[flag]:
            return False
    for limit, col in [("max_minutes", "minutes"), ("max_calories", "calories")]:
        if limit in rule:
            v = recipe.get(col)
            if v is None or (isinstance(v, float) and math.isnan(v)) or v > rule[limit]:
                return False
    return True


def grade(query: dict, recipe: dict) -> int:
    if matches(query.get("relevant"), recipe):
        return 2
    if matches(query.get("partial"), recipe):
        return 1
    return 0


# ----------------------------------------------------------------- metrics
def dcg(grades: list[int]) -> float:
    return sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(grades))


def query_metrics(grades: list[int], n_relevant: int, n_partial: int, k: int = 10) -> dict:
    """grades: relevance grade of each returned result, in rank order."""
    g = grades[:k]
    ideal = [2] * min(n_relevant, k)
    ideal += [1] * min(n_partial, k - len(ideal))
    idcg = dcg(ideal)
    first = next((i for i, x in enumerate(g) if x == 2), None)
    return {
        "ndcg@10": dcg(g) / idcg if idcg else 0.0,
        "mrr@10": 1.0 / (first + 1) if first is not None else 0.0,
        "hit@1": float(bool(g) and g[0] == 2),
        "hit@5": float(2 in g[:5]),
        "recall@10": (sum(x == 2 for x in g) / min(n_relevant, k)) if n_relevant else 0.0,
    }


def aggregate(rows: list[dict]) -> dict:
    keys = ["ndcg@10", "mrr@10", "hit@1", "hit@5", "recall@10"]
    return {k: float(np.mean([r[k] for r in rows])) for k in keys} if rows else {}
