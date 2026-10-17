"""Structure-aware chunking for recipes.

Why not fixed-size chunks? A recipe is already a structured document (title,
description, tags, ingredients, steps). Cutting it every N tokens would split
an ingredient list in half or glue the end of one step to the start of the
next. Instead each recipe is mapped to:

* one **overview** chunk  – title + short description + informative tags +
  ingredients. This is what users search by ("chicken tagine with olives",
  "j'ai des pois chiches et des tomates"), and it always fits in the
  embedding model's window.
* N **method** chunks     – consecutive steps grouped up to a word budget,
  each prefixed with the recipe title ("contextual chunk header") so the
  chunk is still meaningful on its own ("slow-cook in a clay jar overnight").

Retrieval happens on chunks, but scores are aggregated back to the parent
recipe (max-pooling) and the LLM always receives the full recipe
("small-to-big" / parent-document retrieval).

v1 embedded the whole recipe as one string with all-MiniLM-L6-v2, whose
window is 256 tokens: for ~40% of recipes the end of the text (often the
whole method) was silently truncated.
"""
from __future__ import annotations

import re
from collections import Counter

import pandas as pd

from recipe_rag.config import settings

_COUNT_TAG = re.compile(r"^\d+-(minutes|hours|steps|ingredients|or-less|days).*|.*-or-less$")


def uninformative_tags(tags: pd.Series, max_df: float = 0.15) -> set[str]:
    """Tags present in too many recipes carry no signal (Food.com has
    'preparation', 'time-to-make', 'course'... on almost every recipe).
    Time/size tags are dropped too: they are handled as structured filters."""
    df = Counter(t for ts in tags for t in set(ts))
    n = len(tags)
    return {t for t, c in df.items() if c / n > max_df or _COUNT_TAG.match(t)}


def _truncate_words(text: str, n: int) -> str:
    words = text.split()
    return " ".join(words[:n]) + (" …" if len(words) > n else "")


def overview_text(row, drop_tags: set[str], max_description_words: int) -> str:
    tags = [t.replace("-", " ") for t in row["tags"] if t not in drop_tags]
    parts = [row["name"]]
    if row["description"]:
        parts.append(_truncate_words(row["description"], max_description_words))
    if len(tags):
        parts.append("Tags: " + ", ".join(tags))
    parts.append("Ingredients: " + ", ".join(row["ingredients"]))
    return "\n".join(parts)


def method_chunks(name: str, steps: list[str], max_words: int) -> list[str]:
    """Greedily pack consecutive steps into chunks of at most `max_words`.
    A single over-long step becomes its own chunk (never cut mid-step)."""
    chunks, current, count, first = [], [], 0, 1
    for i, step in enumerate(steps, start=1):
        w = len(step.split())
        if current and count + w > max_words:
            chunks.append((first, i - 1, current))
            current, count, first = [], 0, i
        current.append(step)
        count += w
    if current:
        chunks.append((first, len(steps), current))
    return [
        f"How to make {name} (steps {a}-{b} of {len(steps)}):\n" + "\n".join(s)
        for a, b, s in chunks
    ]


def full_text(row) -> str:
    """Whole recipe as one string (used by BM25 and by the v1-style ablation).
    The title is repeated: a cheap BM25F-style field boost."""
    return "\n".join([
        row["name"], row["name"], row["description"] or "",
        " ".join(row["tags"]).replace("-", " "),
        "Ingredients: " + ", ".join(row["ingredients"]),
        "Steps: " + " ".join(row["steps"]),
    ])


def build_chunks(recipes: pd.DataFrame, with_method: bool = True) -> pd.DataFrame:
    cfg = settings.chunking
    drop = uninformative_tags(recipes["tags"])
    records = []
    for idx, row in enumerate(recipes.itertuples(index=False)):
        r = row._asdict()
        records.append((idx, "overview", overview_text(r, drop, cfg.max_description_words)))
        if with_method:
            for text in method_chunks(r["name"], list(r["steps"]), cfg.max_method_words):
                records.append((idx, "method", text))
    chunks = pd.DataFrame(records, columns=["recipe_idx", "chunk_type", "text"])
    chunks.index.name = "chunk_id"
    return chunks.reset_index()
