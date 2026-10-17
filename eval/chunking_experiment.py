"""Chunking-strategy experiment (dense retrieval only, so chunking is the only variable).

    python -m eval.chunking_experiment

Strategies compared:
  A. whole recipe, all-MiniLM-L6-v2      (v1: one vector per recipe, 256-token window)
  B. whole recipe, multilingual-e5-small (one vector per recipe, 512-token window)
  C. fixed-size windows, e5              (100 words, 25 overlap - the "default" RAG chunking)
  D. overview chunk, e5                  (structure-aware: title, tags, ingredients)
  E. overview + method chunks, e5        (structure-aware, steps with a title header)

Embedding the full 235k-recipe corpus five times would take ~15 h on the
laptop CPU this was built on, so the comparison runs on a sub-corpus:
all Moroccan recipes + up to 40 relevant recipes per query + 15k random Food.com
recipes as distractors. Every strategy sees exactly the same corpus.
Constraint queries are excluded (they test filters, not chunking).
"""
from __future__ import annotations

import json
import logging
import time

import numpy as np
import pandas as pd

from eval.judge import EVAL_DIR, aggregate, grade, load_queries, query_metrics
from eval.evaluate_retrieval import recipe_dict
from recipe_rag.build_index import embed_cached
from recipe_rag.config import model_slug, settings
from recipe_rag.chunking import build_chunks
from recipe_rag.dense import Embedder
from recipe_rag.ingest import load_recipes

log = logging.getLogger("chunking_experiment")
E5, MINILM = "intfloat/multilingual-e5-small", "sentence-transformers/all-MiniLM-L6-v2"
CACHE = settings.paths.index_dir / "cache" / "experiments"


def v1_document(r) -> str:
    """Exactly the v1 document format (src/preprocessing.build_recipe_document)."""
    steps = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(r["steps"]))
    return (f"\n    Title: {r['name']}\n    Tags: {', '.join(r['tags'])}\n    Ingredients:\n"
            f"    {chr(10).join(r['ingredients'])}\n\n    Instructions:\n    {steps}\n    ")


def windows(text: str, size: int = 100, overlap: int = 25) -> list[str]:
    words = text.split()
    step = size - overlap
    return [" ".join(words[i:i + size]) for i in range(0, max(len(words) - overlap, 1), step)]


def sub_corpus(recipes: pd.DataFrame, queries, qrels, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx_by_id = pd.Series(np.arange(len(recipes)), index=recipes["recipe_id"])
    keep = set(np.flatnonzero(recipes["is_moroccan"].to_numpy()))
    for q in queries:
        ids = list(qrels[q["id"]])
        rng.shuffle(ids)
        keep |= set(idx_by_id[ids[:40]])
    foodcom = np.flatnonzero(~recipes["is_moroccan"].to_numpy())
    keep |= set(rng.choice(foodcom, 15_000, replace=False))
    return np.array(sorted(keep))


def evaluate(name, embedder, unit_texts, unit_recipe, recipes, queries, qrels, rows, vecs=None):
    t0 = time.perf_counter()
    if vecs is None:
        vecs = embed_cached(unit_texts, embedder, CACHE / name / model_slug(embedder.model_name))
    log.info("%s: %d units embedded/loaded in %.0fs", name, len(vecs), time.perf_counter() - t0)
    for q in queries:
        qv = embedder.encode_queries([q["query"]])[0]
        scores = vecs @ qv
        order = np.argsort(-scores)
        seen, top = set(), []
        for u in order:  # max-pool units -> recipes
            r = unit_recipe[u]
            if r not in seen:
                seen.add(r)
                top.append(r)
                if len(top) == 10:
                    break
        grades = [grade(q, recipe_dict(recipes, r)) for r in top]
        rel = qrels[q["id"]]
        n2 = sum(g == 2 for g in rel.values())
        rows.append({"strategy": name, "qid": q["id"], "category": q["category"],
                     "n_units": len(vecs), **query_metrics(grades, n2, len(rel) - n2)})


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    recipes = load_recipes()
    queries = [q for q in load_queries() if q.get("relevant") and q["category"] != "constraints"]
    qrels = json.loads((EVAL_DIR / "qrels.json").read_text(encoding="utf-8"))
    sub = sub_corpus(recipes, queries, qrels)
    in_sub = set(sub)
    # qrels restricted to the sub-corpus, so ideal DCG is computed on what is findable
    ids_in = set(recipes["recipe_id"].to_numpy()[sub])
    qrels = {k: {r: g for r, g in v.items() if r in ids_in} for k, v in qrels.items()}
    log.info("Sub-corpus: %d recipes", len(sub))

    rows = []
    e5 = Embedder(E5)
    sub_df = recipes.iloc[sub]
    docs = [v1_document(r._asdict()) for r in sub_df.itertuples(index=False)]

    evaluate("A_whole_minilm", Embedder(MINILM), docs, sub, recipes, queries, qrels, rows)
    evaluate("B_whole_e5", e5, docs, sub, recipes, queries, qrels, rows)

    win_text, win_recipe = [], []
    for d, r in zip(docs, sub):
        for w in windows(d):
            win_text.append(w)
            win_recipe.append(r)
    evaluate("C_windows_e5", e5, win_text, np.array(win_recipe), recipes, queries, qrels, rows)

    # Overview vectors are reused from the production index cache (same texts, same
    # model): the overview chunks of the full corpus are stored in recipe order.
    all_overview = build_chunks(recipes, with_method=False)["text"].tolist()
    cache = settings.paths.index_dir / "cache" / model_slug(E5) / "overview"
    n_shards = (len(all_overview) + 9_999) // 10_000
    while len(list(cache.glob("shard_*.npy"))) < n_shards:  # production build still running
        log.info("Waiting for the production overview embeddings (%d/%d shards)...",
                 len(list(cache.glob("shard_*.npy"))), n_shards)
        time.sleep(120)
    ov_vecs = embed_cached(all_overview, e5, settings.paths.index_dir / "cache" / model_slug(E5) / "overview")[sub]
    evaluate("D_overview_e5", e5, None, sub, recipes, queries, qrels, rows, vecs=ov_vecs)

    chunks = build_chunks(recipes.iloc[sub].reset_index(drop=True), with_method=True)
    method = chunks[chunks["chunk_type"] == "method"]
    method_recipe = sub[method["recipe_idx"].to_numpy()]
    m_vecs = embed_cached(method["text"].tolist(), e5, CACHE / "E_method" / model_slug(E5))
    evaluate("E_overview_method_e5", e5, None, np.concatenate([sub, method_recipe]),
             recipes, queries, qrels, rows, vecs=np.vstack([ov_vecs, m_vecs]))

    out = EVAL_DIR / "results"
    out.mkdir(exist_ok=True)
    (out / "chunking.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    cats = ["moroccan_fr", "darija", "intl_en", "intl_fr", "ingredients"]
    lines = [f"Sub-corpus of {len(in_sub)} recipes, {len(queries)} queries, dense retrieval only.", "",
             "| Strategy | vectors | nDCG@10 | MRR@10 | Hit@5 | " + " | ".join(cats) + " |",
             "|---|---|---|---|---|" + "---|" * len(cats)]
    for s in dict.fromkeys(r["strategy"] for r in rows):
        rs = [r for r in rows if r["strategy"] == s]
        a = aggregate(rs)
        per = [f"{aggregate([r for r in rs if r['category'] == c])['ndcg@10']:.3f}" for c in cats]
        lines.append(f"| {s} | {rs[0]['n_units']} | {a['ndcg@10']:.3f} | {a['mrr@10']:.3f} "
                     f"| {a['hit@5']:.3f} | " + " | ".join(per) + " |")
    md = "\n".join(lines) + "\n"
    (out / "chunking.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
