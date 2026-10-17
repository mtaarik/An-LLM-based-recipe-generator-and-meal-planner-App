"""Export the texts that still need embedding, for a GPU run on Google Colab.

    python -m scripts.colab.export_colab_jobs        -> colab_jobs.parquet (upload to Colab)

The laptop CPU embeds ~25 chunks/s; a free Colab T4 GPU does thousands. This
writes one Parquet file containing every text set plus a manifest of jobs
(model, output folder, shards still missing locally, and the expected hash of
each shard). The notebook `embed_on_colab.ipynb` writes `.npy` shards named
exactly like the local cache (`build_index.embed_cached`), so after unzipping
the results into `data/index/cache/` the local pipeline simply finds them.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from eval.chunking_experiment import E5, sub_corpus, v1_document, windows
from eval.judge import EVAL_DIR, load_queries
from recipe_rag.build_index import SHARD
from recipe_rag.chunking import build_chunks
from recipe_rag.config import PROJECT_ROOT, model_slug, settings
from recipe_rag.ingest import load_recipes

E5_BASE = "intfloat/multilingual-e5-base"
CACHE = settings.paths.index_dir / "cache"


def shard_digests(texts: list[str]) -> dict[int, str]:
    return {i: hashlib.md5("\x1f".join(texts[i * SHARD:(i + 1) * SHARD]).encode()).hexdigest()[:12]
            for i in range((len(texts) + SHARD - 1) // SHARD)}


def main(out: Path = PROJECT_ROOT / "colab_jobs.parquet") -> None:
    recipes = load_recipes()
    chunks = build_chunks(recipes, with_method=True)
    sets = {
        "overview": chunks.loc[chunks["chunk_type"] == "overview", "text"].tolist(),
        "method": chunks.loc[chunks["chunk_type"] == "method", "text"].tolist(),
    }

    # Chunking-experiment texts, built exactly as eval/chunking_experiment.py does.
    queries = [q for q in load_queries() if q.get("relevant") and q["category"] != "constraints"]
    qrels = json.loads((EVAL_DIR / "qrels.json").read_text(encoding="utf-8"))
    sub = sub_corpus(recipes, queries, qrels)
    docs = [v1_document(r._asdict()) for r in recipes.iloc[sub].itertuples(index=False)]
    sets["B_docs"] = docs
    sets["C_windows"] = [w for d in docs for w in windows(d)]
    sub_chunks = build_chunks(recipes.iloc[sub].reset_index(drop=True), with_method=True)
    sets["E_method"] = sub_chunks.loc[sub_chunks["chunk_type"] == "method", "text"].tolist()

    plan = [  # most important first: if Colab disconnects, the rest can be resumed
        (E5, f"{model_slug(E5)}/overview", "overview"),
        (E5, f"experiments/B_whole_e5/{model_slug(E5)}", "B_docs"),
        (E5, f"experiments/C_windows_e5/{model_slug(E5)}", "C_windows"),
        (E5, f"experiments/E_method/{model_slug(E5)}", "E_method"),
        (E5, f"{model_slug(E5)}/method", "method"),
        (E5_BASE, f"{model_slug(E5_BASE)}/overview", "overview"),
    ]
    jobs, used = [], set()
    for model, out_dir, text_set in plan:
        digests = shard_digests(sets[text_set])
        missing = {i: d for i, d in digests.items()
                   if not (CACHE / out_dir / f"shard_{i:04d}_{d}.npy").exists()}
        if missing:
            jobs.append({"model": model, "out_dir": out_dir, "text_set": text_set,
                         "digests": {str(i): d for i, d in missing.items()}})
            used.add(text_set)
        print(f"{out_dir:55} {len(missing):3d}/{len(digests)} shards to embed")

    rows = [(name, i, t) for name in used for i, t in enumerate(sets[name])]
    table = pa.table({"text_set": [r[0] for r in rows], "idx": np.array([r[1] for r in rows], dtype="int32"),
                      "text": [r[2] for r in rows]})
    table = table.replace_schema_metadata({"jobs": json.dumps(jobs)})
    pq.write_table(table, out, compression="zstd")
    print(f"\n{len(rows):,} texts, {len(jobs)} jobs -> {out} ({out.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
