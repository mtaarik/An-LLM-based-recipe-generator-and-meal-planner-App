"""Offline indexing pipeline (replaces notebooks 02-03).

    python -m recipe_rag.build_index                 # everything
    python -m recipe_rag.build_index --chunk-types overview method
    python -m recipe_rag.build_index --skip-ingest   # reuse recipes.parquet

Steps: raw CSVs -> recipes.parquet -> chunks -> embeddings (cached, resumable)
-> FAISS index, BM25 index, spelling-variant lexicon.

Embedding ~235k chunks takes a couple of hours on a laptop CPU, so
embeddings are computed in shards of 10k and cached on disk under a key made
of (model, chunk texts). Interrupting and re-running resumes where it stopped,
and changing the chunking only recomputes the shards whose text changed.
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from recipe_rag.chunking import build_chunks, full_text
from recipe_rag.config import settings
from recipe_rag.dense import DenseIndex, Embedder
from recipe_rag.ingest import build_recipe_table, load_recipes
from recipe_rag.lexicon import VariantLexicon
from recipe_rag.sparse import QUERY_NOISE, BM25Index
import bm25s

log = logging.getLogger("build_index")
SHARD = 10_000


def embed_cached(texts: list[str], embedder: Embedder, cache_dir: Path) -> np.ndarray:
    """Embed `texts` shard by shard, reusing shards already on disk."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = []
    n_shards = (len(texts) + SHARD - 1) // SHARD
    for i in range(n_shards):
        shard = texts[i * SHARD:(i + 1) * SHARD]
        digest = hashlib.md5("\x1f".join(shard).encode()).hexdigest()[:12]
        path = cache_dir / f"shard_{i:04d}_{digest}.npy"
        if path.exists():
            out.append(np.load(path).astype("float32"))
            continue
        t0 = time.perf_counter()
        vecs = embedder.encode_passages(shard, show_progress=False)
        np.save(path, vecs.astype("float16"))
        out.append(vecs)
        log.info("Shard %d/%d embedded (%.0f chunks/s)", i + 1, n_shards,
                 len(shard) / (time.perf_counter() - t0))
    return np.vstack(out) if out else np.zeros((0, embedder.dim), dtype="float32")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-ingest", action="store_true")
    parser.add_argument("--chunk-types", nargs="+", default=["overview"],
                        choices=["overview", "method"])
    parser.add_argument("--embed-only", action="store_true",
                        help="only compute/cache embeddings (no FAISS/BM25/lexicon)")
    parser.add_argument("--no-dense", action="store_true",
                        help="only build the BM25 index and the lexicon")
    parser.add_argument("--no-sparse", action="store_true",
                        help="skip the BM25 index and the lexicon")
    parser.add_argument("--embedding-model", default=settings.retrieval.embedding_model)
    args = parser.parse_args(argv)
    p, cfg = settings.paths, settings.retrieval
    p.index_dir.mkdir(parents=True, exist_ok=True)

    recipes = load_recipes() if args.skip_ingest and p.recipes_parquet.exists() else build_recipe_table()

    all_chunks = build_chunks(recipes, with_method="method" in args.chunk_types)
    chunks = all_chunks[all_chunks["chunk_type"].isin(args.chunk_types)]
    # Order by type so every chunk type has its own contiguous, cacheable block.
    chunks = (chunks.assign(_o=chunks["chunk_type"].map({"overview": 0, "method": 1}))
              .sort_values(["_o", "chunk_id"]).drop(columns="_o").reset_index(drop=True))
    chunks["chunk_id"] = np.arange(len(chunks))
    log.info("%d chunks: %s", len(chunks), chunks["chunk_type"].value_counts().to_dict())

    if not args.no_dense:
        build_dense(chunks, args)
    if args.embed_only or args.no_sparse:
        return
    build_sparse(recipes)


def build_dense(chunks: pd.DataFrame, args) -> None:
    p, model = settings.paths, args.embedding_model
    embedder = Embedder(model)
    vecs = [embed_cached(chunks.loc[chunks["chunk_type"] == t, "text"].tolist(), embedder,
                         p.embedding_cache(model, t))
            for t in ["overview", "method"] if t in args.chunk_types]
    if args.embed_only:
        return
    embeddings = np.vstack(vecs)

    out = p.dense_dir(model)
    DenseIndex.build(embeddings).save(out / "dense.faiss")
    chunks.to_parquet(out / "chunks.parquet", index=False)
    log.info("FAISS index (%s): %d vectors (dim %d) -> %s", model, len(embeddings),
             embeddings.shape[1], out)


def build_sparse(recipes: pd.DataFrame) -> None:
    p = settings.paths
    t0 = time.perf_counter()
    BM25Index.build([full_text(r._asdict()) for r in recipes.itertuples(index=False)]).save(p.bm25_dir)
    log.info("BM25 index built in %.0fs", time.perf_counter() - t0)

    VariantLexicon.from_titles(
        recipes["name"],
        stopwords=set(bm25s.stopwords.STOPWORDS_EN) | set(bm25s.stopwords.STOPWORDS_FRENCH) | set(QUERY_NOISE),
    ).save(p.lexicon_json)
    log.info("Done.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
