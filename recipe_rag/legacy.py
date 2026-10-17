"""v1 retrieval, kept only as the evaluation baseline.

Faithful port of the original `src/search_recipes.py` + `src/rag_system.py`:
all-MiniLM-L6-v2 over one embedding per (truncated) recipe document, FAISS
L2 search, and the hard-coded query expansion. The only change is that the
index/model are loaded once instead of on every call.
"""
from __future__ import annotations

import ast

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from recipe_rag.config import settings
from recipe_rag.ingest import infer_diet


def preprocess_query2(query: str) -> str:
    """Verbatim v1 query expansion (the approach being replaced)."""
    query = query.lower().strip()
    expansions_to_add = set()
    expansions = {
        'tajine': ['tajine', 'tagine', 'moroccan', 'stew'],
        'tagine': ['tajine', 'tagine', 'moroccan', 'stew'],
        'couscous': ['couscous', 'moroccan', 'semolina', 'grain'],
        'marocain': ['moroccan', 'marocain', 'maghreb', 'north african'],
        'marocaine': ['moroccan', 'marocaine', 'maghreb', 'north african'],
        'kefta': ['kofta', 'spiced', 'meatball'],
        'pastilla': ['pastilla', 'pigeon', 'pie', 'filo', 'pastry'],
        'harira': ['harira', 'soup', 'lentil', 'chickpea'],
        'tanjia': ['tanjia', 'marrakchi', 'stew', 'lamb'],
    }
    current_terms = set(query.split())
    for term, expansion_list in expansions.items():
        if term in current_terms:
            expansions_to_add.update(expansion_list)
    final_terms = list(current_terms) + [t for t in expansions_to_add if t not in current_terms]
    return " ".join(final_terms)


class LegacyV1Retriever:
    def __init__(self, use_expansion: bool = True):
        d = settings.paths.legacy_index_dir
        self.meta = pd.read_csv(d / "recipe_metadata.csv")
        self.index = faiss.read_index(str(d / "recipe_index.faiss"))
        self.model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        self.use_expansion = use_expansion

    def search_ids(self, query: str, k: int = 10) -> list[int]:
        q = preprocess_query2(query) if self.use_expansion else query
        vec = self.model.encode([q], convert_to_numpy=True, normalize_embeddings=True)
        _, ids = self.index.search(vec, k)
        return [int(i) for i in ids[0] if i >= 0]

    def search(self, query: str, k: int = 10) -> list[dict]:
        return [self._as_recipe(i) for i in self.search_ids(query, k)]

    def _as_recipe(self, i: int) -> dict:
        """Convert a v1 metadata row to the fields the evaluation judge needs."""
        row = self.meta.iloc[i]
        try:
            ings = [str(x) for x in ast.literal_eval(row["ingredients"])]
        except (ValueError, SyntaxError):
            ings = str(row.get("ingredients_text") or "").split("\n")
        tags = str(row.get("tags") or "").lower()
        try:
            nutrition = ast.literal_eval(row["nutrition"])
            calories = float(nutrition[0])
        except (ValueError, SyntaxError, TypeError, IndexError):
            calories = np.nan
        return {
            "recipe_id": f"v1-{i}",
            "name": str(row["name"]),
            "ingredients": ings,
            "is_moroccan": "moroccan" in tags or "marocaine" in tags,
            **infer_diet(ings),
            "minutes": float(row["minutes"]) if pd.notna(row.get("minutes")) else np.nan,
            "calories": calories,
        }


V1_PROMPT = """
You are a helpful culinary assistant who answers cooking-related questions using recipe information as your primary reference.

Answer the question (the Question below that a user has writen to you . it's the only output tha he has given you this whole prompt and context it's just me )naturally and completely, without referencing that you're using a context or dataset. Combine and synthesize useful details from the recipes provided to give a clear, informative, and engaging response.

If the recipes do not fully address the question, you may use your general culinary knowledge to provide the best possible answer(don't mention that you have used your general knowledge) — but always prioritize and integrate any useful information found in the recipes.

Do not mention recipe numbers, IDs, or refer to the recipes explicitly. Just answer the question as if you're drawing from your own expertise.

Recipe Context:
{context}

Question: {query}

Answer:
"""


def v1_generate(llm, retriever: LegacyV1Retriever, query: str, top_k: int = 2) -> tuple[str, list[str]]:
    """Original v1 RAG answer: top-2 documents, temperature 0.9, 500 tokens."""
    ids = retriever.search_ids(query, top_k)
    docs = [str(retriever.meta.iloc[i]["document"]) for i in ids]
    context = "\n\n".join(f"Recette {n + 1} (Nom: {retriever.meta.iloc[i]['name']}):\n{d}"
                          for n, (i, d) in enumerate(zip(ids, docs)))
    answer = llm.complete(
        [{"role": "system", "content": "Tu es un assistant culinaire utile qui répond aux questions "
                                       "sur les recettes marocaines, en utilisant uniquement le contexte fourni."},
         {"role": "user", "content": V1_PROMPT.format(context=context, query=query)}],
        temperature=0.9, max_tokens=500)
    return answer, docs
