# Notebooks (v1 exploration)

These notebooks are the original first-year exploration (data loading, preprocessing,
embedding generation, retrieval and RAG experiments). They are kept for history.

The production pipeline now lives in the `recipe_rag/` package and is run with:

```bash
python -m recipe_rag.build_index
```

The notebooks import the v1 `src/` modules, which were replaced by `recipe_rag/`
(see the git history before the `rag-v2` branch to run them as they were).
