import pandas as pd

from recipe_rag.chunking import build_chunks, method_chunks, overview_text, uninformative_tags


def test_method_chunks_never_split_a_step():
    steps = [" ".join(["word"] * 50)] * 5  # 250 words
    chunks = method_chunks("Harira", steps, max_words=120)
    assert len(chunks) == 3
    assert chunks[0].startswith("How to make Harira (steps 1-2 of 5)")
    assert all(c.count("word") % 50 == 0 for c in chunks)


def test_overlong_step_is_its_own_chunk():
    chunks = method_chunks("X", ["a " * 500, "short step"], max_words=100)
    assert len(chunks) == 2


def test_overview_has_title_tags_and_ingredients():
    row = {"name": "Chicken Tagine", "description": "A classic.", "tags": ["moroccan", "preparation"],
           "ingredients": ["chicken", "olives"]}
    text = overview_text(row, drop_tags={"preparation"}, max_description_words=10)
    assert text.splitlines()[0] == "Chicken Tagine"
    assert "Tags: moroccan" in text and "preparation" not in text
    assert text.endswith("Ingredients: chicken, olives")


def test_uninformative_tags_by_document_frequency():
    tags = pd.Series([["preparation", "moroccan"], ["preparation"], ["preparation", "30-minutes-or-less"]])
    drop = uninformative_tags(tags, max_df=0.5)
    assert "preparation" in drop and "30-minutes-or-less" in drop and "moroccan" not in drop


def test_build_chunks_links_chunks_to_parent_recipe():
    recipes = pd.DataFrame([{
        "name": "Harira", "description": "", "tags": ["moroccan"],
        "ingredients": ["lentils"], "steps": ["Boil.", "Serve."],
    }])
    chunks = build_chunks(recipes)
    assert list(chunks["chunk_type"]) == ["overview", "method"]
    assert set(chunks["recipe_idx"]) == {0}
