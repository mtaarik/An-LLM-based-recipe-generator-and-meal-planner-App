from recipe_rag.ingest import (dedup_key, display_name, infer_diet, parse_list_field,
                               parse_moroccan_prompt)


def test_parse_list_literal():
    assert parse_list_field("['1. Chop onions', '2. Fry them#']") == ["Chop onions", "Fry them"]


def test_parse_dict_of_sections():
    raw = "{'Steps to Make It': ['Wash the fish.', 'Marinate it.']}"
    assert parse_list_field(raw) == ["Wash the fish.", "Marinate it."]


def test_parse_pipe_separated():
    assert parse_list_field("Heat oil.|Add\xa0onions.| |Serve.") == ["Heat oil.", "Add onions.", "Serve."]


def test_parse_nan_and_empty():
    assert parse_list_field("nan") == []
    assert parse_list_field(None) == []


def test_parse_moroccan_prompt_all_sections():
    prompt = (
        "Recipe Name : Moroccan Tangia Marrakchia Recipe\n"
        " Description : Slow-cooked beef.\n"
        " Ingredients : ['1 kg beef', 'preserved lemon']\n"
        " Total time for preparation : 300\n"
        " Steps for preparation : Put everything in the tangia.|Cook overnight.\n"
    )
    r = parse_moroccan_prompt(prompt)
    assert r["name"] == "Moroccan Tangia Marrakchia"
    assert r["ingredients"] == ["1 kg beef", "preserved lemon"]
    assert r["steps"] == ["Put everything in the tangia.", "Cook overnight."]
    assert r["minutes"] == 300


def test_parse_moroccan_prompt_falls_back_to_completion():
    prompt = "Recipe Name : Harira\n Ingredients : ['lentils']\n Steps for preparation : nan\n"
    assert parse_moroccan_prompt(prompt, "['Boil the lentils.']")["steps"] == ["Boil the lentils."]


def test_non_recipe_rows_are_skipped():
    assert parse_moroccan_prompt("Top 10 foods to try in Morocco : Harira - Tagine") is None


def test_display_name():
    assert display_name("arriba   baked winter squash") == "Arriba Baked Winter Squash"
    assert display_name("it's maaaaa-gic carrots") == "It's Maaaaa-gic Carrots"
    assert display_name("Chicken Tagine Recipe") == "Chicken Tagine"


def test_infer_diet():
    assert infer_diet(["chickpeas", "vegetable broth", "cumin"]) == {
        "is_vegetarian": True, "is_vegan": True, "is_gluten_free": True}
    assert not infer_diet(["chicken broth", "rice"])["is_vegetarian"]
    assert not infer_diet(["eggplant", "feta cheese"])["is_vegan"]
    assert infer_diet(["eggplant", "peanut butter"])["is_vegan"]
    assert not infer_diet(["couscous", "carrots"])["is_gluten_free"]
    assert infer_diet(["rice flour", "water"])["is_gluten_free"]
    assert infer_diet(["vegetarian sausage", "beans"])["is_vegetarian"]


def test_dedup_key_ignores_order_and_case():
    assert dedup_key("Chicken Tagine", ["Olives", "chicken"]) == dedup_key("chicken tagine", ["chicken", "olives"])
