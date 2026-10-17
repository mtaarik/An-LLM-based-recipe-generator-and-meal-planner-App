from recipe_rag.lexicon import VariantLexicon, is_common_word, skeleton, translit_key

TITLES = (
    ["Moroccan Tangia Marrakchia"]
    + ["Chicken Tagine with Olives"] * 30 + ["Kefta Tajine"] * 5
    + ["Chicken Bastilla Pie", "Seafood Bastilla", "Pastilla au poulet"]
    + ["Rfissa Moroccan Chicken", "Chicken rfissa", "Rfissa Express"]
    + ["Tomato Paste Sauce"] * 20 + ["Pasta Bake"] * 200
    + ["Tangy Chicken Wings"] * 15 + ["Chebakia", "Moroccan Chebakia cookies"]
)


def lexicon():
    return VariantLexicon.from_titles(TITLES, stopwords={"comment", "faire", "une"})


def test_translit_key_merges_spellings():
    assert translit_key("tanjia") == translit_key("tangia")
    assert translit_key("tajine") == translit_key("tagine")
    assert translit_key("bastilla") == translit_key("pastilla")
    assert translit_key("chebakia") == translit_key("chebakya")
    assert skeleton("beghrir") == skeleton("baghrir")


def test_tanjia_maps_to_corpus_spelling():
    # The original failure case: "tanjia" never appears in the corpus.
    assert lexicon().expand("comment faire une tanjia")["tanjia"] == ["tangia"]


def test_unknown_spellings_are_corrected():
    lex = lexicon()
    assert "rfissa" in lex.variants("rafissa")
    assert "bastilla" in lex.variants("bstilla")
    assert "chebakia" in lex.variants("chebakya")


def test_tajine_maps_to_dominant_spelling_only():
    lex = lexicon()
    assert lex.variants("tajine") == ["tagine"]
    assert lex.variants("tagine") == []


def test_real_words_are_not_corrected():
    lex = lexicon()
    assert lex.variants("paste") == []          # frequent word, not a typo of "pasta"
    assert "tangy" not in lex.variants("tanjia")  # same skeleton, too different


def test_common_french_words_are_not_fuzzy_corrected():
    lex = VariantLexicon.from_titles(["Poison Ivy Cocktail"] * 5 + ["Facial Mask"] * 5)
    assert is_common_word("poisson") and is_common_word("facile")
    assert lex.expand("tajine de poisson facile") == {}


def test_save_load_roundtrip(tmp_path):
    lex = lexicon()
    lex.save(tmp_path / "lex.json")
    again = VariantLexicon.load(tmp_path / "lex.json")
    assert again.expand("tanjia") == lex.expand("tanjia")
