from eval.judge import grade, query_metrics
from recipe_rag.query_understanding import rule_based_analysis
from recipe_rag.retriever import rrf_fuse


def test_rrf_rewards_agreement_between_retrievers():
    fused = dict(rrf_fuse([[1, 2, 3], [3, 1, 4]], k=60))
    assert max(fused, key=fused.get) == 1     # ranked 1st and 2nd
    assert fused[3] > fused[2]                # in both lists beats a single 2nd place
    assert fused[4] < fused[2]


def test_rule_based_analysis_extracts_constraints():
    a = rule_based_analysis("un plat végétarien rapide, en moins de 30 minutes")
    assert a.language == "fr" and a.diet == ["vegetarian"] and a.max_minutes == 30
    b = rule_based_analysis("vegan chocolate dessert under 300 calories")
    assert b.diet == ["vegan"] and b.max_calories == 300
    assert rule_based_analysis("soupe sans gluten").diet == ["gluten-free"]


def test_rule_based_analysis_no_false_constraints():
    a = rule_based_analysis("comment faire une tanjia")
    assert not a.has_filters


RECIPE = {"name": "Moroccan Tangia Marrakchia", "ingredients": ["1 kg beef", "cumin"],
          "is_moroccan": True, "is_vegetarian": False, "is_vegan": False,
          "is_gluten_free": True, "minutes": 300.0, "calories": float("nan")}


def test_grade_rules():
    q = {"relevant": {"name": "tangia|tanjia"},
         "partial": {"moroccan": True, "ing": ["beef|lamb"]}}
    assert grade(q, RECIPE) == 2
    assert grade(q, {**RECIPE, "name": "Beef Tagine"}) == 1
    assert grade({"relevant": {"max_calories": 400}}, RECIPE) == 0  # unknown calories


def test_query_metrics():
    m = query_metrics([0, 2, 1], n_relevant=1, n_partial=1)
    assert m["mrr@10"] == 0.5 and m["hit@1"] == 0.0 and m["hit@5"] == 1.0
    assert 0 < m["ndcg@10"] < 1
    assert query_metrics([2, 1], 1, 1)["ndcg@10"] == 1.0


def test_llm_output_is_validated_leniently():
    from recipe_rag.query_understanding import QueryAnalysis
    a = QueryAnalysis.model_validate({
        "standalone_question": "q", "intent": "nonsense", "diet": ["vegetarian", "keto"],
        "dish_names": [{"name": "pastilla", "variants": ["bastilla", "bstilla"]}, "pastilla"],
        "max_minutes": "30", "max_calories": -5,
    })
    assert a.intent == "recipe" and a.diet == ["vegetarian"]
    assert a.dish_names == ["pastilla", "bastilla", "bstilla"]
    assert a.max_minutes == 30 and a.max_calories is None


def test_rules_complete_what_the_llm_missed():
    from recipe_rag.query_understanding import QueryAnalysis, merge_rules
    llm = QueryAnalysis(standalone_question="q", search_query="quick vegetarian dish", source="llm")
    merged = merge_rules(llm, rule_based_analysis("un plat végétarien en moins de 30 minutes"))
    assert merged.diet == ["vegetarian"] and merged.max_minutes == 30
    assert merged.search_query == "quick vegetarian dish" and merged.source == "llm"


class FakeLLM:
    available = True

    def __init__(self, reply):
        self.reply, self.messages = reply, None

    def complete_json(self, messages, model=None, schema=None):
        self.messages = messages
        return self.reply


def test_query_parser_is_grounded_with_corpus_spellings():
    import json
    from recipe_rag.query_understanding import QueryAnalyzer
    llm = FakeLLM(json.dumps({
        "standalone_question": "harsha", "language": "fr", "intent": "recipe",
        "search_query": "harcha semolina flatbread", "dishes": [
            {"name": "harcha", "variants": ["harsha"], "description": ""}],
        "include_ingredients": [], "exclude_ingredients": [], "diet": [],
        "max_minutes": None, "max_calories": None, "cuisine": None}))
    a = QueryAnalyzer(llm, "m").analyze("harsha", corpus_spellings={"harsha": ["harcha"]})
    assert "harsha -> harcha" in llm.messages[-1]["content"]
    assert a.source == "llm" and a.dish_names == ["harcha", "harsha"]


def test_query_parser_falls_back_on_invalid_output():
    from recipe_rag.query_understanding import QueryAnalyzer
    a = QueryAnalyzer(FakeLLM("not json"), "m").analyze("tajine végétarien")
    assert a.source == "rules" and a.diet == ["vegetarian"]


def test_native_citations_are_normalized():
    from recipe_rag.assistant import normalize_citations
    assert normalize_citations("1 cup flour【2†L1-L2】 and salt 【3】") == "1 cup flour[2] and salt [3]"


def test_llm_falls_back_to_next_model_on_rate_limit():
    from recipe_rag.config import LLMConfig
    from recipe_rag.llm import LLMClient, LLMError

    class Client(LLMClient):
        def __init__(self):
            self.cfg, self._client, self.calls = LLMConfig(api_key="x"), object(), []

        def _complete_once(self, messages, model, *args):
            self.calls.append(model)
            if model == "openai/gpt-oss-20b":
                raise LLMError("Error code: 429 - rate limit reached")
            return "ok"

    c = Client()
    assert c.complete([], model="openai/gpt-oss-20b") == "ok"
    assert c.calls == ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]
