"""Turn a raw user message into a structured search request.

One small-LLM call (structured output constrained by a JSON schema) returns:
  * a standalone question (resolves follow-ups like "et sans oignons ?"
    using the chat history - "conversational query rewriting"),
  * the intent (recipe lookup, ingredient-based, meal plan, off-topic...),
  * an English, keyword-rich search query (the corpus is English while users
    write French / Darija / English),
  * the dishes mentioned, with their usual transliterations,
  * structured constraints (diet, max time, max calories, excluded
    ingredients) used as metadata filters ("self-querying retrieval").

Robustness:
  * explicit constraints are ALSO extracted by high-precision regexes and
    merged in (the LLM occasionally drops "vegetarian" from a long sentence);
  * if the LLM is unavailable (no key, timeout, invalid output) the
    rule-based analysis is used alone, so the system degrades gracefully
    instead of failing: retrieval still works on the raw query + lexicon.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, field_validator

from recipe_rag.llm import LLMClient, LLMError

log = logging.getLogger(__name__)

INTENTS = ["recipe", "ingredients", "meal_plan", "cooking_question", "off_topic", "greeting"]
DIETS = ["vegetarian", "vegan", "gluten-free"]
Intent = Literal["recipe", "ingredients", "meal_plan", "cooking_question", "off_topic", "greeting"]
Diet = Literal["vegetarian", "vegan", "gluten-free"]


class QueryAnalysis(BaseModel):
    standalone_question: str
    language: str = "fr"
    intent: Intent = "recipe"
    search_query: str = ""
    dish_names: list[str] = Field(default_factory=list)
    include_ingredients: list[str] = Field(default_factory=list)
    exclude_ingredients: list[str] = Field(default_factory=list)
    diet: list[Diet] = Field(default_factory=list)
    max_minutes: int | None = None
    max_calories: int | None = None
    cuisine: str | None = None
    source: Literal["llm", "rules"] = "rules"

    @field_validator("dish_names", mode="before")
    @classmethod
    def _flatten_dishes(cls, v):
        """Accept ["tangia"] or [{"name": "tanjia", "variants": ["tangia"]}]."""
        out = []
        for d in v or []:
            if isinstance(d, str):
                out.append(d)
            elif isinstance(d, dict):
                out += [d.get("name") or d.get("canonical") or "", *(d.get("variants") or [])]
        return list(dict.fromkeys(x.strip() for x in out if x and x.strip()))

    @field_validator("diet", mode="before")
    @classmethod
    def _diet(cls, v):
        return [d for d in (v or []) if d in DIETS]

    @field_validator("intent", mode="before")
    @classmethod
    def _intent(cls, v):
        return v if v in INTENTS else "recipe"

    @field_validator("max_minutes", "max_calories", mode="before")
    @classmethod
    def _positive_int(cls, v):
        try:
            v = int(float(v))
            return v if v > 0 else None
        except (TypeError, ValueError):
            return None

    @property
    def has_filters(self) -> bool:
        return bool(self.diet or self.max_minutes or self.max_calories or self.exclude_ingredients)


SYSTEM_PROMPT = """You are the query-understanding module of a recipe search engine.
The recipe database is in ENGLISH and contains international recipes (Food.com) and
Moroccan recipes. Users write in French, English, Moroccan Darija or a mix.

Fill the JSON fields:
- standalone_question: the user's latest message rewritten as a self-contained question,
  in the user's language, resolving references to the conversation history
  (e.g. "et sans lentilles ?" after a harira recipe -> "Recette de harira sans lentilles").
- language: ISO code of the user's language ("fr", "en", "ar"...).
- intent: "recipe" (a specific dish), "ingredients" (ideas from ingredients they have),
  "meal_plan", "cooking_question" (technique, substitution, timing...), "greeting",
  "off_topic" (not about food or cooking).
- search_query: short ENGLISH keyword query for the recipe database: dish name(s) with their
  usual spellings, main ingredients, cuisine, cooking style. No filler words.
- dishes: dishes mentioned or implied. name = canonical name; variants = other common
  transliterations (e.g. pastilla: bastilla, bstilla, b'stilla); description = a few
  English words describing the dish.
  If you are not SURE what a dish is, keep the user's dish name (and the database
  spellings given below) and leave the description empty: never guess its ingredients.
  A wrong guess (e.g. reading "harsha", a semolina flatbread, as "harira", a soup)
  sends the search to the wrong recipes.
- When "spellings found in the recipe database" are given, they are how the user's words
  are written in the database: include them in search_query.
- include_ingredients / exclude_ingredients: in English, singular.
- diet: subset of ["vegetarian", "vegan", "gluten-free"], ONLY if explicitly requested.
- max_minutes / max_calories: integer limits ONLY if explicitly requested, else null.
- cuisine: cuisine if requested (e.g. "moroccan"), else null.
Never invent constraints the user did not express."""

_STR_LIST = {"type": "array", "items": {"type": "string"}}
_NULLABLE_INT = {"anyOf": [{"type": "integer"}, {"type": "null"}]}
LLM_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "standalone_question": {"type": "string"},
        "language": {"type": "string"},
        "intent": {"type": "string", "enum": INTENTS},
        "search_query": {"type": "string"},
        "dishes": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "properties": {"name": {"type": "string"}, "variants": _STR_LIST,
                           "description": {"type": "string"}},
            "required": ["name", "variants", "description"]}},
        "include_ingredients": _STR_LIST,
        "exclude_ingredients": _STR_LIST,
        "diet": {"type": "array", "items": {"type": "string", "enum": DIETS}},
        "max_minutes": _NULLABLE_INT,
        "max_calories": _NULLABLE_INT,
        "cuisine": {"anyOf": [{"type": "string"}, {"type": "null"}]},
    },
}
LLM_SCHEMA["required"] = list(LLM_SCHEMA["properties"])
PROMPT_VERSION = hashlib.md5((SYSTEM_PROMPT + json.dumps(LLM_SCHEMA)).encode()).hexdigest()[:8]


class QueryAnalyzer:
    def __init__(self, llm: LLMClient | None, model: str):
        self.llm = llm
        self.model = model

    def analyze(self, message: str, history: list[dict] | None = None,
                corpus_spellings: dict[str, list[str]] | None = None) -> QueryAnalysis:
        """`corpus_spellings`: lexicon output ({"harsha": ["harcha"]}), given to the
        LLM as evidence of how the user's words are written in the database."""
        rules = rule_based_analysis(message)
        if self.llm is not None and self.llm.available:
            try:
                return merge_rules(self._analyze_llm(message, history or [], corpus_spellings or {}),
                                   rules)
            except (LLMError, ValidationError, json.JSONDecodeError) as exc:
                log.warning("Query understanding fell back to rules: %s", exc)
        return rules

    def _analyze_llm(self, message: str, history: list[dict],
                     corpus_spellings: dict[str, list[str]]) -> QueryAnalysis:
        convo = "\n".join(f"{m['role']}: {m['content'][:500]}" for m in history[-6:])
        user = (f"Conversation history:\n{convo}\n\n" if convo else "") + f"Latest user message:\n{message}"
        if corpus_spellings:
            user += "\n\nSpellings found in the recipe database: " + "; ".join(
                f"{w} -> {', '.join(v)}" for w, v in corpus_spellings.items())
        raw = self.llm.complete_json(
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}],
            model=self.model, schema=LLM_SCHEMA,
        )
        data = json.loads(raw)
        data["dish_names"] = data.pop("dishes", [])
        data["standalone_question"] = data.get("standalone_question") or message
        data["search_query"] = data.get("search_query") or message
        data["source"] = "llm"
        return QueryAnalysis.model_validate(data)


def merge_rules(llm: QueryAnalysis, rules: QueryAnalysis) -> QueryAnalysis:
    """Explicit constraints caught by the regexes are kept even if the LLM missed them."""
    diet = list(dict.fromkeys([*llm.diet, *rules.diet]))
    if "vegan" in diet and "vegetarian" in diet:
        diet.remove("vegetarian")
    return llm.model_copy(update={
        "diet": diet,
        "max_minutes": llm.max_minutes or rules.max_minutes,
        "max_calories": llm.max_calories or rules.max_calories,
    })


# --------------------------------------------------------------------------
# Rule-based fallback (high precision, low recall)
# --------------------------------------------------------------------------
_FR_HINTS = re.compile(r"\b(je|j'ai|comment|recette|avec|sans|pour|une?|des|du|le|la|les|moins|plat)\b", re.I)
_TIME = re.compile(r"(?:moins de|under|less than|en|in|max(?:imum)?)\s*(\d{1,3})\s*(?:min|minutes|mn)\b", re.I)
_HOURS = re.compile(r"(?:moins d'|under|less than|en|in)\s*(?:une?|1|one)\s*(?:heure|hour)\b", re.I)
_CAL = re.compile(r"(\d{2,4})\s*(?:kcal|calories|cal)\b", re.I)
_DIETS = {
    "vegan": re.compile(r"\bv[ée]g[ae]n[e]?s?\b|\bv[ée]g[ée]talien", re.I),
    "vegetarian": re.compile(r"\bv[ée]g[ée]tarien(?:ne)?s?\b|\bvegetarian\b|\bveggie\b", re.I),
    "gluten-free": re.compile(r"sans gluten|gluten[- ]free|no gluten", re.I),
}


def rule_based_analysis(message: str) -> QueryAnalysis:
    diet = [d for d, rx in _DIETS.items() if rx.search(message)]
    if "vegan" in diet and "vegetarian" in diet:
        diet.remove("vegetarian")
    minutes = None
    if m := _TIME.search(message):
        minutes = int(m.group(1))
    elif _HOURS.search(message):
        minutes = 60
    calories = int(m.group(1)) if (m := _CAL.search(message)) else None
    return QueryAnalysis(
        standalone_question=message,
        language="fr" if _FR_HINTS.search(message) else "en",
        search_query=message,
        diet=diet,
        max_minutes=minutes,
        max_calories=calories,
        source="rules",
    )
