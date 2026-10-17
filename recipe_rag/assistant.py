"""End-to-end RAG orchestration.

    message + history
        -> query understanding (standalone question, intent, filters)
        -> hybrid retrieval + rerank (RecipeSearchEngine)
        -> confidence gate
        -> grounded generation with citations
        -> answer + sources (shown as recipe cards in the UI)

Degrades gracefully: without an LLM key the assistant still answers, in
"retrieval-only" mode, by listing the best recipes.
"""
from __future__ import annotations

import logging
import math
import re
import time
from dataclasses import asdict, dataclass, field

from recipe_rag.config import settings
from recipe_rag.llm import LLMClient, LLMError
from recipe_rag.prompts import (ANSWER_SYSTEM_PROMPT, GREETING, LOW_CONFIDENCE_NOTE,
                                NO_LLM_HEADER, OFF_TOPIC, build_user_prompt,
                                format_recipe, localized)
from recipe_rag.query_understanding import QueryAnalysis, QueryAnalyzer
from recipe_rag.retriever import RecipeSearchEngine, SearchOptions, SearchResult

log = logging.getLogger(__name__)


@dataclass
class AssistantResponse:
    answer: str
    sources: list[dict] = field(default_factory=list)
    analysis: dict | None = None
    confidence: float | None = None
    low_confidence: bool = False
    expanded_terms: dict = field(default_factory=dict)
    timings_ms: dict = field(default_factory=dict)
    mode: str = "rag"  # rag | retrieval_only | canned

    def to_dict(self) -> dict:
        return asdict(self)


class RecipeAssistant:
    def __init__(self, engine: RecipeSearchEngine, llm: LLMClient, analyzer: QueryAnalyzer):
        self.engine, self.llm, self.analyzer = engine, llm, analyzer
        self.cfg = settings.retrieval

    @classmethod
    def load(cls) -> "RecipeAssistant":
        llm = LLMClient(settings.llm)
        if not llm.available:
            log.warning("GROQ_API_KEY not set: running in retrieval-only mode")
        analyzer = QueryAnalyzer(llm if settings.retrieval.use_query_understanding else None,
                                 settings.llm.query_model)
        return cls(RecipeSearchEngine.load(), llm, analyzer)

    # ------------------------------------------------------------------
    def answer(self, message: str, history: list[dict] | None = None) -> AssistantResponse:
        t0 = time.perf_counter()
        history = history or []
        analysis = self.analyzer.analyze(message, history,
                                         corpus_spellings=self.engine.lexicon.expand(message))
        t_qu = (time.perf_counter() - t0) * 1000
        lang = analysis.language

        if analysis.intent in ("off_topic", "greeting"):
            table = OFF_TOPIC if analysis.intent == "off_topic" else GREETING
            return AssistantResponse(answer=localized(table, lang), analysis=analysis.model_dump(),
                                     mode="canned", timings_ms={"query_understanding": t_qu})

        top_k = 6 if analysis.intent in ("meal_plan", "ingredients") else self.cfg.top_k
        result = self.engine.search(analysis.standalone_question, analysis,
                                    SearchOptions(top_k=top_k))
        low = result.confidence is not None and result.confidence < self.cfg.confidence_threshold
        sources = self._sources(result)
        timings = {"query_understanding": round(t_qu, 1), **result.timings_ms}

        if self.llm.available:
            try:
                t1 = time.perf_counter()
                text = self._generate(analysis, result, low, history)
                timings["generation"] = round((time.perf_counter() - t1) * 1000, 1)
                mode = "rag"
            except LLMError as exc:
                log.error("Generation failed, falling back to retrieval-only: %s", exc)
                text, mode = self._retrieval_only(sources, lang, low), "retrieval_only"
        else:
            text, mode = self._retrieval_only(sources, lang, low), "retrieval_only"

        timings["total"] = round((time.perf_counter() - t0) * 1000, 1)
        return AssistantResponse(answer=text, sources=sources, analysis=analysis.model_dump(),
                                 confidence=result.confidence, low_confidence=low,
                                 expanded_terms=result.expanded_terms, timings_ms=timings,
                                 mode=mode)

    # ------------------------------------------------------------------
    def _generate(self, a: QueryAnalysis, result: SearchResult, low: bool,
                  history: list[dict]) -> str:
        recipes = self.engine.recipes
        block = "\n\n".join(format_recipe(n, recipes.iloc[h.recipe_idx])
                            for n, h in enumerate(result.hits, 1))
        messages = [{"role": "system", "content": ANSWER_SYSTEM_PROMPT.format(language=a.language)}]
        for m in history[-4:]:  # short memory: enough for follow-ups, cheap in tokens
            messages.append({"role": m["role"], "content": m["content"][:1500]})
        messages.append({"role": "user", "content": build_user_prompt(
            a.standalone_question, block, low, describe_constraints(a))})
        return normalize_citations(self.llm.complete(messages))

    def _retrieval_only(self, sources: list[dict], lang: str, low: bool) -> str:
        lines = [localized(LOW_CONFIDENCE_NOTE if low else NO_LLM_HEADER, lang), ""]
        for s in sources:
            extra = f" - {s['minutes']} min" if s["minutes"] else ""
            lines.append(f"**[{s['n']}] {s['name']}**{extra}  \n"
                         f"{', '.join(s['ingredients'][:8])}{'…' if len(s['ingredients']) > 8 else ''}")
        return "\n".join(lines)

    def _sources(self, result: SearchResult) -> list[dict]:
        out = []
        for n, h in enumerate(result.hits, 1):
            r = self.engine.recipes.iloc[h.recipe_idx]
            out.append({
                "n": n, "recipe_id": r["recipe_id"], "name": r["name"],
                "source": r["source"], "is_moroccan": bool(r["is_moroccan"]),
                "minutes": None if math.isnan(r["minutes"]) else int(r["minutes"]),
                "calories": None if math.isnan(r["calories"]) else int(r["calories"]),
                "ingredients": list(r["ingredients"]),
                "score": h.rerank_score if h.rerank_score is not None else h.score,
            })
        return out


_NATIVE_CITATION = re.compile(r"【(\d+)(?:†[^】]*)?】")


def normalize_citations(text: str) -> str:
    """gpt-oss sometimes emits its native citation format 【2†L1-L3】; the UI expects [2]."""
    return _NATIVE_CITATION.sub(r"[\1]", text)


def describe_constraints(a: QueryAnalysis) -> str:
    parts = list(a.diet)
    if a.max_minutes:
        parts.append(f"max {a.max_minutes} minutes")
    if a.max_calories:
        parts.append(f"max {a.max_calories} kcal")
    if a.exclude_ingredients:
        parts.append("without " + ", ".join(a.exclude_ingredients))
    return "; ".join(parts)
