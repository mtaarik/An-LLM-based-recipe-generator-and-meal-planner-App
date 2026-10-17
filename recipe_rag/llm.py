"""Thin wrapper around the Groq chat-completions API.

Kept deliberately small: one place to handle the API key, timeouts, retries,
JSON mode and model fallback, so the rest of the code never imports the SDK
directly and the provider can be swapped (any OpenAI-compatible endpoint works
the same way).

Model fallback: free tiers have per-model daily token quotas. When a model is
rate-limited (HTTP 429) or retired, the next model of `LLMConfig.fallbacks` is
tried instead of failing the request.
"""
from __future__ import annotations

import logging
import time

from recipe_rag.config import LLMConfig

log = logging.getLogger(__name__)


class LLMError(RuntimeError):
    pass


def _is_unavailable(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(k in text for k in ("429", "rate limit", "rate_limit", "model_not_found",
                                   "decommissioned", "503", "over capacity"))


class LLMClient:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self._client = None
        if cfg.api_key:
            from groq import Groq
            self._client = Groq(api_key=cfg.api_key, timeout=cfg.timeout_s, max_retries=2)

    @property
    def available(self) -> bool:
        return self._client is not None

    def complete(self, messages: list[dict], model: str | None = None,
                 temperature: float | None = None, max_tokens: int | None = None,
                 json_mode: bool = False, schema: dict | None = None) -> str:
        if not self.available:
            raise LLMError("No GROQ_API_KEY configured")
        model = model or self.cfg.model
        chain = [model] + [m for m in self.cfg.fallbacks.get(model, ()) if m != model]
        for i, m in enumerate(chain):
            try:
                return self._complete_once(messages, m, temperature, max_tokens, json_mode, schema)
            except LLMError as exc:
                if i + 1 < len(chain) and _is_unavailable(exc.__cause__ or exc):
                    log.warning("%s unavailable (%s), falling back to %s", m,
                                str(exc)[:80], chain[i + 1])
                    continue
                raise
        raise LLMError("no model available")  # pragma: no cover

    def _complete_once(self, messages, model, temperature, max_tokens, json_mode, schema) -> str:
        kwargs = dict(
            model=model,
            messages=messages,
            temperature=self.cfg.temperature if temperature is None else temperature,
            max_tokens=max_tokens or self.cfg.max_tokens,
        )
        gpt_oss = kwargs["model"].startswith("openai/gpt-oss")
        if schema is not None and gpt_oss:
            # Constrained decoding: the output is guaranteed to match the schema.
            kwargs["response_format"] = {"type": "json_schema", "json_schema": {
                "name": "output", "strict": True, "schema": schema}}
        elif json_mode or schema is not None:
            kwargs["response_format"] = {"type": "json_object"}
        if gpt_oss:
            # Reasoning models: hidden chain-of-thought counts in max_tokens.
            kwargs["reasoning_effort"] = self.cfg.reasoning_effort
            kwargs["max_tokens"] += 1000
        t0 = time.perf_counter()
        try:
            resp = self._client.chat.completions.create(**kwargs)
        except Exception as exc:  # SDK raises many subclasses; normalise them
            raise LLMError(str(exc)) from exc
        log.debug("LLM %s answered in %.2fs", kwargs["model"], time.perf_counter() - t0)
        return resp.choices[0].message.content or ""

    def complete_json(self, messages: list[dict], model: str | None = None,
                      schema: dict | None = None) -> str:
        return self.complete(messages, model=model, temperature=0.0, max_tokens=600,
                             json_mode=True, schema=schema)
