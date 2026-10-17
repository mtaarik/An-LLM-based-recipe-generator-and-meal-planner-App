"""Prompt templates for grounded answer generation.

v1's prompt told the model: "If the recipes do not fully address the
question, you may use your general culinary knowledge (don't mention that you
have used your general knowledge)". Combined with irrelevant retrieved
recipes, that sentence is a licence to hallucinate *silently*. The rules
below do the opposite: every recipe fact must come from a numbered source,
general tips are allowed but must be labelled, and when retrieval confidence
is low the model must say the dish is not in the database.
"""
from __future__ import annotations

import math

ANSWER_SYSTEM_PROMPT = """You are a cooking assistant specialised in Moroccan and international cuisine.
You answer with the RECIPES retrieved from a recipe database, given in the user message.

Rules:
1. Answer in the user's language: {language}.
2. Ground every recipe fact (ingredients, quantities, times, temperatures, steps) in the
   provided recipes and cite them with [1], [2]... right after the facts they support.
   Use exactly this citation format: a number in square brackets, nothing else.
3. Never invent ingredients, quantities or steps that are not in the sources. Translating
   or summarising them is fine. Keep the quantities and units of the source exactly
   (e.g. "1 cup"); you may add a conversion in parentheses ("1 cup (~240 ml)").
   Do not combine quantities from different recipes into one ingredient list.
4. You may add at most one general cooking tip that is not in the sources, on a separate
   line starting with "💡" and without a citation, so the user can tell it apart.
5. If the retrieved recipes do not contain the requested dish (in particular when
   RETRIEVAL CONFIDENCE is LOW), say clearly and first that the database does not have a
   recipe for it. Then, optionally, propose the closest retrieved recipes. Do NOT write a
   recipe from memory.
6. Respect the user's constraints (diet, time, excluded ingredients) and warn if a source
   recipe does not satisfy one of them.
7. Format: one-sentence intro, then the ingredients as a bullet list and the steps as a
   numbered list. Be concise. For ingredient-based or meal-plan questions, propose 2-4
   options with one line each instead of a single full recipe."""


def _fmt_minutes(m: float) -> str:
    if m is None or (isinstance(m, float) and math.isnan(m)):
        return ""
    m = int(m)
    return f"{m // 60} h {m % 60:02d} min" if m >= 60 else f"{m} min"


def format_recipe(n: int, r, max_steps: int = 20) -> str:
    meta = [("Moroccan recipes dataset" if r["source"] == "moroccan" else "Food.com")]
    if t := _fmt_minutes(r["minutes"]):
        meta.append(f"total time {t}")
    if not math.isnan(r["calories"]):
        meta.append(f"~{int(r['calories'])} kcal/serving")
    lines = [f"[{n}] {r['name']} ({'; '.join(meta)})"]
    if r["description"]:
        lines.append("Description: " + " ".join(r["description"].split()[:60]))
    lines.append("Ingredients:")
    lines += [f"- {i}" for i in r["ingredients"]]
    lines.append("Steps:")
    steps = list(r["steps"])
    lines += [f"{i}. {s}" for i, s in enumerate(steps[:max_steps], 1)]
    if len(steps) > max_steps:
        lines.append(f"... ({len(steps) - max_steps} more steps)")
    return "\n".join(lines)


def build_user_prompt(question: str, recipes_block: str, low_confidence: bool,
                      constraints: str) -> str:
    conf = ("LOW - the recipes below probably do NOT match the request"
            if low_confidence else "HIGH")
    parts = [f"RETRIEVAL CONFIDENCE: {conf}"]
    if constraints:
        parts.append(f"USER CONSTRAINTS: {constraints}")
    parts += ["RECIPES:", recipes_block or "(no recipe found)", "", f"QUESTION: {question}"]
    return "\n".join(parts)


OFF_TOPIC = {
    "fr": "Je suis un assistant culinaire 🍳 : je peux vous aider à trouver des recettes "
          "(marocaines ou internationales), à cuisiner avec ce que vous avez, ou à "
          "adapter un plat à un régime. Que voulez-vous cuisiner ?",
    "en": "I'm a cooking assistant 🍳: I can help you find recipes (Moroccan or "
          "international), cook with what you have, or adapt a dish to a diet. "
          "What would you like to cook?",
}
GREETING = {
    "fr": "Bonjour ! 👋 Dites-moi un plat (tajine, pastilla, lasagnes…) ou les ingrédients "
          "que vous avez, et je vous propose une recette.",
    "en": "Hello! 👋 Tell me a dish (tagine, pastilla, lasagna…) or the ingredients you have, "
          "and I'll suggest a recipe.",
}
NO_LLM_HEADER = {
    "fr": "Voici les recettes les plus pertinentes de la base (mode recherche seule, sans "
          "génération) :",
    "en": "Here are the most relevant recipes in the database (retrieval-only mode, no "
          "generation):",
}
LOW_CONFIDENCE_NOTE = {
    "fr": "⚠️ Je n'ai pas trouvé de recette correspondant vraiment à votre demande ; voici "
          "les plus proches.",
    "en": "⚠️ I couldn't find a recipe that really matches your request; here are the closest ones.",
}


def localized(table: dict, language: str) -> str:
    return table.get(language, table["en"])
