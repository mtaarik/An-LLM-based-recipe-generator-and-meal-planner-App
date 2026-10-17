"""Load the two raw datasets and normalise them into one clean recipe table.

Sources
-------
* Food.com "Recipes and Interactions" (Kaggle) – ~231k international recipes,
  with tags, cooking time and nutrition.
* Moroccan recipes dataset (Kaggle) – ~4.6k scraped Moroccan recipes stored as
  a free-text ``prompt`` column ("Recipe Name : ... Ingredients : [...] ...").

The Moroccan file is messy: steps come in three different encodings (Python
list literal, dict of sections, or ``|``-separated text). The v1 parser only
understood list literals, so ~15% of Moroccan recipes silently lost their
steps and were later dropped by the "empty steps" filter. This module parses
all three formats.

Output: ``data/processed/recipes.parquet`` with one row per unique recipe.
"""
from __future__ import annotations

import ast
import hashlib
import logging
import re
import string
from pathlib import Path

import numpy as np
import pandas as pd

from recipe_rag.config import settings

log = logging.getLogger(__name__)

NUTRITION_FIELDS = [
    "calories", "fat_pdv", "sugar_pdv", "sodium_pdv",
    "protein_pdv", "sat_fat_pdv", "carbs_pdv",
]
MAX_MINUTES = 60 * 24 * 30  # anything above 30 days is a data-entry error
MAX_CALORIES = 5000         # per serving

# --------------------------------------------------------------------------
# Text helpers
# --------------------------------------------------------------------------
_WS = re.compile(r"\s+")
_STEP_NUM = re.compile(r"^\s*(?:step\s*)?\d+\s*[.)-]\s*", re.IGNORECASE)


def clean_text(text: object) -> str:
    if not isinstance(text, str):
        return ""
    text = text.replace("\xa0", " ").replace("​", "")
    return _WS.sub(" ", text).strip().strip("#").strip()


def display_name(name: str) -> str:
    """Food.com names are all lower-case ('arriba   baked winter squash')."""
    name = clean_text(name)
    name = re.sub(r"\s+recipe$", "", name, flags=re.IGNORECASE)
    return string.capwords(name) if name.islower() else name


def _flatten(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for v in value.values() for s in _flatten(v)]
    if isinstance(value, (list, tuple)):
        return [s for v in value for s in _flatten(v)]
    return []


def parse_list_field(raw: object) -> list[str]:
    """Parse a list stored as a Python literal, a dict of sections, or '|' text."""
    if not isinstance(raw, str):
        return []
    raw = raw.strip()
    if not raw or raw.lower() == "nan":
        return []
    items: list[str] = []
    if raw[0] in "[{":
        try:
            items = _flatten(ast.literal_eval(raw))
        except (ValueError, SyntaxError):
            # Truncated / malformed literal: fall back to the quoted strings.
            items = re.findall(r"'((?:[^'\\]|\\.)*)'|\"((?:[^\"\\]|\\.)*)\"", raw)
            items = [a or b for a, b in items]
    if not items:
        items = raw.split("|") if "|" in raw else [raw]
    cleaned = (_STEP_NUM.sub("", clean_text(i)) for i in items)
    return [c for c in cleaned if len(c) > 1]


# --------------------------------------------------------------------------
# Dietary inference (consistent across both sources, which have very
# different tagging). Deliberately conservative: when unsure, not vegetarian.
# --------------------------------------------------------------------------
_PLANT_BASED = re.compile(r"\b(vegetarian|vegan|veggie|meatless|plant[- ]based|soy|tofu|seitan)\b")
_MEAT_FISH = re.compile(
    r"\b(chicken|beef|lamb|mutton|pork|bacon|ham|sausages?|turkey|duck|veal|goat|"
    r"prosciutto|pancetta|salami|pepperoni|chorizo|merguez|meat|meatballs?|steaks?|"
    r"kefta|kofta|liver|pigeon|quail|rabbit|venison|gelatin|lard|anchov(?:y|ies)|"
    r"fish|salmon|tuna|shrimps?|prawns?|crab|lobster|clams?|mussels?|oysters?|"
    r"scallops?|squid|calamari|octopus|cod|tilapia|sardines?|halibut|trout|"
    r"mackerel|sea ?bass|snapper|haddock|sole|swordfish|seafood|broth|stock|bouillon)\b"
)
_VEG_STOCK = re.compile(r"\b(vegetable|veggie|mushroom)\s+(broth|stock|bouillon)\b")
_ANIMAL = re.compile(
    r"\b(milk|butter|cheeses?|cream|yogh?urt|eggs?|honey|ghee|smen|buttermilk|whey|"
    r"mayonnaise|parmesan|mozzarella|feta|ricotta)\b"
)
_ANIMAL_EXCEPTIONS = re.compile(
    r"\b(peanut|almond|cashew|coconut|soy|oat|rice|cocoa|vegan|nut|seed)\s+(butter|milk|cream|cheese|yogh?urt)\b"
    r"|\beggplants?\b|\bcream of tartar\b|\bbutternut\b|\bbutter beans?\b|\begg[- ]free\b"
)
_GLUTEN = re.compile(
    r"\b(flour|wheat|bread|breadcrumbs?|pasta|spaghetti|noodles?|couscous|semolina|"
    r"barley|rye|bulgur|farro|spelt|tortillas?|pita|crackers?|cookies?|cake|pastry|"
    r"phyllo|filo|warqa|brioche|croutons?|beer|soy sauce|seitan|vermicelli|biscuits?)\b"
)
_GLUTEN_FREE = re.compile(
    r"\b(gluten[- ]free|rice|almond|coconut|corn|chickpea|buckwheat|oat|tapioca|potato)\s+"
    r"(flour|pasta|noodles?|bread|tortillas?|crackers?)\b"
)


def infer_diet(ingredients: list[str]) -> dict[str, bool]:
    vegetarian = vegan_ok = gluten_free = True
    for ing in ingredients:
        low = ing.lower()
        if not _PLANT_BASED.search(low) and _MEAT_FISH.search(_VEG_STOCK.sub("", low)):
            vegetarian = False
        if _ANIMAL.search(_ANIMAL_EXCEPTIONS.sub("", low)):
            vegan_ok = False
        if _GLUTEN.search(_GLUTEN_FREE.sub("", low)):
            gluten_free = False
    return {
        "is_vegetarian": vegetarian,
        "is_vegan": vegetarian and vegan_ok,
        "is_gluten_free": gluten_free,
    }


# --------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------
def _literal_list(raw: object) -> list[str]:
    try:
        return [clean_text(x) for x in ast.literal_eval(raw) if clean_text(x)]
    except (ValueError, SyntaxError):
        return []


def load_foodcom(path: Path) -> pd.DataFrame:
    log.info("Loading Food.com recipes from %s", path)
    df = pd.read_csv(path)
    df = df.dropna(subset=["name"])
    out = pd.DataFrame({
        "recipe_id": "fc-" + df["id"].astype(str),
        "source": "food.com",
        "name": df["name"].map(display_name),
        "description": df["description"].map(clean_text),
        "ingredients": df["ingredients"].map(_literal_list),
        "steps": df["steps"].map(lambda s: [_STEP_NUM.sub("", x) for x in _literal_list(s)]),
        "tags": df["tags"].map(_literal_list),
        "minutes": pd.to_numeric(df["minutes"], errors="coerce"),
    })
    nutrition = np.array(
        [_parse_nutrition(n) for n in df["nutrition"]], dtype="float32"
    )
    for i, col in enumerate(NUTRITION_FIELDS):
        out[col] = nutrition[:, i]
    return out.reset_index(drop=True)


def _parse_nutrition(raw: object) -> list[float]:
    try:
        vals = [float(v) for v in ast.literal_eval(raw)]
        if len(vals) == len(NUTRITION_FIELDS):
            return vals
    except (ValueError, SyntaxError, TypeError):
        pass
    return [np.nan] * len(NUTRITION_FIELDS)


_MOROCCAN_KEYS = ["Recipe Name", "Description", "Ingredients",
                  "Total time for preparation", "Steps for preparation"]
_SECTION_RE = re.compile(r"(?:^|\n)\s*(" + "|".join(_MOROCCAN_KEYS) + r")\s*:\s*")


def parse_moroccan_prompt(prompt: object, completion: object = None) -> dict | None:
    if not isinstance(prompt, str) or "Recipe Name" not in prompt:
        return None  # e.g. "Top 10 foods to try in Morocco : Harira - Tagine - ..."
    parts = _SECTION_RE.split(prompt)
    sections = {parts[i]: parts[i + 1].strip() for i in range(1, len(parts) - 1, 2)}
    steps = parse_list_field(sections.get("Steps for preparation"))
    if not steps:
        steps = parse_list_field(completion)
    minutes = pd.to_numeric(
        re.sub(r"[^\d.]", "", sections.get("Total time for preparation", "")) or np.nan,
        errors="coerce",
    )
    return {
        "name": display_name(sections.get("Recipe Name", "")),
        "description": clean_text(sections.get("Description")),
        "ingredients": parse_list_field(sections.get("Ingredients")),
        "steps": steps,
        "tags": ["moroccan", "north-african"],
        "minutes": minutes,
    }


def load_moroccan(path: Path) -> pd.DataFrame:
    log.info("Loading Moroccan recipes from %s", path)
    df = pd.read_csv(path)
    rows = []
    for i, (prompt, completion) in enumerate(zip(df["prompt"], df["completion"])):
        parsed = parse_moroccan_prompt(prompt, completion)
        if parsed:
            parsed["recipe_id"] = f"ma-{i}"
            rows.append(parsed)
    out = pd.DataFrame(rows)
    out["source"] = "moroccan"
    for col in NUTRITION_FIELDS:
        out[col] = np.nan
    return out


# --------------------------------------------------------------------------
# Cleaning + dedup
# --------------------------------------------------------------------------
def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def dedup_key(name: str, ingredients: list[str]) -> str:
    ing = "|".join(sorted({_norm(i) for i in ingredients}))
    return hashlib.md5(f"{_norm(name)}#{ing}".encode()).hexdigest()


def finalize(df: pd.DataFrame) -> pd.DataFrame:
    n0 = len(df)
    df = df[(df["name"].str.len() > 0)
            & (df["ingredients"].map(len) > 0)
            & (df["steps"].map(len) > 0)].copy()
    log.info("Dropped %d recipes without name/ingredients/steps", n0 - len(df))

    df.loc[(df["minutes"] <= 0) | (df["minutes"] > MAX_MINUTES), "minutes"] = np.nan
    df.loc[df["calories"] > MAX_CALORIES, NUTRITION_FIELDS] = np.nan

    # Exact duplicates (same name + same ingredient set). The Moroccan file in
    # particular contains the same scraped page several times. Moroccan
    # entries are kept first so they win ties.
    n1 = len(df)
    df["_key"] = [dedup_key(n, i) for n, i in zip(df["name"], df["ingredients"])]
    df = df.sort_values("source", key=lambda s: s.ne("moroccan")).drop_duplicates("_key")
    log.info("Removed %d exact duplicates", n1 - len(df))

    tags_low = df["tags"].map(lambda t: {x.lower() for x in t})
    df["is_moroccan"] = df["source"].eq("moroccan") | tags_low.map(lambda t: "moroccan" in t)
    diet = pd.DataFrame([infer_diet(i) for i in df["ingredients"]], index=df.index)
    df = pd.concat([df, diet], axis=1)
    # Food.com authors explicitly tagged some recipes gluten-free; trust that too.
    df["is_gluten_free"] |= tags_low.map(lambda t: "gluten-free" in t)
    df["n_ingredients"] = df["ingredients"].map(len)
    df["n_steps"] = df["steps"].map(len)

    cols = ["recipe_id", "source", "name", "description", "ingredients", "steps",
            "tags", "minutes", *NUTRITION_FIELDS, "is_moroccan", "is_vegetarian",
            "is_vegan", "is_gluten_free", "n_ingredients", "n_steps"]
    return df[cols].sort_values("recipe_id").reset_index(drop=True)


def build_recipe_table(out_path: Path | None = None) -> pd.DataFrame:
    paths = settings.paths
    df = pd.concat([load_foodcom(paths.foodcom_csv), load_moroccan(paths.moroccan_csv)],
                   ignore_index=True)
    df = finalize(df)
    out_path = out_path or paths.recipes_parquet
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    log.info("Saved %d recipes (%d Moroccan) to %s", len(df), df["is_moroccan"].sum(), out_path)
    return df


def load_recipes(path: Path | None = None) -> pd.DataFrame:
    return pd.read_parquet(path or settings.paths.recipes_parquet)
