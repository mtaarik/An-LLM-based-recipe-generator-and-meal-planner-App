"""Answer-quality evaluation with an LLM judge (needs GROQ_API_KEY).

    python -m eval.evaluate_generation                 # v1 vs v2
    python -m eval.evaluate_generation --per-category 4

For each question the judge sees the question, the recipes the system
actually retrieved (its only legitimate knowledge source) and the answer, and
returns:
  * faithfulness (1-5) and the number of recipe facts NOT supported by the
    sources -> measures hallucination;
  * relevance (1-5) -> does it answer what was asked;
  * declines (bool) -> does it say the recipe isn't available / the request
    is out of scope.

Abstention is then scored on two sides: answerable questions should NOT be
declined (false refusals), unanswerable ones (dish not in the database,
off-topic) SHOULD be (otherwise the system made a recipe up).

Caveat: LLM judges are noisy and can be lenient; a stronger model than the
generator is used, at temperature 0, with a strict rubric. Numbers are for
comparing systems on the same questions, not absolute truths.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from datetime import datetime

from eval.judge import EVAL_DIR, load_queries
from recipe_rag.config import settings
from recipe_rag.llm import LLMClient, LLMError
from recipe_rag.prompts import format_recipe

JUDGE_PROMPT = """You are a strict evaluator of a recipe assistant.

You get the USER QUESTION, the SOURCES the assistant retrieved from its recipe database
(its ONLY allowed knowledge for recipe facts) and the ASSISTANT ANSWER.

Return ONLY a JSON object:
{{
 "n_unsupported": number of recipe facts in the answer (ingredients, quantities,
                  times, temperatures, steps) that are NOT supported by the SOURCES
                  (a clearly labelled general tip does not count),
 "unsupported_examples": [up to 5 of those facts, a few words each],
 "faithfulness": 1-5 (5 = every recipe fact is supported by the sources,
                      1 = the recipe is essentially made up),
 "relevance": 1-5 (5 = fully addresses what the user asked for, e.g. the requested dish,
                   1 = answers something else),
 "declines": true if the answer says it has no recipe for the request / cannot help
             with the request, else false
}}

USER QUESTION:
{question}

SOURCES:
{sources}

ASSISTANT ANSWER:
{answer}"""


def pick_queries(per_category: int, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    by_cat = defaultdict(list)
    for q in load_queries():
        by_cat[q["category"]].append(q)
    picked = []
    for cat, qs in by_cat.items():
        picked += qs if cat == "unanswerable" else rng.sample(qs, min(per_category, len(qs)))
    # The original failure case is always included.
    if not any(q["id"] == "dar01" for q in picked):
        picked += [q for q in by_cat["darija"] if q["id"] == "dar01"]
    return picked


def judge(llm: LLMClient, question: str, sources: list[str], answer: str) -> dict:
    prompt = JUDGE_PROMPT.format(question=question, sources="\n\n".join(sources) or "(none)",
                                 answer=answer)
    raw = llm.complete([{"role": "user", "content": prompt}], model=settings.llm.judge_model,
                       temperature=0.0, max_tokens=1500, json_mode=True)
    out = json.loads(raw)
    try:
        out["n_unsupported"] = int(out.get("n_unsupported") or 0)
    except (TypeError, ValueError):
        out["n_unsupported"] = len(out.get("unsupported_examples") or [])
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="+", default=["v1", "v2"], choices=["v1", "v2"])
    ap.add_argument("--per-category", type=int, default=4)
    ap.add_argument("--tag", default="")
    ap.add_argument("--report-only", nargs="+", metavar="JSON",
                    help="merge previous result files into one table, no LLM calls")
    args = ap.parse_args(argv)
    if args.report_only:
        rows = [r for f in args.report_only for r in json.loads(open(f, encoding="utf-8").read())]
        systems = list(dict.fromkeys(r["system"] for r in rows))
        print(report(rows, systems))
        return

    llm = LLMClient(settings.llm)
    if not llm.available:
        raise SystemExit("Set GROQ_API_KEY in .env first.")
    queries = pick_queries(args.per_category)
    rows = []

    for system in args.systems:
        if system == "v1":
            from recipe_rag.legacy import LegacyV1Retriever, v1_generate
            legacy = LegacyV1Retriever()
        else:
            from recipe_rag.assistant import RecipeAssistant
            assistant = RecipeAssistant.load()
            by_id = assistant.engine.recipes.set_index("recipe_id")
        for q in queries:
            t0 = time.perf_counter()
            try:
                if system == "v1":
                    answer, sources = v1_generate(llm, legacy, q["query"])
                    mode, qu_source = "rag", None
                else:
                    res = assistant.answer(q["query"])
                    answer, mode = res.answer, res.mode
                    qu_source = (res.analysis or {}).get("source")
                    sources = [format_recipe(s["n"], by_id.loc[s["recipe_id"]]) for s in res.sources]
                latency = time.perf_counter() - t0
                verdict = judge(llm, q["query"], sources, answer)
            except (LLMError, json.JSONDecodeError) as exc:
                print(f"  ! {system} {q['id']}: {exc}")
                continue
            answerable = q["category"] != "unanswerable"
            rows.append({"system": system, "qid": q["id"], "category": q["category"],
                         "query": q["query"], "answerable": answerable, "mode": mode,
                         "query_parser": qu_source,
                         "answer": answer, "latency_s": latency, **verdict})
            print(f"{system} {q['id']:6} faith={verdict.get('faithfulness')} "
                  f"rel={verdict.get('relevance')} declines={verdict.get('declines')} "
                  f"unsupported={verdict['n_unsupported']}", flush=True)

    out_dir = EVAL_DIR / "results"
    out_dir.mkdir(exist_ok=True)
    stamp = args.tag or datetime.now().strftime("%Y%m%d_%H%M")
    (out_dir / f"generation_{stamp}.json").write_text(
        json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    md = report(rows, args.systems)
    (out_dir / f"generation_{stamp}.md").write_text(md, encoding="utf-8")
    print("\n" + md)


def report(rows: list[dict], systems: list[str]) -> str:
    def mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else float("nan")

    lines = ["| System | Faithfulness (1-5) | Answers with 0 unsupported facts | Relevance (1-5) "
             "| False refusals (answerable) | Correct abstentions (unanswerable) | Latency |",
             "|---|---|---|---|---|---|---|"]
    for s in systems:
        rs = [r for r in rows if r["system"] == s]
        ans = [r for r in rs if r["answerable"]]
        una = [r for r in rs if not r["answerable"]]
        lines.append(
            f"| {s} | {mean([r.get('faithfulness') for r in ans]):.2f} "
            f"| {mean([r['n_unsupported'] == 0 for r in ans]):.0%} "
            f"| {mean([r.get('relevance') for r in ans]):.2f} "
            f"| {mean([bool(r.get('declines')) for r in ans]):.0%} "
            f"| {mean([bool(r.get('declines')) for r in una]):.0%} "
            f"| {mean([r['latency_s'] for r in rs]):.1f} s |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
