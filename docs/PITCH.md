# Pitch and interview preparation

## 30-second pitch (English)

> I built a recipe assistant that answers in French, English or Darija using a retrieval-
> augmented LLM over 235,000 recipes, with a focus on Moroccan cuisine. My first version
> hallucinated on regional dishes like *tanjia*, so I rebuilt it the way you would in
> industry: I wrote an 81-query benchmark, found that the dish was simply spelled *tangia* in
> the data and that half the Moroccan recipes had been lost by a parsing bug, and replaced a
> hard-coded workaround with hybrid search (BM25 + multilingual embeddings), a spelling-variant
> lexicon mined from the corpus, LLM query understanding and cross-encoder reranking.
> Retrieval quality went from 0.42 to 0.83 nDCG@10, from 0.09 to 0.77 on
> Darija queries, and the assistant now says "I don't have this recipe" instead of inventing one.

## 30-second pitch (Français)

> J'ai construit un assistant culinaire qui répond en français, en anglais ou en darija grâce
> à un LLM augmenté par la recherche (RAG) sur 235 000 recettes, avec un accent sur la cuisine
> marocaine. Ma première version hallucinait sur des plats comme la *tanjia* ; je l'ai
> reconstruite comme on le ferait en entreprise : un benchmark de 81 requêtes, qui a montré
> que le plat était écrit *tangia* dans les données et qu'un bug de parsing avait fait perdre
> la moitié des recettes marocaines. J'ai remplacé une solution codée en dur par une recherche
> hybride (BM25 + embeddings multilingues), un lexique de variantes orthographiques appris sur
> le corpus, une compréhension de requête par LLM et un reranking par cross-encoder.
> Le nDCG@10 est passé de 0.42 à 0.83 (0.09 → 0.77 sur les requêtes en
> darija), et l'assistant dit maintenant « je n'ai pas cette recette » au lieu d'en inventer une.

## 2-minute version (structure to tell it as a story)

1. **Problem (15 s).** General LLMs invent regional recipes. I wanted an assistant grounded
   in a recipe database, usable in French and Darija, honest when it doesn't know.
2. **v1 and its failure (20 s).** First-year version: one embedding per recipe, top-2
   results, an LLM. It worked for lasagna, failed for *tanjia*: it retrieved American lamb
   stews and the prompt told the model to fill gaps with general knowledge, so it invented.
3. **Measure first (25 s).** Before changing anything I built a benchmark: 81 queries in
   7 categories, relevance defined by rules over recipe metadata so that no system is
   favoured, and standard IR metrics. Error analysis on it found the real causes: the dish is
   spelled *tangia* in the data, the Moroccan parser had silently dropped half the recipes,
   the embedding model was English-only, a third of the documents were truncated, and the
   hard-coded expansion actually made results slightly worse.
4. **Fix with the right tools (40 s).** Hybrid retrieval: BM25 for rare dish names plus
   multilingual embeddings for meaning, fused with Reciprocal Rank Fusion. A lexicon mined
   from the corpus that maps transliterations to the spelling the data uses
   (tanjia → tangia, bstilla → bastilla), with guards I added each time the benchmark showed
   a regression (it was turning *poisson* into *poison*). An LLM that rewrites the question
   into English with filters (vegetarian, under 30 minutes) using structured outputs, and a
   cross-encoder reranker. Two results surprised me and I kept them: the LLM first made
   Darija queries *worse* (it guessed that *harsha*, a flatbread, was *harira*, a soup) until
   I grounded it with the spellings the lexicon found in the data; and my structure-aware
   chunking barely beat fixed-size chunking once the model was multilingual, so I chose the
   cheapest variant and dropped the method chunks that hurt on the full corpus.
5. **Results (20 s).** nDCG@10 0.42 → 0.83; Darija 0.09 → 0.77;
   Moroccan-in-French 0.15 → 0.80. On answers, judged by a different LLM family,
   faithfulness 2.1 → 4.1-4.5 and correct "I don't know" on unanswerable questions
   0% → 100%.
6. **What I learned (10 s).** Most "model" problems were data and evaluation problems; and
   every component had to earn its place in the ablation table.

## Resume bullets

* Rebuilt a French/Darija recipe RAG assistant over 235k recipes: hybrid BM25 + multilingual
  embedding retrieval with RRF, cross-encoder reranking and LLM query understanding;
  nDCG@10 0.42 → 0.83 on an 81-query benchmark I designed.
* Diagnosed a regional-dish hallucination down to spelling variants and a data-loss bug
  (50% of Moroccan recipes dropped); replaced a hard-coded fix with a corpus-mined
  transliteration lexicon (Darija queries 0.09 → 0.77 nDCG@10).
* Added grounded generation with citations, confidence-gated abstention and an LLM-as-judge
  evaluation (faithfulness 2.1 → 4.1-4.5/5); Flask app with 38 unit tests and CI.

## Likely interview questions

**Why RAG and not fine-tuning?**
The answers must be factual, citable and updatable: adding a recipe should not need
retraining. Fine-tuning teaches style and format; it doesn't make a model reliably recall
235k recipes, and it would make the hallucination problem harder to see. RAG also lets me
show the sources.

**Why hybrid search? Isn't a good embedding model enough?**
The benchmark says no. Dense models smooth over words they never saw in training: v1
matched *rafissa* to "rasamarula" because they look alike in sub-words. BM25 matches the
exact token and gives a rare word a high weight. Dense handles meaning and translation. They
fail on different queries, so fusing them wins (see the ablation table).

**Why Reciprocal Rank Fusion rather than a weighted sum?**
BM25 scores are unbounded and cosine similarities are not; a weighted sum needs calibration
and a weight tuned per query type, re-tuned whenever a model changes. RRF only uses ranks.
I kept k=60 from the original paper to avoid tuning on my own test set.

**How did you build the test set without biasing it toward your system?**
Relevance is defined by rules over metadata (title regex, ingredients, Moroccan flag, diet,
time), resolved over the whole corpus, not by labelling what a system returned. The same
judge scores v1 and v2. Caveats I'd mention: 81 queries is small, and I wrote the rules
myself, so the next step is an independent annotator and real user logs.

**How does the "tanjia" fix work, and why is it better than the dictionary?**
The lexicon is mined from recipe titles. A query word is mapped to a corpus spelling that
has the same transliteration key (g/j, p/b, ch/sh, ou/w are merged), the same consonant
skeleton (Arabic doesn't write short vowels) or a very high character similarity, and only
towards a spelling the corpus actually uses. Nothing is specific to a dish, so it covers
dishes nobody wrote a rule for, and it improves when data is added. The dictionary also
injected generic words like "stew" and measurably hurt.

**Couldn't the LLM do that alone?**
I tested it: neither gpt-oss-20b nor gpt-oss-120b knew that *tanjia* is written *tangia*
in this data. The LLM translates and knows dishes in general; the lexicon knows the corpus.
They are complementary, and the lexicon still works if the LLM call fails.

**How do you control hallucination?**
Retrieval quality first (most hallucinations were retrieval failures). Then a prompt where
every recipe fact must be cited, general tips must be labelled, and the model must say when
the dish isn't in the sources; a confidence gate on the reranker score; temperature 0.3.
I measure it with an LLM judge from another model family, validated with planted errors.

**Tell me about something that didn't work.**
Two things. First, adding LLM query rewriting lowered quality on Darija queries from 0.70 to
0.60: for dish names it didn't know, the model guessed confidently (*harsha* → *harira*,
*bessara* → "lentil soup") and the reranker trusted its English rewrite over the user's
words. I fixed it by giving the LLM evidence (the lexicon's corpus spellings in the prompt,
plus "never guess an unknown dish") and by scoring candidates against both the user's query
and the rewrite. Darija went to 0.77. Second, I expected structure-aware chunking to be a big
win; the experiment said the multilingual model was the big win and chunking strategies were
within noise, and method chunks even hurt on the full corpus. I reported it and kept the
simplest design.

**Did you try a bigger embedding model?**
Yes, e5-base on a free Colab GPU (the laptop CPU would have needed 9 hours). Alone it is
better (0.680 vs 0.667), but inside the full pipeline it gives nothing (0.822 vs 0.828),
because BM25, the reranker and the LLM already cover what it adds. I kept e5-small: 4×
cheaper to index.

**What was the hardest bug?**
The lexicon "fixing" French words: *poisson* became *poison*, *facile* became *facial*. Fuzzy
matching is language-blind. I added a word-frequency guard (an everyday French or English
word is never fuzzy-corrected) and found the bug only because I compared per-query results
between two runs, not just the average.

**Why not LangChain?**
For a project whose point is understanding and measuring each stage, plain modules are
clearer: no hidden prompts, every component is unit-tested and can be switched off for the
ablation. I'd happily use a framework in a team that standardised on one.

**Latency? How would you scale it?**
Retrieval is ~1.4 s on a laptop CPU (reranking dominates; I cut it by 36% by reranking 20
candidates at 256 tokens with no measurable quality loss), plus ~1 s for the LLM parse. The
full answer takes 12-15 s because a reasoning model writes the recipe: streaming would fix
the perceived latency.
To scale: approximate vector index (HNSW / IVF-PQ) above a few million vectors, GPU or
quantised reranker, caching frequent queries, streaming the answer.

**What would you do with more time or compute?**
A bigger encoder (bge-m3 / e5-large) on GPU; fine-tune the encoder on synthetic
French/Darija queries generated per recipe; a larger benchmark labelled independently; and
the meal-planning feature as a tool on top of the retriever.
