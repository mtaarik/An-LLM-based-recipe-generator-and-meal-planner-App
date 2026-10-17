### Overall (74 queries)

| System | nDCG@10 | MRR@10 | Hit@1 | Hit@5 | Recall@10 | p50 latency |
|---|---|---|---|---|---|---|
| v1: MiniLM, whole recipe, hard-coded expansion | 0.416 | 0.543 | 0.486 | 0.622 | 0.390 | 111 ms |
| v1 without the hard-coded expansion | 0.426 | 0.555 | 0.500 | 0.622 | 0.395 | 107 ms |
| BM25 only | 0.583 | 0.719 | 0.649 | 0.838 | 0.569 | 19 ms |
| BM25 + variant lexicon | 0.642 | 0.737 | 0.676 | 0.851 | 0.649 | 35 ms |

### nDCG@10 by query category

| System | moroccan_fr | darija | intl_en | intl_fr | ingredients | constraints |
|---|---|---|---|---|---|---|
| v1: MiniLM, whole recipe, hard-coded expansion | 0.146 | 0.090 | 0.892 | 0.776 | 0.410 | 0.451 |
| v1 without the hard-coded expansion | 0.146 | 0.072 | 0.932 | 0.794 | 0.392 | 0.518 |
| BM25 only | 0.614 | 0.423 | 0.872 | 0.639 | 0.433 | 0.518 |
| BM25 + variant lexicon | 0.594 | 0.805 | 0.872 | 0.582 | 0.434 | 0.481 |
