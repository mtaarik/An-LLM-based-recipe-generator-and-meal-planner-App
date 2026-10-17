### Overall (74 queries)

| System | nDCG@10 | MRR@10 | Hit@1 | Hit@5 | Recall@10 | p50 latency |
|---|---|---|---|---|---|---|
| BM25 only | 0.583 | 0.719 | 0.649 | 0.838 | 0.569 | 19 ms |
| BM25 + variant lexicon | 0.667 | 0.774 | 0.716 | 0.878 | 0.661 | 30 ms |

### nDCG@10 by query category

| System | moroccan_fr | darija | intl_en | intl_fr | ingredients | constraints |
|---|---|---|---|---|---|---|
| BM25 only | 0.614 | 0.423 | 0.872 | 0.639 | 0.433 | 0.518 |
| BM25 + variant lexicon | 0.642 | 0.805 | 0.872 | 0.639 | 0.433 | 0.522 |
