### Overall (74 queries)

| System | nDCG@10 | MRR@10 | Hit@1 | Hit@5 | Recall@10 | p50 latency |
|---|---|---|---|---|---|---|
| + cross-encoder reranking | 0.792 | 0.887 | 0.797 | 1.000 | 0.793 | 2437 ms |
| + LLM query understanding | 0.822 | 0.921 | 0.851 | 1.000 | 0.817 | 3696 ms |

### nDCG@10 by query category

| System | moroccan_fr | darija | intl_en | intl_fr | ingredients | constraints |
|---|---|---|---|---|---|---|
| + cross-encoder reranking | 0.778 | 0.758 | 0.897 | 0.864 | 0.555 | 0.930 |
| + LLM query understanding | 0.809 | 0.717 | 0.891 | 0.831 | 0.826 | 0.932 |
