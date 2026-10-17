### Overall (74 queries)

| System | nDCG@10 | MRR@10 | Hit@1 | Hit@5 | Recall@10 | p50 latency |
|---|---|---|---|---|---|---|
| + cross-encoder reranking | 0.779 | 0.882 | 0.784 | 1.000 | 0.782 | 3362 ms |
| + LLM query understanding | 0.742 | 0.857 | 0.770 | 0.973 | 0.740 | 2906 ms |

### nDCG@10 by query category

| System | moroccan_fr | darija | intl_en | intl_fr | ingredients | constraints |
|---|---|---|---|---|---|---|
| + cross-encoder reranking | 0.783 | 0.702 | 0.903 | 0.814 | 0.617 | 0.893 |
| + LLM query understanding | 0.694 | 0.544 | 0.914 | 0.797 | 0.722 | 0.931 |
