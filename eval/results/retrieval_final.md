### Overall (74 queries)

| System | nDCG@10 | MRR@10 | Hit@1 | Hit@5 | Recall@10 | p50 latency |
|---|---|---|---|---|---|---|
| e5 dense, overview chunks | 0.667 | 0.832 | 0.743 | 0.946 | 0.645 | 99 ms |
| BM25 + dense (RRF) | 0.690 | 0.794 | 0.689 | 0.946 | 0.688 | 173 ms |
| hybrid + lexicon | 0.739 | 0.827 | 0.730 | 0.973 | 0.748 | 210 ms |
| + metadata filters (rule-based query parsing) | 0.766 | 0.837 | 0.743 | 0.973 | 0.776 | 215 ms |
| + cross-encoder reranking | 0.791 | 0.891 | 0.797 | 1.000 | 0.789 | 1430 ms |
| + LLM query understanding | 0.832 | 0.920 | 0.851 | 1.000 | 0.830 | 2491 ms |

### nDCG@10 by query category

| System | moroccan_fr | darija | intl_en | intl_fr | ingredients | constraints |
|---|---|---|---|---|---|---|
| e5 dense, overview chunks | 0.636 | 0.479 | 0.873 | 0.794 | 0.585 | 0.719 |
| BM25 + dense (RRF) | 0.683 | 0.590 | 0.897 | 0.770 | 0.529 | 0.685 |
| hybrid + lexicon | 0.697 | 0.814 | 0.897 | 0.770 | 0.529 | 0.695 |
| + metadata filters (rule-based query parsing) | 0.697 | 0.814 | 0.897 | 0.770 | 0.529 | 0.938 |
| + cross-encoder reranking | 0.775 | 0.750 | 0.913 | 0.841 | 0.595 | 0.903 |
| + LLM query understanding | 0.795 | 0.770 | 0.908 | 0.837 | 0.817 | 0.934 |
