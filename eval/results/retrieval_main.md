### Overall (74 queries)

| System | nDCG@10 | MRR@10 | Hit@1 | Hit@5 | Recall@10 | p50 latency |
|---|---|---|---|---|---|---|
| v1: MiniLM, whole recipe, hard-coded expansion | 0.423 | 0.547 | 0.500 | 0.622 | 0.395 | 38 ms |
| v1 without the hard-coded expansion | 0.426 | 0.555 | 0.500 | 0.622 | 0.395 | 35 ms |
| BM25 only | 0.583 | 0.719 | 0.649 | 0.838 | 0.569 | 11 ms |
| BM25 + variant lexicon | 0.667 | 0.774 | 0.716 | 0.878 | 0.661 | 19 ms |
| e5 dense, overview chunks | 0.667 | 0.832 | 0.743 | 0.946 | 0.645 | 341 ms |
| e5 dense + lexicon | 0.662 | 0.833 | 0.757 | 0.932 | 0.636 | 404 ms |
| BM25 + dense (RRF) | 0.690 | 0.784 | 0.662 | 0.946 | 0.692 | 417 ms |
| hybrid + lexicon | 0.728 | 0.811 | 0.689 | 0.973 | 0.742 | 456 ms |
| + metadata filters (rule-based query parsing) | 0.755 | 0.821 | 0.703 | 0.973 | 0.772 | 462 ms |
| + cross-encoder reranking | 0.779 | 0.882 | 0.784 | 1.000 | 0.782 | 3534 ms |
| + LLM query understanding | 0.805 | 0.909 | 0.865 | 0.973 | 0.799 | 4199 ms |
| + method chunks | 0.792 | 0.899 | 0.865 | 0.946 | 0.788 | 5295 ms |

### nDCG@10 by query category

| System | moroccan_fr | darija | intl_en | intl_fr | ingredients | constraints |
|---|---|---|---|---|---|---|
| v1: MiniLM, whole recipe, hard-coded expansion | 0.139 | 0.095 | 0.919 | 0.785 | 0.394 | 0.484 |
| v1 without the hard-coded expansion | 0.146 | 0.072 | 0.932 | 0.794 | 0.392 | 0.518 |
| BM25 only | 0.614 | 0.423 | 0.872 | 0.639 | 0.433 | 0.518 |
| BM25 + variant lexicon | 0.642 | 0.805 | 0.872 | 0.639 | 0.433 | 0.522 |
| e5 dense, overview chunks | 0.637 | 0.478 | 0.873 | 0.788 | 0.587 | 0.724 |
| e5 dense + lexicon | 0.627 | 0.448 | 0.873 | 0.788 | 0.587 | 0.754 |
| BM25 + dense (RRF) | 0.687 | 0.580 | 0.897 | 0.773 | 0.535 | 0.684 |
| hybrid + lexicon | 0.689 | 0.756 | 0.897 | 0.773 | 0.535 | 0.702 |
| + metadata filters (rule-based query parsing) | 0.689 | 0.756 | 0.897 | 0.773 | 0.535 | 0.957 |
| + cross-encoder reranking | 0.783 | 0.702 | 0.903 | 0.814 | 0.617 | 0.893 |
| + LLM query understanding | 0.803 | 0.599 | 0.902 | 0.853 | 0.855 | 0.925 |
| + method chunks | 0.828 | 0.512 | 0.892 | 0.869 | 0.852 | 0.910 |
