Sub-corpus of 20247 recipes, 66 queries, dense retrieval only.

| Strategy | vectors | nDCG@10 | MRR@10 | Hit@5 | moroccan_fr | darija | intl_en | intl_fr | ingredients |
|---|---|---|---|---|---|---|---|---|---|
| A_whole_minilm | 20247 | 0.555 | 0.705 | 0.788 | 0.400 | 0.309 | 0.971 | 0.848 | 0.395 |
| B_whole_e5 | 20247 | 0.727 | 0.850 | 0.879 | 0.744 | 0.553 | 0.965 | 0.830 | 0.573 |
| C_windows_e5 | 48577 | 0.715 | 0.826 | 0.955 | 0.696 | 0.618 | 0.978 | 0.770 | 0.543 |
| D_overview_e5 | 20247 | 0.717 | 0.872 | 0.970 | 0.708 | 0.619 | 0.919 | 0.809 | 0.546 |
| E_overview_method_e5 | 43602 | 0.731 | 0.838 | 0.939 | 0.724 | 0.642 | 0.932 | 0.817 | 0.556 |
