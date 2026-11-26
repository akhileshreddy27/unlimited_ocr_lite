# Multi-page / long-horizon parsing (synthetic documents from OmniDocBench pages)

All modes at longest_edge=1024. Model context = 8192 positions; 'Max pos' > 8192 means beyond the trained range.

| Doc | Pages | Mode | Text ED↓ | TEDS↑ | Formula ED↓ | Distinct-35↑ | Page breaks (expected) | Generated | Max pos | Peak KV MB | Wall s | Stop |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| book_2p | 2 | pagewise | 0.632 | 75.8 | 0.846 | 94.3 | 1 (1) | 3136 | 2848 | 125.16 | 139.0 | eos |
| book_2p | 2 | long_full | 0.671 | 63.2 | 1.000 | 90.6 | 1 (1) | 2866 | 3547 | 155.87 | 139.2 | eos |
| book_2p | 2 | long_ring512 | 0.528 | 47.2 | 1.000 | 97.5 | 0 (1) | 4202 | 4883 | 52.47 | 149.8 | eos |
| book_4p | 4 | pagewise | 0.473 | 75.8 | 0.857 | 97.4 | 3 (3) | 6814 | 2848 | 125.16 | 299.3 | eos |
| book_4p | 4 | long_full | 0.896 | 29.3 | 1.000 | 90.9 | 1 (3) | 10000 | 11347 | 498.65 | 830.1 | max_new_tokens |
| book_4p | 4 | long_ring512 | 0.622 | 32.5 | 1.000 | 100.0 | 2 (3) | 7689 | 9036 | 81.74 | 286.1 | eos |
| book_8p | 8 | pagewise | 0.564 | 75.8 | 0.825 | 97.5 | 7 (7) | 15580 | 2848 | 125.16 | 689.6 | eos |
| book_8p | 8 | long_full | 0.710 | 33.3 | 1.000 | 96.5 | 3 (7) | 5871 | 8550 | 375.73 | 453.7 | eos |
| book_8p | 8 | long_ring512 | 0.821 | 35.7 | 1.000 | 99.8 | 2 (7) | 3787 | 6466 | 140.27 | 172.5 | eos |
| academic_literature_2p | 2 | pagewise | 0.778 | 4.9 | 1.000 | 100.0 | 1 (1) | 4008 | 2848 | 125.16 | 175.9 | eos |
| academic_literature_2p | 2 | long_full | 1.000 | 0.3 | 1.000 | 78.2 | 0 (1) | 4816 | 5497 | 241.57 | 265.2 | eos |
| academic_literature_2p | 2 | long_ring512 | 0.749 | 8.4 | 1.000 | 98.7 | 1 (1) | 3381 | 4062 | 52.47 | 122.5 | eos |
| academic_literature_4p | 4 | pagewise | 0.766 | 4.9 | 0.688 | 99.7 | 3 (3) | 8138 | 2848 | 125.16 | 355.6 | eos |
| academic_literature_4p | 4 | long_full | 0.625 | 6.0 | 0.897 | 100.0 | 1 (3) | 2119 | 3466 | 152.31 | 114.2 | eos |
| academic_literature_4p | 4 | long_ring512 | 0.872 | 5.4 | 0.968 | 100.0 | 1 (3) | 2744 | 4091 | 81.74 | 109.8 | eos |
| academic_literature_8p | 8 | pagewise | 0.561 | 0.0 | 0.698 | 99.9 | 7 (7) | 15515 | 2848 | 125.16 | 674.0 | eos |
| academic_literature_8p | 8 | long_full | 0.804 | 1.8 | 1.000 | 100.0 | 1 (7) | 5788 | 8467 | 372.08 | 441.9 | eos |
| academic_literature_8p | 8 | long_ring512 | 1.000 | 0.7 | 1.000 | 100.0 | 3 (7) | 5947 | 8626 | 140.27 | 256.0 | eos |

## Mean over documents, by size

| Pages | Mode | Text ED↓ | Distinct-35↑ | Peak KV MB |
|---|---|---|---|---|
| 2 | pagewise | 0.705 | 97.2 | 125.16 |
| 2 | long_full | 0.835 | 84.4 | 241.57 |
| 2 | long_ring512 | 0.639 | 98.1 | 52.47 |
| 4 | pagewise | 0.619 | 98.6 | 125.16 |
| 4 | long_full | 0.760 | 95.4 | 498.65 |
| 4 | long_ring512 | 0.747 | 100.0 | 81.74 |
| 8 | pagewise | 0.563 | 98.7 | 125.16 |
| 8 | long_full | 0.757 | 98.3 | 375.73 |
| 8 | long_ring512 | 0.911 | 99.9 | 140.27 |
