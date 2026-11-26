# Evaluation on OmniDocBench (English subset)

Overall* = mean(100·(1−TextED), TEDS, 100·(1−FormulaED)); FormulaED replaces CDM.

| Setting | Pages | Text ED↓ | TEDS↑ | TEDS-S↑ | Formula ED↓ | Order ED↓ | Layout F1↑ | Distinct-35↑ | Overall*↑ | Hit max tokens |
|---|---|---|---|---|---|---|---|---|---|---|
| full | 56 | 0.143 | 54.2 | 58.1 | 0.177 | 0.062 | 61.3 | 98.15 | 74.0 | 2 |
| 512 | 56 | 0.312 | 45.5 | 48.5 | 0.444 | 0.161 | 49.5 | 95.10 | 56.6 | 8 |
| 256 | 56 | 0.377 | 44.1 | 47.6 | 0.546 | 0.209 | 42.0 | 96.16 | 50.6 | 8 |
| 128 | 56 | 0.540 | 40.0 | 42.2 | 0.673 | 0.191 | 34.9 | 98.23 | 39.6 | 6 |

## Per document type (full)

| Type | Pages | Text ED↓ | TEDS↑ | Formula ED↓ | Order ED↓ |
|---|---|---|---|---|---|
| PPT2PDF | 8 | 0.135 | 58.7 | 0.354 | 0.000 |
| academic_literature | 8 | 0.070 | 59.6 | 0.275 | 0.028 |
| book | 8 | 0.034 | 94.2 | 0.170 | 0.000 |
| colorful_textbook | 8 | 0.274 | 53.4 | 0.211 | 0.036 |
| exam_paper | 8 | 0.110 | 33.2 | 0.019 | 0.000 |
| magazine | 8 | 0.157 | 46.2 | — | 0.111 |
| newspaper | 8 | 0.225 | 39.8 | — | 0.219 |

## Layout F1 @ IoU 0.5 per class (full)

| Class | F1 |
|---|---|
| figure | 63.3 |
| formula | 69.0 |
| table | 80.9 |
| text | 58.5 |
| title | 68.5 |
