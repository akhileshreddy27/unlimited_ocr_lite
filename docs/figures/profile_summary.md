Prefix L_m = 878, 7000 decoded tokens, 3 repeat(s), 14 threads

| Setting | Cache | Decode tok/s | First-100 ms/tok | Last-100 ms/tok | Peak KV MB |
|---|---|---|---|---|---|
| full | dynamic | 11.84 ± 1.2 | 48.06 ± 11.86 | 129.77 ± 1.87 | 346.2 |
| 512 | trim | 16.14 ± 0.19 | 54.93 ± 4.23 | 66.49 ± 0.9 | 61.08 |
| ring512 | ring | 28.09 ± 2.21 | 34.51 ± 2.28 | 33.82 ± 0.48 | 61.13 |
| ring128 | ring | 30.56 ± 0.06 | 32.79 ± 0.42 | 32.31 ± 0.43 | 44.25 |
