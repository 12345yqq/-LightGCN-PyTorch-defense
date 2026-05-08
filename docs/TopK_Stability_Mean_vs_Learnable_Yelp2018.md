# Top-K Stability: LightGCN-mean vs LightGCN-learnable (Yelp2018)

Settings: 3 seeds (2020/2021/2022), checkpoints at 300 epochs, paired evaluation at K=[10,20,50].

## Mean ± Std Across Seeds

| Metric | K | mean model | learnable model | diff (mean - learnable) |
|---|---:|---:|---:|---:|
| Precision | 10 | 0.032204 +- 0.000122 | 0.024805 +- 0.000253 | 0.007399 |
| Precision | 20 | 0.027677 +- 0.000025 | 0.021793 +- 0.000277 | 0.005884 |
| Precision | 50 | 0.021751 +- 0.000075 | 0.017707 +- 0.000178 | 0.004044 |
| Recall | 10 | 0.036223 +- 0.000283 | 0.027688 +- 0.000451 | 0.008535 |
| Recall | 20 | 0.061774 +- 0.000185 | 0.048261 +- 0.000812 | 0.013513 |
| Recall | 50 | 0.119547 +- 0.000627 | 0.096989 +- 0.001252 | 0.022558 |
| NDCG | 10 | 0.041279 +- 0.000103 | 0.031276 +- 0.000570 | 0.010003 |
| NDCG | 20 | 0.050629 +- 0.000062 | 0.038938 +- 0.000729 | 0.011691 |
| NDCG | 50 | 0.072150 +- 0.000306 | 0.057071 +- 0.000860 | 0.015079 |

## Seed-wise Direction Check
- Precision: all (mean - learnable) > 0 across all K and all seeds.
- Recall: all (mean - learnable) > 0 across all K and all seeds.
- NDCG: all (mean - learnable) > 0 across all K and all seeds.
