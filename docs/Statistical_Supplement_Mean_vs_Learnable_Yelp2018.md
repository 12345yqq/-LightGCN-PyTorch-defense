# Statistical Supplement: LightGCN-mean vs LightGCN-learnable (Yelp2018)

## Setup
- Dataset: Yelp2018
- Backbone: LightGCN (3 layers, embedding dim 64)
- Comparison: layer_agg=mean vs layer_agg=learnable (best tuned config)
- Seeds: 2020, 2021, 2022
- Checkpoints: 300-epoch models
- Metric level: @20

## Per-seed Results (@20)

| Seed | Model | Precision@20 | Recall@20 | NDCG@20 |
|---|---|---:|---:|---:|
| 2020 | mean | 0.027692 | 0.061566 | 0.050726 |
| 2020 | learnable | 0.021402 | 0.047112 | 0.037910 |
| 2021 | mean | 0.027698 | 0.062016 | 0.050670 |
| 2021 | learnable | 0.022014 | 0.048861 | 0.039480 |
| 2022 | mean | 0.027641 | 0.061741 | 0.050578 |
| 2022 | learnable | 0.021962 | 0.048809 | 0.039430 |

## Paired Difference Analysis (mean - learnable)

- Precision@20:
  - Mean paired difference: 0.005884
  - 95% bootstrap CI (paired, B=20000): [0.005679, 0.006290]
  - Seed-wise paired diffs: [0.006290, 0.005684, 0.005679]
- Recall@20:
  - Mean paired difference: 0.013513
  - 95% bootstrap CI (paired, B=20000): [0.012932, 0.014453]
  - Seed-wise paired diffs: [0.014453, 0.013154, 0.012932]
- NDCG@20:
  - Mean paired difference: 0.011718
  - 95% bootstrap CI (paired, B=20000): [0.011148, 0.012816]
  - Seed-wise paired diffs: [0.012816, 0.011190, 0.011148]

## Exact Sign Test (two-sided, n=3)
- All three metrics show 3/3 seed pairs in favor of mean.
- Exact two-sided sign-test p-value: 0.25.
- Interpretation: direction is fully consistent, but n=3 is small; report as supportive evidence rather than definitive frequentist significance.

## Suggested Paper Wording
Across three paired seeds on Yelp2018, LightGCN with mean layer aggregation consistently outperformed the learnable aggregation variant. Paired bootstrap confidence intervals of the difference (mean - learnable) remained strictly positive for Precision@20, Recall@20, and NDCG@20. This indicates a stable practical gap in favor of mean aggregation. An exact two-sided sign test on three paired seeds yielded p=0.25, which is conservative given the small sample size. We therefore report these results as consistent directional evidence rather than definitive significance.
