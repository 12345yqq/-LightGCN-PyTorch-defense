# Statistical Upgrade: L2 Variant vs Mean / Learnable-best (Yelp2018)

- Setting: 3 paired seeds (2020/2021/2022), Top-K = [10,20,50]
- Compared models: mean, learnable-best, learnable+L2 (layer_w_l2=1e-3)
- Inference note: n=3 is small; p-values are conservative and reported as supportive evidence.

## Table 1. Core Comparison (3-seed means)

| Metric | K | mean | learnable-best | learnable+L2 | L2-best gain | Mean gap shrink |
|---|---:|---:|---:|---:|---:|---:|
| Precision | 10 | 0.032204 | 0.024805 | 0.024724 | -0.000081 | -1.10% |
| Precision | 20 | 0.027677 | 0.021793 | 0.021706 | -0.000086 | -1.47% |
| Precision | 50 | 0.021751 | 0.017707 | 0.017682 | -0.000025 | -0.62% |
| Recall | 10 | 0.036223 | 0.027688 | 0.027427 | -0.000261 | -3.06% |
| Recall | 20 | 0.061774 | 0.048261 | 0.047943 | -0.000318 | -2.36% |
| Recall | 50 | 0.119547 | 0.096989 | 0.096691 | -0.000298 | -1.32% |
| NDCG | 10 | 0.041279 | 0.031276 | 0.031120 | -0.000156 | -1.56% |
| NDCG | 20 | 0.050629 | 0.038938 | 0.038748 | -0.000189 | -1.62% |
| NDCG | 50 | 0.072150 | 0.057071 | 0.056898 | -0.000174 | -1.15% |

## Table 2. Paired Statistics at K=20

| Metric@20 | mean - L2 (mean, 95% CI, p) | L2 - learnable-best (mean, 95% CI, p) |
|---|---|---|
| Precision | 0.005971, [0.005744, 0.006151], p=0.25 | -0.000086, [-0.000467, 0.000546], p=1.00 |
| Recall | 0.013832, [0.013063, 0.014425], p=0.25 | -0.000318, [-0.001271, 0.001390], p=1.00 |
| NDCG | 0.011880, [0.011544, 0.012080], p=0.25 | -0.000189, [-0.000904, 0.001238], p=1.00 |

## Paper-ready Conclusion Paragraph

We further upgraded the statistical analysis by adding a paired comparison for the L2-regularized learnable variant (layer_w_l2=1e-3). Across three paired seeds on Yelp2018, learnable+L2 remained below the mean aggregation baseline at all evaluated K values. At K=20, the paired difference (mean - L2) was 0.005971 for Precision, 0.013832 for Recall, and 0.011880 for NDCG, with bootstrap 95% CIs [0.005744, 0.006151], [0.013063, 0.014425], and [0.011544, 0.012080], respectively. Compared with learnable-best, L2 showed slight negative deltas at K=20 (Precision -0.000086, Recall -0.000318, NDCG -0.000189), and the corresponding bootstrap intervals covered zero, indicating no reliable improvement from this L2 setting. In terms of gap-to-mean change at K=20, L2 shifted the original mean-vs-learnable-best gap by -1.47% (Precision), -2.36% (Recall), and -1.62% (NDCG), i.e., a small widening rather than a reduction. Given the small sample size (n=3), p-values are conservative and should be interpreted as supportive rather than definitive significance evidence; overall, this result strengthens the main conclusion that mean aggregation remains strongest under the current setting.
