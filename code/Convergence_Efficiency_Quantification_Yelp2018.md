# Convergence Efficiency Quantification (Yelp2018)

- Models: LightGCN-mean vs LightGCN-learnable(best)
- Seeds: 2020, 2021, 2022
- Metrics: Recall@20, NDCG@20
- Threshold: 95% of mean-model final value (per metric)
- AUC: normalized trapezoid area over step domain

## Summary Table

| Metric | Threshold | Model | Reach Step (mean±std) | Reach Epoch (mean±std) | Norm-AUC (mean±std) | Final Value (mean±std) |
|---|---:|---|---:|---:|---:|---:|
| Recall@20 | 0.058689 | mean | 130.0±0.0 | 140.0±0.0 | 0.056076±0.000064 | 0.061778±0.000117 |
| Recall@20 | 0.058689 | learnable(best) | N/A | N/A | 0.042987±0.000381 | 0.048241±0.000808 |
| NDCG@20 | 0.048129 | mean | 143.3±4.7 | 153.3±4.7 | 0.045535±0.000071 | 0.050662±0.000174 |
| NDCG@20 | 0.048129 | learnable(best) | N/A | N/A | 0.034823±0.000359 | 0.038831±0.000654 |

## Key Findings
- Recall@20: mean reaches threshold at step 130.0 (vs N/A for learnable), and has normalized AUC 0.056076 (vs 0.042987).
- NDCG@20: mean reaches threshold at step 143.3 (vs N/A for learnable), and has normalized AUC 0.045535 (vs 0.034823).
