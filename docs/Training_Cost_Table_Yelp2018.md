# Training Cost Table (Yelp2018, 300 Epochs)

Method: calculated from existing logs using Sample timer accumulation (proxy).

| Model | Mean Total Time (proxy, hours) | Mean Total Time (proxy, sec) | Mean per-epoch (proxy, sec) | Std per-epoch (proxy, sec) |
|---|---:|---:|---:|---:|
| MF | 2.285 | 8225.20 | 27.4173 | 0.5301 |
| LightGCN | 2.246 | 8083.90 | 26.9463 | 0.3174 |
| NGCF | 2.229 | 8025.57 | 26.7519 | 0.3033 |

## Per-seed Details

| Model | Seed | Total Time (proxy, sec) | per-epoch (proxy, sec) |
|---|---:|---:|---:|
| MF | 2020 | 8077.36 | 26.9245 |
| MF | 2021 | 8445.88 | 28.1529 |
| MF | 2022 | 8152.37 | 27.1746 |
| LightGCN | 2020 | 8074.82 | 26.9161 |
| LightGCN | 2021 | 7972.09 | 26.5736 |
| LightGCN | 2022 | 8204.80 | 27.3493 |
| NGCF | 2020 | 8077.59 | 26.9253 |
| NGCF | 2021 | 7897.65 | 26.3255 |
| NGCF | 2022 | 8101.47 | 27.0049 |
