
#### Update

2020-09:
* Change the print format of each epoch
* Add Cpp Extension in  `code/sources/`  for negative sampling. To use the extension, please install `pybind11` and `cppimport` under your environment

---

## LightGCN-pytorch

This is the Pytorch implementation for our SIGIR 2020 paper:

>SIGIR 2020. Xiangnan He, Kuan Deng ,Xiang Wang, Yan Li, Yongdong Zhang, Meng Wang(2020). LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation, [Paper in arXiv](https://arxiv.org/abs/2002.02126).

Author: Prof. Xiangnan He (staff.ustc.edu.cn/~hexn/)

(Also see Tensorflow [implementation](https://github.com/kuandeng/LightGCN))

## 答辩结果索引

本仓库包含一套用于毕业答辩的实验与分析材料，核心场景为 Yelp2018 下的统一协议多种子对比。

核心一页文档：
- [答辩一页总览](Defense_OnePage_Overview_Yelp2018.md)
- [LightGCN vs NGCF（300 epoch，3 seeds）](Comparison_LightGCN_vs_NGCF_Yelp2018_300_3Seeds.md)
- [训练代价表](Training_Cost_Table_Yelp2018.md)
- [一页误差分析](Error_Analysis_OnePage_LightGCN_vs_NGCF_Yelp2018.md)

结构化结果文件（json）：
- [LightGCN vs NGCF 汇总 json](../results/Comparison_LightGCN_vs_NGCF_Yelp2018_300_3Seeds.json)
- [NGCF 300 epoch 3-seed 汇总 json](../results/NGCF_Yelp2018_300_3Seeds_Summary.json)
- [训练代价表 json](../results/Training_Cost_Table_Yelp2018.json)

实验脚本：
- [NGCF yelp2018 300 epoch 3-seed 脚本](../scripts/run_yelp2018_300_3seeds_ngcf.sh)
- [LightGCN yelp2018 300 epoch 3-seed 脚本](../scripts/run_yelp2018_300_3seeds_lgn.sh)
- [HNM 快筛脚本](../scripts/run_hnm_quick_screen_3groups.sh)

仓库级说明文档：
- [仓库答辩执行清单](Defense_Repository_Checklist.md)
- [答辩变更日志](CHANGELOG_Defense.md)
- [实验索引](EXPERIMENT_INDEX.md)

## Introduction

In this work, we aim to simplify the design of GCN to make it more concise and appropriate for recommendation. We propose a new model named LightGCN,including only the most essential component in GCN—neighborhood aggregation—for collaborative filtering



## Enviroment Requirement

`pip install -r requirements.txt`

### 答辩一体化可视化页面

本仓库提供了一个一体化 Streamlit 页面，包含两部分：
- 左侧：实验结果看板（Recall@20、NDCG@20、模型对比、seed 波动、训练耗时、收敛曲线）
- 右侧：在线推荐演示（输入用户ID，基于训练好的 checkpoint 实时输出 Top-K 推荐）

运行方式：

```bash
pip install streamlit
streamlit run code/dashboard_app.py
```

说明：
- 页面默认读取 Yelp2018 相关结果文件与 checkpoints。
- 如果希望演示不同模型，可在页面右侧切换不同权重文件（LightGCN/NGCF/MF）。



## Dataset

We provide three processed datasets: Gowalla, Yelp2018 and Amazon-book and one small dataset LastFM.

see more in `dataloader.py`

## An example to run a 3-layer LightGCN

run LightGCN on **Gowalla** dataset:

* command

` cd code && python main.py --decay=1e-4 --lr=0.001 --layer=3 --seed=2020 --dataset="gowalla" --topks="[20]" --recdim=64`

* log output

```shell
...
======================
EPOCH[5/1000]
BPR[sample time][16.2=15.84+0.42]
[saved][[BPR[aver loss1.128e-01]]
[0;30;43m[TEST][0m
{'precision': array([0.03315359]), 'recall': array([0.10711388]), 'ndcg': array([0.08940792])}
[TOTAL TIME] 35.9975962638855
...
======================
EPOCH[116/1000]
BPR[sample time][16.9=16.60+0.45]
[saved][[BPR[aver loss2.056e-02]]
[TOTAL TIME] 30.99874997138977
...
```

*NOTE*:

1. Even though we offer the code to split user-item matrix for matrix multiplication, we strongly suggest you don't enable it since it will extremely slow down the training speed.
2. If you feel the test process is slow, try to increase the ` testbatch` and enable `multicore`(Windows system may encounter problems with `multicore` option enabled)
3. Use `tensorboard` option, it's good.
4. Since we fix the seed(`--seed=2020` ) of `numpy` and `torch` in the beginning, if you run the command as we do above, you should have the exact output log despite the running time (check your output of *epoch 5* and *epoch 116*).


## Extend:
* If you want to run lightGCN on your own dataset, you should go to `dataloader.py`, and implement a dataloader inherited from `BasicDataset`.  Then register it in `register.py`.
* If you want to run your own models on the datasets we offer, you should go to `model.py`, and implement a model inherited from `BasicModel`.  Then register it in `register.py`.
* If you want to run your own sampling methods on the datasets and models we offer, you should go to `Procedure.py`, and implement a function. Then modify the corresponding code in `main.py`


## Results
*all metrics is under top-20*

***pytorch* version results** (stop at 1000 epochs):

(*for seed=2020*)

* gowalla:

|             | Recall | ndcg | precision |
| ----------- | ---------------------------- | ----------------- | ---- |
| **layer=1** | 0.1687               | 0.1417    | 0.05106 |
| **layer=2** | 0.1786                     | 0.1524    | 0.05456 |
| **layer=3** | 0.1824                | 0.1547 | 0.05589 |
| **layer=4** | 0.1825                 | 0.1537       | 0.05576 |

* yelp2018

|             | Recall | ndcg | precision |
| ----------- | ---------------------------- | ----------------- | ---- |
| **layer=1** | 0.05604     | 0.04557 | 0.02519 |
| **layer=2** | 0.05988               | 0.04956 | 0.0271 |
| **layer=3** | 0.06347          | 0.05238 | 0.0285 |
| **layer=4** | 0.06515                | 0.05325 | 0.02917 |

