# 实验索引

本索引将每个关键实验对应到：脚本、配置、输出文件与结论，便于答辩追溯。

## E1. LightGCN vs NGCF（Yelp2018，300 epoch，3 seeds）
- 目标：在统一协议下建立核心基线对比。
- 脚本：
  - run_yelp2018_300_3seeds_ngcf.sh
  - run_yelp2018_300_3seeds_lgn.sh
- 协议：
  - 数据集：yelp2018
  - 训练轮数：300
  - 随机种子：2020, 2021, 2022
  - 评测 topk：[20]
  - 统一预算：recdim=64, layer=3, bpr_batch=4096, testbatch=2048, lr=1e-3, decay=1e-4
- 输出：
  - code/NGCF_Yelp2018_300_3Seeds_Summary.md
  - code/Comparison_LightGCN_vs_NGCF_Yelp2018_300_3Seeds.md
  - code/Comparison_LightGCN_vs_NGCF_Yelp2018_300_3Seeds.json
- 关键结论：
  - 在三个 seed 上，LightGCN 的 Recall@20 与 NDCG@20 均优于 NGCF。

## E2. HNM 快筛（Yelp2018，150 epoch，seed=2020）
- 目标：验证硬负样本策略是否能带来改进。
- 脚本：
  - run_hnm_quick_screen_3groups.sh
- 组别：
  - baseline（hnm_mode=none）
  - HNM-only（hnm_mode=mix, hnm_ratio=0.4）
  - HNM+curriculum（hnm_curr_start=0.2, hnm_curr_end=0.6）
- 日志：
  - code/exp_logs/hnm_qs_baseline_yelp2018_s2020.log
  - code/exp_logs/hnm_qs_mix_yelp2018_s2020.log
  - code/exp_logs/hnm_qs_curr_yelp2018_s2020.log
- 关键结论：
  - 在当前设置下，两个 HNM 变体均低于 baseline。

## E3. 代价与稳健性分析（无需重跑）
- 目标：补充答辩所需的训练代价与误差证据。
- 输入：
  - code/runs 与 code/exp_logs 的既有日志
  - 已生成的模型对比 JSON
- 输出：
  - code/Training_Cost_Table_Yelp2018.md
  - code/Training_Cost_Table_Yelp2018.json
  - code/Error_Analysis_OnePage_LightGCN_vs_NGCF_Yelp2018.md
- 关键结论：
  - 当前 3-seed 协议下模型排序稳定；代价采用日志代理统计。

## E4. 答辩一页包
- 目标：生成可直接放入答辩 PPT 的材料。
- 输出：
  - code/Defense_OnePage_Overview_Yelp2018.md
  - code/Defense_Repository_Checklist.md
- 关键结论：
  - 仓库已具备从实现到实验再到分析的完整证据链。

## 复现说明
- 优先使用固定种子与上述脚本进行复现。
- 训练代价表采用日志 Sample 计时代理，避免重复计算。
- 结论适用范围限定在当前数据集与超参数预算。
