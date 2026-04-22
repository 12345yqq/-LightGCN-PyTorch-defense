# 答辩变更日志

本文件记录用于毕业答辩的关键实现改动与实验里程碑。

## 2026-04-22

### 文档与答辩材料打包
- 在 README 增加仓库级“答辩结果索引”。
- 新增答辩一页总览文档：
  - code/Defense_OnePage_Overview_Yelp2018.md
- 新增一页误差分析文档：
  - code/Error_Analysis_OnePage_LightGCN_vs_NGCF_Yelp2018.md
- 新增训练代价表文档：
  - code/Training_Cost_Table_Yelp2018.md
  - code/Training_Cost_Table_Yelp2018.json
- 新增 LightGCN vs NGCF 合并对比文档：
  - code/Comparison_LightGCN_vs_NGCF_Yelp2018_300_3Seeds.md
  - code/Comparison_LightGCN_vs_NGCF_Yelp2018_300_3Seeds.json
- 新增仓库答辩执行清单：
  - code/Defense_Repository_Checklist.md

### NGCF 接入与运行
- 将 NGCF 接入现有训练管线：
  - code/model.py
  - code/register.py
  - code/world.py
  - code/utils.py
  - code/parse.py
- 新增 NGCF 300 epoch 3-seed 批量脚本：
  - run_yelp2018_300_3seeds_ngcf.sh
- 完成 NGCF 在 Yelp2018 上的 300 epoch 三种子实验（2020/2021/2022）。
- 新增 NGCF 结果汇总文件：
  - code/NGCF_Yelp2018_300_3Seeds_Summary.md
  - code/NGCF_Yelp2018_300_3Seeds_Summary.json

## 2026-04-21

### HNM 实现与快筛实验
- 新增 HNM 参数并完成全局配置传递：
  - code/parse.py
  - code/world.py
- 在训练流程中实现 HNM 策略（none/mix/curriculum）：
  - code/Procedure.py
- 新增 HNM 三组快筛脚本：
  - run_hnm_quick_screen_3groups.sh
- 完成 Yelp2018 三组快筛：
  - baseline
  - HNM-only（mix=0.4）
  - HNM+curriculum（0.2 到 0.6）

### 既有实验结果整理
- 将 L2/tau/收敛/统计相关结果文件统一整理到 code/ 下。
- 保留 LightGCN 多种子对比结果，保证答辩阶段可追溯。

## 备注
- 部分训练代价统计采用日志代理（Sample 计时累计），用于避免不必要重跑。
- 结论适用范围限定在当前数据集、预算与超参数协议。
