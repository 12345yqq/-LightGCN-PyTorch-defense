# 仓库答辩化整理清单（可直接执行）

## 使用说明
- 目标：把“做过很多工作”转成“仓库内可见、可追溯、可复现”的证据链。
- 原则：不追求全量重跑，只保留关键证据最小集。
- 做法：按下面清单逐项打勾，优先完成“必须项”。

## A. 结构整理（必须）
- [ ] 在仓库保留统一说明入口（README 增加“答辩结果索引”小节）。
- [ ] 结果文档统一放在 `code/` 下，命名统一使用 `Model_Dataset_Setting_Summary` 风格。
- [ ] 训练脚本统一放在仓库根目录，文件名体现数据集、epoch、seed 规模。
- [ ] 对“历史跑过但未完整落盘”的实验，新增一份“历史实验记录”说明文件。

建议目录视图：
- `code/Comparison_LightGCN_vs_NGCF_Yelp2018_300_3Seeds.md`
- `code/Training_Cost_Table_Yelp2018.md`
- `code/Error_Analysis_OnePage_LightGCN_vs_NGCF_Yelp2018.md`
- `code/Defense_OnePage_Overview_Yelp2018.md`
- `code/NGCF_Yelp2018_300_3Seeds_Summary.md`
- `run_yelp2018_300_3seeds_ngcf.sh`

## B. 证据链文件（必须）
- [ ] 一张核心对比表（模型间最终指标对比，含均值和方差）。
- [ ] 一张训练代价表（总时长/每 epoch 时长，注明统计方法）。
- [ ] 一页误差分析（均值±方差、误差模式、可能原因、结论边界）。
- [ ] 一页答辩总览（可直接贴 PPT，含三段结论）。
- [ ] 三个 seed 的原始日志文件保留（至少最终测试与终轮信息可追溯）。

## C. 复现性（必须）
- [ ] 每个关键结果对应一个可运行脚本（脚本名与结果文件名能对应）。
- [ ] 脚本写明固定参数：dataset/epochs/seeds/topk/model/lr/decay/batch 等。
- [ ] 在结果文档中注明来源日志与提取方式（最后一次 [TEST] / 指定 epoch）。
- [ ] 说明“未重跑项目”的理由（时间/算力约束）与影响范围。

## D. 工作量可见化（高优先）
- [ ] 新增 `CHANGELOG_Defense.md`（按时间列出关键实现与实验动作）。
- [ ] 新增 `EXPERIMENT_INDEX.md`（实验编号、目的、脚本、结果、结论）。
- [ ] 每类工作独立提交一次 commit（实现/实验/汇总/文档）。
- [ ] 给当前可答辩状态打一个 tag（例如 `defense-v1`）。

推荐 commit 主题：
- `feat: integrate NGCF into training pipeline`
- `exp: run ngcf yelp2018 300epoch 3seeds`
- `analysis: add lgn vs ngcf comparison and cost table`
- `docs: add one-page defense overview and error analysis`

## E. 答辩问答预案（高优先）
- [ ] 准备“创新点是什么”30 秒版本。
- [ ] 准备“为什么是负结果也有价值”30 秒版本。
- [ ] 准备“公平性如何保证（同预算/同指标/多 seed）”30 秒版本。
- [ ] 准备“若继续工作下一步做什么”20 秒版本。

可直接口径（简版）：
- “我做了模型与训练策略层面的实质改动，并进行了同预算对照验证。”
- “在当前数据集与预算下未观察到正向提升，但得到稳定负结果与边界条件。”
- “该结论可复现、可追溯，且完成了 MF/LightGCN/NGCF 基线链路对比。”

## F. 最终自检（必须）
- [ ] 从仓库首页 3 次点击内能到达所有核心结果文件。
- [ ] 每个结论都能追溯到具体日志或 JSON。
- [ ] 所有图表数字与文档数字一致（随机抽查 3 处）。
- [ ] 删除无关临时文件，保留最小但完整证据集。

## G. 你当前状态（按现有产出）
- [x] 已有核心对比表。
- [x] 已有 NGCF 三种子汇总。
- [x] 已有训练代价表。
- [x] 已有一页误差分析。
- [x] 已有答辩一页总览。
- [ ] 待补：`CHANGELOG_Defense.md`
- [ ] 待补：`EXPERIMENT_INDEX.md`
- [ ] 待补：README 的“答辩结果索引”入口

---

执行建议（今天就能做完）：
1. 先补 README 入口与两个索引文档。
2. 再按“实现/实验/分析/文档”分 4 次 commit。
3. 最后打 `defense-v1` tag，冻结答辩版本。
