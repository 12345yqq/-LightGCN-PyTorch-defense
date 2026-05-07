# REPRODUCE — 实验可复现说明（初稿）

说明：本文件记录当前工作区中用于运行与复现主要实验的脚本、日志与命令示例；作为初步归档，可提交给导师或附到论文发表材料中。

1) 重要脚本（位于 `code/`）
- `code/run_lastfm_300x3.sh` — LastFM baseline 与 SimGCL（各 3 个 seed）的批量运行脚本。
- `code/run_gowalla_combo1_300x3.sh` — Gowalla 上的 `gowalla_best` 组合（layer_agg=learnable + layer_w regs + SimGCL）3 seed 批量脚本。
- 其它可能相关脚本：
  - `code/run_learnable_l2_300_3seeds.sh`
  - `code/run_l2_resume_seed2021_seed2022.sh`
  - `code/run_tau_screen_150.sh`, `code/run_tau_screen_resume.sh`

2) 日志与结果位置
- 运行日志与自动汇总位于 `code/exp_logs/` 目录下，示例：
  - `code/exp_logs/paper_lastfm_300x3/summary.tsv`
  - `code/exp_logs/paper_gowalla_best_300x3/summary.tsv`
  - `code/exp_logs/paper_gowalla_combo1_300x3/summary.tsv`
  - 训练检查点存放在仓库根目录下的 `code/checkpoints/`（请确认对应子目录）。

3) 如何复现（示例命令）
- 直接使用脚本（在仓库根目录执行）
```bash
bash code/run_lastfm_300x3.sh
bash code/run_gowalla_combo1_300x3.sh
```

- 单次命令（与脚本等价，示例：）
```bash
python3 code/main.py --dataset lastfm --model lgn --epochs 300 --seed 2020 --cl_weight 0.0 --gpu_id 0 --comment lastfm_baseline_seed2020
python3 code/main.py --dataset lastfm --model lgn --epochs 300 --seed 2020 --cl_weight 0.01 --gpu_id 0 --comment lastfm_simgcl_seed2020

python3 code/main.py --dataset gowalla --model lgn --epochs 300 --seed 2020 --layer_agg learnable --layer_w_l2 1e-4 --layer_w_tau 0.5 --cl_weight 0.01 --cl_temp 0.2 --gpu_id 0 --comment gowalla_best_seed2020
```

4) 关于 summary.tsv 中数值重复的说明（已知问题）
- 注意：仓库中现有的 `summary.tsv`（例如 `paper_lastfm_300x3/summary.tsv`、`paper_gowalla_best_300x3/summary.tsv`）使用的简单 grep 提取方式会导致 precision/recall/ndcg 列被错误地填成相同数值（这是因为脚本用的是 `tail -n1 | grep -oE "[0-9]+\.[0-9]+" | tail -n1` 之类的取最后一个浮点的策略）。
- 推荐的稳定做法：直接从每个训练日志中解析训练脚本打印出的测试字典行（例如包含 `{'precision': array([...]), 'recall': array([...]), 'ndcg': array([...])}` 的行），或修改训练脚本以输出 JSON 格式的结果。

示例从日志中提取正确三项的命令（在 `code/exp_logs/...` 目录下执行）：
```bash
# 示例：从单个 log 中提取最后一次测试的 precision/recall/ndcg
grep -a "precision" ${LOGFILE} | tail -n1
# 或（解析 dict 行，提取三个数）
grep -a "ndcg" ${LOGFILE} | tail -n1 | sed -E "s/.*precision': \[?([0-9\.eE+-]+).*/\1/"
```

（我可以为你补一个小脚本 `tools/extract_results.py`，自动解析所有日志并生成修正后的 `summary_corrected.tsv`。）

5) 快速检查点与日志打包建议
- 打包当前代码与必要文件：
  - `git rev-parse --short HEAD`（记录 commit hash）或 `zip -r experiments_snapshot.zip .`。
  - 导出依赖：`pip freeze > requirements.current.txt`。

6) 下步建议（我可以代劳）
- 生成 `REPRODUCE.md`（已完成此初稿）。
- 添加 `tools/extract_results.py` 用于生成修正后的 summary（可选）。
- 导出当前依赖到 `requirements.current.txt`（可选）。
- 创建一个 zip 快照或 git tag 并归档到指定远程（需你的授权/凭据）。

---
最后更新：初稿由自动脚本生成，若你同意我可以继续：
- 1) 自动生成 `tools/extract_results.py` 並运行，输出 `code/exp_logs/*/summary_corrected.tsv`；
- 2) 导出 `requirements.current.txt` 并把 TODO 标记推进。
