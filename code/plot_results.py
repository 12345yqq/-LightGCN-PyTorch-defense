import re
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_theme(style="whitegrid")

# 解析带 log 的文本文件提取最后测试结果
def parse_log(filename):
    epochs, recalls, ndcgs, precisions = [], [], [], []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if "{'precision'" in line and "'recall'" in line:
                # 提取数字
                try:
                    prec = float(re.search(r"'precision': array\(\[([0-9\.]+)\]\)", line).group(1))
                    rec = float(re.search(r"'recall': array\(\[([0-9\.]+)\]\)", line).group(1))
                    ndcg = float(re.search(r"'ndcg': array\(\[([0-9\.]+)\]\)", line).group(1))
                    recalls.append(rec)
                    ndcgs.append(ndcg)
                    precisions.append(prec)
                except:
                    pass
    if recalls:
        return {'recall': recalls, 'ndcg': ndcgs, 'precision': precisions}
    return None

print("正在提取日志信息，生成精美对比图...")

# 我们先画 MF 和 LightGCN 在 Gowalla 的收敛曲线对比
mf_log = "runs/mf_gowalla_seed2020_1000_0328_093723.log"
lgn_log = "runs/lgn_gowalla_layer3_seed2020_1000_0329_094428.log"

mf_metrics = parse_log(mf_log)
lgn_metrics = parse_log(lgn_log)

if mf_metrics and lgn_metrics:
    epochs_mf = np.arange(0, len(mf_metrics['recall']) * 10, 10)
    epochs_lgn = np.arange(0, len(lgn_metrics['recall']) * 10, 10)

    # 画 Recall 曲线
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_mf, mf_metrics['recall'], label="MF (Baseline)", linewidth=2)
    plt.plot(epochs_lgn, lgn_metrics['recall'], label="LightGCN", linewidth=2)
    plt.title("Gowalla Dataset - Recall@20 over Epochs", fontsize=16, fontweight='bold')
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Recall@20", fontsize=14)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig("Recall20_Convergence_Gowalla.png", dpi=300)
    print("生成了曲线图: Recall20_Convergence_Gowalla.png")

    # 画 NDCG 曲线
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_mf, mf_metrics['ndcg'], label="MF (Baseline)", linewidth=2, linestyle='--')
    plt.plot(epochs_lgn, lgn_metrics['ndcg'], label="LightGCN", linewidth=2, linestyle='-')
    plt.title("Gowalla Dataset - NDCG@20 over Epochs", fontsize=16, fontweight='bold')
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("NDCG@20", fontsize=14)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig("NDCG20_Convergence_Gowalla.png", dpi=300)
    print("生成了曲线图: NDCG20_Convergence_Gowalla.png")

# 生成多种子（Seed）对比柱状图 + 误差棒
seeds = ["2020", "2021", "2022"]
mf_recalls, lgn_recalls = [], []
for s in seeds:
    # Yelp2018 logs
    log_mf = glob.glob(f"runs/*mf_yelp*seed{s}*.log")
    log_lgn = glob.glob(f"runs/*lgn_yelp*seed{s}*.log")
    if log_mf and log_lgn:
        r_mf = parse_log(log_mf[0])
        r_lgn = parse_log(log_lgn[0])
        if r_mf and r_lgn:
            mf_recalls.append(max(r_mf['recall']))
            lgn_recalls.append(max(r_lgn['recall']))

if len(mf_recalls) >= 2:
    mean_mf = np.mean(mf_recalls)
    std_mf = np.std(mf_recalls)
    mean_lgn = np.mean(lgn_recalls)
    std_lgn = np.std(lgn_recalls)

    plt.figure(figsize=(8, 6))
    bars = plt.bar(['MF', 'LightGCN'], [mean_mf, mean_lgn], yerr=[std_mf, std_lgn], 
                   capsize=10, color=['#1f77b4', '#ff7f0e'], alpha=0.8, edgecolor='black', linewidth=1.5)
    plt.title("Yelp2018 Dataset - Best Recall@20 (3 Random Seeds)", fontsize=16, fontweight='bold')
    plt.ylabel("Recall@20", fontsize=14)
    
    # 添加具体数值标签
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval - (0.005 if yval > 0.05 else 0), 
                 round(yval, 4), ha='center', va='bottom', fontsize=12, fontweight='bold', color='white')

    plt.tight_layout()
    plt.savefig("Yelp2018_MultiSeed_BarChart.png", dpi=300)
    print("生成了带有误差棒的对比图: Yelp2018_MultiSeed_BarChart.png")

print("所有图表生成完毕！可以下载到本地放入PPT中。")
