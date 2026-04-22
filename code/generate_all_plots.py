import re
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

sns.set_theme(style="whitegrid")

data = []
for f in glob.glob("runs/*.log") + glob.glob("runs/*.txt"):
    try:
        with open(f, 'r') as file:
            content = file.read()
            recalls = [float(x) for x in re.findall(r"'recall': array\(\[([0-9\.]+)\]\)", content)]
            ndcgs = [float(x) for x in re.findall(r"'ndcg': array\(\[([0-9\.]+)\]\)", content)]
            
            if recalls:
                max_rec = max(recalls)
                max_ndcg = max(ndcgs)
                fname = os.path.basename(f).lower()
                model = "LightGCN" if "lgn" in fname or "layer" in fname else "MF"
                dataset = "Yelp2018" if "yelp" in fname else "Gowalla"
                
                seed_m = re.search(r"seed(\d+)", fname)
                seed = seed_m.group(1) if seed_m else "2020"
                
                layer_m = re.search(r"layer(\d+)", fname)
                layer = int(layer_m.group(1)) if layer_m else (3 if model == "LightGCN" else 0)
                
                recdim_m = re.search(r"recdim(\d+)", fname)
                recdim = int(recdim_m.group(1)) if recdim_m else 64
                
                exp_type = "Standard"
                if "sens" in fname: exp_type = "Sensitivity"
                if "ablation" in fname: 
                    exp_type = "Ablation-Comb" if "comb" in fname else "Ablation-Single"
                
                data.append({
                    "File": fname, "Model": model, "Dataset": dataset, "Seed": seed,
                    "Layer": layer, "RecDim": recdim, "Type": exp_type,
                    "Recall@20": max_rec, "NDCG@20": max_ndcg
                })
    except Exception as e:
        pass

df = pd.DataFrame(data)

print("---- 已提取的所有实验 ----")
print(df.groupby(['Dataset', 'Model', 'Type', 'Layer', 'RecDim']).size().reset_index(name='Seeds_Count'))

plt.rcParams['font.sans-serif']=['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus']=False

plt.figure(figsize=(10, 6))
standard_df = df[df['Type'] == 'Standard'].copy()
if not standard_df.empty:
    sns.barplot(data=standard_df, x="Dataset", y="Recall@20", hue="Model", capsize=.1, palette="Set2")
    plt.title("MF vs LightGCN", fontsize=16, fontweight='bold')
    plt.savefig("Fig1_MF_vs_LightGCN_Datasets.png", dpi=300)

layer_df = df[(df['Model'] == 'LightGCN') & (df['Dataset'] == 'Gowalla')].copy()
if not layer_df.empty:
    plt.figure(figsize=(8, 6))
    sns.lineplot(data=layer_df, x="Layer", y="Recall@20", marker='o', markersize=10, linewidth=3)
    plt.title("Layer Sensitivity (n_layers vs Recall)", fontsize=16, fontweight='bold')
    plt.xticks([1, 2, 3, 4])
    plt.savefig("Fig2_Layer_Sensitivity.png", dpi=300)

dim_df = df[(df['Layer'] == 3) & (df['Model'] == 'LightGCN')].copy()
if len(dim_df['RecDim'].unique()) > 1:
    plt.figure(figsize=(8, 6))
    sns.barplot(data=dim_df, x="RecDim", y="Recall@20", capsize=.1, palette="Blues")
    plt.title("RecDim Sensitivity", fontsize=16, fontweight='bold')
    plt.savefig("Fig3_RecDim_Sensitivity.png", dpi=300)

ablation_df = df[df['Type'].str.contains("Ablation") | ((df['Type']=='Standard') & (df['Model']=='LightGCN'))].copy()
ablation_df.loc[ablation_df['Type'] == 'Standard', 'Type'] = 'Standard LightGCN'
if len(ablation_df['Type'].unique()) > 1:
    plt.figure(figsize=(10, 6))
    sns.barplot(data=ablation_df, x="Type", y="NDCG@20", capsize=.1, palette="pastel")
    plt.title("Ablation Study (Comb vs Single)", fontsize=16, fontweight='bold')
    plt.savefig("Fig4_Ablation_Study.png", dpi=300)

print("\n4张补充图表已全部生成！")
