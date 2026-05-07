import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import torch


ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / "code"
DATA_DIR = ROOT / "data" / "yelp2018"
CHECKPOINT_DIR = CODE_DIR / "checkpoints"

COMPARISON_JSON = CODE_DIR / "Comparison_LightGCN_vs_NGCF_Yelp2018_300_3Seeds.json"
COST_JSON = CODE_DIR / "Training_Cost_Table_Yelp2018.json"
TOPK_JSON = CODE_DIR / "TopK_Stability_Mean_vs_Learnable_Yelp2018.json"
NGCF_JSON = CODE_DIR / "NGCF_Yelp2018_300_3Seeds_Summary.json"


def inject_custom_style() -> None:
    st.markdown(
        """
        <style>
            .stApp {
                background:
                    radial-gradient(circle at 10% 5%, rgba(22, 163, 74, 0.10), transparent 28%),
                    radial-gradient(circle at 95% 8%, rgba(217, 119, 6, 0.11), transparent 30%),
                    linear-gradient(180deg, #f7faf7 0%, #f3f6fb 100%);
            }
            .main .block-container {
                max-width: 1500px;
                padding-top: 1.2rem;
                padding-bottom: 2rem;
            }
            .hero-wrap {
                border: 1px solid rgba(15, 23, 42, 0.09);
                border-radius: 16px;
                background: linear-gradient(130deg, rgba(15,118,110,0.08), rgba(180,83,9,0.10));
                padding: 18px 20px;
                margin-bottom: 14px;
                box-shadow: 0 12px 24px rgba(15, 23, 42, 0.07);
            }
            .hero-title {
                font-size: 1.52rem;
                font-weight: 750;
                color: #0f172a;
                margin-bottom: 0.20rem;
            }
            .hero-sub {
                color: #334155;
                font-size: 0.98rem;
                margin-bottom: 0;
            }
            .section-title {
                border-left: 4px solid #0f766e;
                padding-left: 10px;
                margin-top: 8px;
                margin-bottom: 10px;
                font-weight: 700;
                color: #0f172a;
            }
            .explain-card {
                border: 1px solid rgba(15, 23, 42, 0.12);
                border-radius: 14px;
                background: rgba(255, 255, 255, 0.88);
                padding: 12px 14px;
                margin: 8px 0 12px 0;
                box-shadow: 0 8px 20px rgba(15, 23, 42, 0.06);
            }
            .explain-title {
                font-size: 0.98rem;
                font-weight: 700;
                color: #0f172a;
                margin-bottom: 6px;
            }
            .explain-text {
                font-size: 0.92rem;
                color: #334155;
                line-height: 1.55;
                margin-bottom: 0;
            }
            [data-testid="stMetric"] {
                background: rgba(255, 255, 255, 0.85);
                border: 1px solid rgba(15, 23, 42, 0.10);
                border-radius: 12px;
                padding: 8px 10px;
            }
            [data-testid="stDataFrame"] {
                border: 1px solid rgba(15, 23, 42, 0.10);
                border-radius: 12px;
                overflow: hidden;
            }
            div[data-testid="stHorizontalBlock"] > div {
                gap: 0.8rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_explain_card(title: str, text: str) -> None:
    st.markdown(
        (
            '<div class="explain-card">'
            f'<div class="explain-title">{title}</div>'
            f'<p class="explain-text">{text}</p>'
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def load_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_id_map(path: Path) -> Tuple[Dict[str, int], Dict[int, str]]:
    org_to_remap: Dict[str, int] = {}
    remap_to_org: Dict[int, str] = {}
    if not path.exists():
        return org_to_remap, remap_to_org

    with path.open("r", encoding="utf-8") as f:
        next(f, None)  # skip header
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            org_id, remap_id_str = parts
            remap_id = int(remap_id_str)
            org_to_remap[org_id] = remap_id
            remap_to_org[remap_id] = org_id
    return org_to_remap, remap_to_org


def build_model_metrics_df(comparison_data: Dict, cost_data: Dict) -> pd.DataFrame:
    rows: List[Dict] = []
    agg = comparison_data.get("aggregate", {})

    rows.append(
        {
            "Model": "LightGCN",
            "Recall@20": agg.get("lgn_recall_mean", np.nan),
            "Recall_std": agg.get("lgn_recall_std", np.nan),
            "NDCG@20": agg.get("lgn_ndcg_mean", np.nan),
            "NDCG_std": agg.get("lgn_ndcg_std", np.nan),
        }
    )
    rows.append(
        {
            "Model": "NGCF",
            "Recall@20": agg.get("ngcf_recall_mean", np.nan),
            "Recall_std": agg.get("ngcf_recall_std", np.nan),
            "NDCG@20": agg.get("ngcf_ndcg_mean", np.nan),
            "NDCG_std": agg.get("ngcf_ndcg_std", np.nan),
        }
    )

    models_cost = cost_data.get("models", {})
    for row in rows:
        m = row["Model"]
        if m in models_cost:
            c_agg = models_cost[m].get("aggregate", {})
            row["Sec/Epoch"] = c_agg.get("sample_proxy_sec_per_epoch_mean", np.nan)
            row["Total Hours"] = c_agg.get("sample_proxy_total_hours_mean", np.nan)
        else:
            row["Sec/Epoch"] = np.nan
            row["Total Hours"] = np.nan

    if "MF" in models_cost:
        mf_agg = models_cost["MF"].get("aggregate", {})
        rows.append(
            {
                "Model": "MF",
                "Recall@20": np.nan,
                "Recall_std": np.nan,
                "NDCG@20": np.nan,
                "NDCG_std": np.nan,
                "Sec/Epoch": mf_agg.get("sample_proxy_sec_per_epoch_mean", np.nan),
                "Total Hours": mf_agg.get("sample_proxy_total_hours_mean", np.nan),
            }
        )

    return pd.DataFrame(rows)


def build_seed_variation_df(comparison_data: Dict) -> pd.DataFrame:
    rows = []
    for item in comparison_data.get("per_seed", []):
        seed = item.get("seed")
        rows.append({"seed": seed, "model": "LightGCN", "Recall@20": item.get("lgn_recall@20"), "NDCG@20": item.get("lgn_ndcg@20")})
        rows.append({"seed": seed, "model": "NGCF", "Recall@20": item.get("ngcf_recall@20"), "NDCG@20": item.get("ngcf_ndcg@20")})
    return pd.DataFrame(rows)


def build_topk_df(topk_data: Dict) -> pd.DataFrame:
    topks = topk_data.get("topks", [])
    agg = topk_data.get("aggregate", {}).get("mean", {})
    recall_mean = agg.get("recall", {}).get("mean", [])
    ndcg_mean = agg.get("ndcg", {}).get("mean", [])

    rows = []
    for idx, k in enumerate(topks):
        rows.append(
            {
                "K": k,
                "Recall@K": recall_mean[idx] if idx < len(recall_mean) else np.nan,
                "NDCG@K": ndcg_mean[idx] if idx < len(ndcg_mean) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def parse_curve_from_log(log_path: Path) -> pd.DataFrame:
    if not log_path.exists():
        return pd.DataFrame(columns=["epoch", "precision", "recall", "ndcg"])

    metric_pattern = re.compile(
        r"\{'precision': array\(\[([-+0-9.eE]+)\]\), 'recall': array\(\[([-+0-9.eE]+)\]\), 'ndcg': array\(\[([-+0-9.eE]+)\]\)\}"
    )
    epoch_pattern = re.compile(r"EPOCH\[(\d+)/(\d+)\]")

    rows = []
    pending_metric: Optional[Tuple[float, float, float]] = None

    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            metric_match = metric_pattern.search(line)
            if metric_match:
                pending_metric = (
                    float(metric_match.group(1)),
                    float(metric_match.group(2)),
                    float(metric_match.group(3)),
                )
                continue

            epoch_match = epoch_pattern.search(line)
            if epoch_match and pending_metric is not None:
                train_epoch = int(epoch_match.group(1))
                eval_epoch = max(train_epoch - 1, 0)
                rows.append(
                    {
                        "epoch": eval_epoch,
                        "precision": pending_metric[0],
                        "recall": pending_metric[1],
                        "ndcg": pending_metric[2],
                    }
                )
                pending_metric = None

    return pd.DataFrame(rows)


def discover_curve_logs() -> Dict[str, List[Tuple[int, Path]]]:
    groups = {"lgn": [], "ngcf": []}
    for log_path in (CODE_DIR / "exp_logs").glob("*.log"):
        name = log_path.name.lower()
        if "yelp2018" not in name:
            continue
        model_key = "lgn" if "lgn" in name else ("ngcf" if "ngcf" in name else None)
        if not model_key:
            continue
        seed_match = re.search(r"s(\d{4})", name)
        seed = int(seed_match.group(1)) if seed_match else 0
        groups[model_key].append((seed, log_path))

    for key in groups:
        groups[key] = sorted(groups[key], key=lambda x: x[0])
    return groups


def checkpoint_options() -> List[str]:
    files = sorted([p.name for p in CHECKPOINT_DIR.glob("*.pth.tar") if "yelp2018" in p.name])
    return files


def infer_model_from_ckpt(ckpt_name: str) -> str:
    name = ckpt_name.lower()
    if name.startswith("ngcf-"):
        return "ngcf"
    if name.startswith("mf-"):
        return "mf"
    return "lgn"


def infer_layer_agg_from_ckpt(ckpt_name: str) -> str:
    return "learnable" if "learnable" in ckpt_name.lower() or "_lrn_" in ckpt_name.lower() else "mean"


@st.cache_resource(show_spinner=False)
def load_legacy_modules():
    saved_argv = list(sys.argv)
    try:
        sys.argv = [saved_argv[0]]
        import world  # type: ignore
        import dataloader  # type: ignore
        import model  # type: ignore
    finally:
        sys.argv = saved_argv
    return world, dataloader, model


@st.cache_resource(show_spinner=True)
def load_inference_bundle(ckpt_name: str):
    world, dataloader, model = load_legacy_modules()

    model_key = infer_model_from_ckpt(ckpt_name)
    layer_agg = infer_layer_agg_from_ckpt(ckpt_name)

    world.dataset = "yelp2018"
    world.model_name = model_key
    world.config["latent_dim_rec"] = 64
    world.config["lightGCN_n_layers"] = 3
    world.config["layer_agg"] = layer_agg
    world.config["dropout"] = 0
    world.config["multicore"] = 0

    if torch.cuda.is_available():
        world.device = torch.device("cuda:0")
    else:
        world.device = torch.device("cpu")

    dataset = dataloader.Loader(config=world.config, path=str(DATA_DIR))

    if model_key == "lgn":
        rec_model = model.LightGCN(world.config, dataset)
    elif model_key == "ngcf":
        rec_model = model.NGCF(world.config, dataset)
    else:
        rec_model = model.PureMF(world.config, dataset)

    rec_model = rec_model.to(world.device)

    ckpt_path = CHECKPOINT_DIR / ckpt_name
    state = torch.load(str(ckpt_path), map_location=world.device)
    rec_model.load_state_dict(state, strict=True)
    rec_model.eval()

    return rec_model, dataset, str(world.device)


def recommend_topk(rec_model, dataset, user_id: int, topk: int) -> List[Tuple[int, float]]:
    device = next(rec_model.parameters()).device
    with torch.no_grad():
        users = torch.tensor([user_id], dtype=torch.long, device=device)
        scores = rec_model.getUsersRating(users).squeeze(0)

        seen = dataset.getUserPosItems([user_id])[0]
        if len(seen) > 0:
            seen_idx = torch.tensor(seen, dtype=torch.long, device=device)
            scores[seen_idx] = -1e9

        k = min(topk, scores.numel())
        values, indices = torch.topk(scores, k=k)

    return [(int(i), float(v)) for i, v in zip(indices.cpu().numpy(), values.cpu().numpy())]


def get_seen_items(dataset, user_id: int) -> np.ndarray:
    seen = dataset.getUserPosItems([user_id])[0]
    return np.array(seen, dtype=np.int64)


def main():
    st.set_page_config(page_title="LightGCN Defense Dashboard", page_icon="📊", layout="wide")
    inject_custom_style()
    st.markdown(
        """
        <div class="hero-wrap">
            <div class="hero-title">LightGCN/NGCF 一体化答辩页面</div>
            <p class="hero-sub">左侧：实验结果看板 | 右侧：在线实时推荐演示（真实模型推理）</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    comparison_data = load_json(COMPARISON_JSON)
    cost_data = load_json(COST_JSON)
    topk_data = load_json(TOPK_JSON)
    ngcf_data = load_json(NGCF_JSON)

    model_df = build_model_metrics_df(comparison_data, cost_data)
    seed_df = build_seed_variation_df(comparison_data)
    topk_df = build_topk_df(topk_data)

    left_col, right_col = st.columns([1.2, 1.0], gap="large")

    with left_col:
        st.markdown('<div class="section-title">实验结果展示层</div>', unsafe_allow_html=True)

        if not model_df.empty:
            best_row = model_df.dropna(subset=["Recall@20"]).sort_values("Recall@20", ascending=False).iloc[0]
            c1, c2, c3 = st.columns(3)
            c1.metric("Best Recall@20", f"{best_row['Recall@20']:.5f}", best_row["Model"])
            c2.metric("Best NDCG@20", f"{best_row['NDCG@20']:.5f}", best_row["Model"])
            delta_recall = comparison_data.get("aggregate", {}).get("delta_recall_mean", np.nan)
            c3.metric("LightGCN - NGCF (Recall@20)", f"{delta_recall:.5f}")

        st.markdown("#### 模型总览（Recall@20 / NDCG@20 / 训练耗时）")
        st.dataframe(
            model_df.style.format(
                {
                    "Recall@20": "{:.5f}",
                    "Recall_std": "{:.5f}",
                    "NDCG@20": "{:.5f}",
                    "NDCG_std": "{:.5f}",
                    "Sec/Epoch": "{:.3f}",
                    "Total Hours": "{:.3f}",
                }
            ),
            use_container_width=True,
            height=220,
        )

        if not seed_df.empty:
            st.markdown("#### seed 波动")
            metric_pick = st.radio("波动指标", ["Recall@20", "NDCG@20"], horizontal=True)
            pivot_seed = seed_df.pivot(index="seed", columns="model", values=metric_pick)
            st.line_chart(pivot_seed, use_container_width=True)

        if not topk_df.empty:
            st.markdown("#### Top-K 趋势")
            topk_curve = topk_df.set_index("K")[["Recall@K", "NDCG@K"]]
            st.line_chart(topk_curve, use_container_width=True)

        st.markdown("#### 收敛曲线")
        logs = discover_curve_logs()
        lgn_choices = [f"seed={seed} | {path.name}" for seed, path in logs["lgn"]]
        ngcf_choices = [f"seed={seed} | {path.name}" for seed, path in logs["ngcf"]]

        selected_lgn = st.selectbox("LightGCN 曲线日志", lgn_choices, index=0 if lgn_choices else None)
        selected_ngcf = st.selectbox("NGCF 曲线日志", ngcf_choices, index=0 if ngcf_choices else None)

        def pick_path(selection: Optional[str], all_logs: List[Tuple[int, Path]]) -> Optional[Path]:
            if not selection:
                return None
            name = selection.split("|", 1)[1].strip()
            for _, p in all_logs:
                if p.name == name:
                    return p
            return None

        lgn_path = pick_path(selected_lgn, logs["lgn"])
        ngcf_path = pick_path(selected_ngcf, logs["ngcf"])

        curve_metric = st.radio("收敛指标", ["recall", "ndcg", "precision"], horizontal=True)

        curve_series = {}
        if lgn_path:
            lgn_curve = parse_curve_from_log(lgn_path)
            if not lgn_curve.empty:
                curve_series["LightGCN"] = lgn_curve.set_index("epoch")[curve_metric]
        if ngcf_path:
            ngcf_curve = parse_curve_from_log(ngcf_path)
            if not ngcf_curve.empty:
                curve_series["NGCF"] = ngcf_curve.set_index("epoch")[curve_metric]

        if curve_series:
            curve_df = pd.concat(curve_series, axis=1)
            st.line_chart(curve_df, use_container_width=True)
        else:
            st.info("未找到可解析的收敛日志。")

        if ngcf_data:
            st.caption("注: NGCF 统计来自 300 epoch x 3 seeds 汇总。")

    with right_col:
        st.markdown('<div class="section-title">在线推荐演示层</div>', unsafe_allow_html=True)
        render_explain_card(
            "实时推理说明",
            "右侧结果由所选 checkpoint 现场计算得到，不是预先写死的静态表。切换模型权重、切换用户ID，输出会即时变化。",
        )

        ckpt_files = checkpoint_options()
        if not ckpt_files:
            st.error("checkpoints 目录下没有可用的 yelp2018 权重文件。")
            st.stop()

        default_idx = 0
        for i, name in enumerate(ckpt_files):
            if "lgn-yelp2018-3-64-mean-epoch300-seed2020" in name:
                default_idx = i
                break

        ckpt_name = st.selectbox("选择推理权重", ckpt_files, index=default_idx)
        model_key = infer_model_from_ckpt(ckpt_name).upper()
        agg_mode = infer_layer_agg_from_ckpt(ckpt_name)
        render_explain_card(
            "当前推理配置",
            f"模型: {model_key} | 层聚合: {agg_mode}。该配置仅影响右侧在线推理，不会改动左侧离线实验统计。",
        )

        user_org_to_remap, user_remap_to_org = load_id_map(DATA_DIR / "user_list.txt")
        _, item_remap_to_org = load_id_map(DATA_DIR / "item_list.txt")

        input_mode = st.radio("用户输入方式", ["remap_id", "org_id"], horizontal=True)
        topk = st.slider("Top-K", min_value=5, max_value=50, value=20, step=5)

        if input_mode == "remap_id":
            max_user = max(user_remap_to_org.keys()) if user_remap_to_org else 0
            user_input_num = st.number_input("用户 remap_id", min_value=0, max_value=max_user, value=0, step=1)
            input_user_remap = int(user_input_num)
        else:
            user_input_text = st.text_input("用户 org_id", value="")
            input_user_remap = user_org_to_remap.get(user_input_text, -1)
            if user_input_text and input_user_remap < 0:
                st.warning("这个 org_id 不在当前 yelp2018 映射表中。")

        run_btn = st.button("开始实时推荐", type="primary")
        if run_btn:
            if input_user_remap < 0:
                st.error("用户ID无效，请检查输入。")
            else:
                try:
                    rec_model, dataset, device_str = load_inference_bundle(ckpt_name)
                    seen_items = get_seen_items(dataset, input_user_remap)
                    seen_count = int(seen_items.size)
                    results = recommend_topk(rec_model, dataset, input_user_remap, topk)

                    st.success(f"推理完成 (device={device_str})")
                    render_explain_card(
                        "已训练交互过滤说明",
                        (
                            f"该用户在训练集中已交互物品数: {seen_count}。系统在实时推理时会先把这些已见物品屏蔽，"
                            "再从剩余候选中选 Top-K，所以输出是可推荐的新物品列表。"
                        ),
                    )
                    render_explain_card(
                        "如何向老师解释这不是静态表",
                        "你可以现场更换用户ID或切换 checkpoint，推荐结果会即时变化；同时已训练交互过滤数量也会变化，这是在线推理路径的直接证据。",
                    )
                    st.markdown("#### 推荐结果")

                    rows = []
                    for rank, (item_remap, score) in enumerate(results, start=1):
                        rows.append(
                            {
                                "rank": rank,
                                "item_remap_id": item_remap,
                                "item_org_id": item_remap_to_org.get(item_remap, "N/A"),
                                "score": score,
                            }
                        )
                    result_df = pd.DataFrame(rows)
                    st.dataframe(result_df, use_container_width=True, height=520)

                    with st.expander("查看该用户训练阶段已交互物品（用于过滤）"):
                        seen_show = pd.DataFrame(
                            {
                                "item_remap_id": seen_items[:200],
                                "item_org_id": [item_remap_to_org.get(int(x), "N/A") for x in seen_items[:200]],
                            }
                        )
                        st.dataframe(seen_show, use_container_width=True, height=220)
                        if seen_count > 200:
                            st.caption(f"仅展示前 200 条，完整已交互数量为 {seen_count}。")

                    if input_user_remap in user_remap_to_org:
                        st.caption(f"当前用户 org_id: {user_remap_to_org[input_user_remap]}")
                except Exception as exc:
                    st.exception(exc)


if __name__ == "__main__":
    main()
