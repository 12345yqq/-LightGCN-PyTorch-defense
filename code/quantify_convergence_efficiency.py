import json
import os
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

BASE = 'runs'
MEAN_RUNS = [
    '04-14-14h56m06s--lgn_mean_yelp300',
    '04-14-21h11m53s--yelp300_mean_s2021',
    '04-15-14h02m34s--yelp300_mean_s2022',
]
LEARN_RUNS = [
    '04-17-00h01m39s--yelp300_lrn_best_lr1e3_dec5e4_s2020',
    '04-17-08h09m35s--yelp300_lrn_best_lr1e3_dec5e4_s2021',
    '04-17-13h00m22s--yelp300_lrn_best_lr1e3_dec5e4_s2022',
]
METRICS = ['Recall@[20]', 'NDCG@[20]']


def first_event_file(metric_dir):
    if not os.path.isdir(metric_dir):
        return None
    files = [f for f in os.listdir(metric_dir) if f.startswith('events.out.tfevents.')]
    if not files:
        return None
    files.sort()
    return os.path.join(metric_dir, files[0])


def read_metric(run_name, metric):
    metric_dir = os.path.join(BASE, run_name, 'Test', metric, '20')
    event_file = first_event_file(metric_dir)
    if event_file is None:
        raise FileNotFoundError(metric_dir)
    ea = EventAccumulator(event_file)
    ea.Reload()
    tags = ea.Tags().get('scalars', [])
    if not tags:
        raise RuntimeError(f'No scalar tag in {event_file}')
    ev = ea.Scalars(tags[0])
    steps = np.array([x.step for x in ev], dtype=int)
    vals = np.array([x.value for x in ev], dtype=float)
    return steps, vals


def seed_stats(run_name, metric, threshold):
    steps, vals = read_metric(run_name, metric)
    # Use trapezoid area in step-domain; normalize by step span to compare across different lengths.
    auc_raw = float(np.trapz(vals, steps))
    span = float(max(steps[-1] - steps[0], 1))
    auc_norm = auc_raw / span

    hit_idx = np.where(vals >= threshold)[0]
    if len(hit_idx) == 0:
        reach_step = None
        reach_epoch = None
    else:
        reach_step = int(steps[hit_idx[0]])
        reach_epoch = int(reach_step + 10)

    return {
        'final_value': float(vals[-1]),
        'num_points': int(len(vals)),
        'last_step': int(steps[-1]),
        'auc_raw': auc_raw,
        'auc_norm': float(auc_norm),
        'reach_step_to_threshold': reach_step,
        'reach_epoch_to_threshold': reach_epoch,
    }


def summarize(items, key):
    arr = np.array([x[key] for x in items], dtype=float)
    return {'mean': float(arr.mean()), 'std': float(arr.std())}


def main():
    out = {
        'definition': {
            'threshold_rule': '95% of mean model final value (per metric, using 3-seed mean final)',
            'auc_rule': 'trapz(value over step), and normalized AUC = trapz / (last_step-first_step)',
        },
        'metrics': {}
    }

    for metric in METRICS:
        mean_series = [read_metric(r, metric) for r in MEAN_RUNS]
        mean_final_target = float(np.mean([vals[-1] for _, vals in mean_series]))
        threshold = 0.95 * mean_final_target

        mean_seed = [seed_stats(r, metric, threshold) for r in MEAN_RUNS]
        learn_seed = [seed_stats(r, metric, threshold) for r in LEARN_RUNS]

        reached_mean = [x for x in mean_seed if x['reach_step_to_threshold'] is not None]
        reached_learn = [x for x in learn_seed if x['reach_step_to_threshold'] is not None]

        out['metrics'][metric] = {
            'threshold': float(threshold),
            'mean_model': {
                'seed_stats': mean_seed,
                'reach_step_summary': summarize(reached_mean, 'reach_step_to_threshold') if reached_mean else None,
                'reach_epoch_summary': summarize(reached_mean, 'reach_epoch_to_threshold') if reached_mean else None,
                'auc_norm_summary': summarize(mean_seed, 'auc_norm'),
                'final_value_summary': summarize(mean_seed, 'final_value'),
            },
            'learnable_model': {
                'seed_stats': learn_seed,
                'reach_step_summary': summarize(reached_learn, 'reach_step_to_threshold') if reached_learn else None,
                'reach_epoch_summary': summarize(reached_learn, 'reach_epoch_to_threshold') if reached_learn else None,
                'auc_norm_summary': summarize(learn_seed, 'auc_norm'),
                'final_value_summary': summarize(learn_seed, 'final_value'),
            }
        }

    with open('Convergence_Efficiency_Quantification_Yelp2018.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)

    # Markdown report
    lines = []
    lines.append('# Convergence Efficiency Quantification (Yelp2018)')
    lines.append('')
    lines.append('- Models: LightGCN-mean vs LightGCN-learnable(best)')
    lines.append('- Seeds: 2020, 2021, 2022')
    lines.append('- Metrics: Recall@20, NDCG@20')
    lines.append('- Threshold: 95% of mean-model final value (per metric)')
    lines.append('- AUC: normalized trapezoid area over step domain')
    lines.append('')
    lines.append('## Summary Table')
    lines.append('')
    lines.append('| Metric | Threshold | Model | Reach Step (mean±std) | Reach Epoch (mean±std) | Norm-AUC (mean±std) | Final Value (mean±std) |')
    lines.append('|---|---:|---|---:|---:|---:|---:|')

    for metric in METRICS:
        block = out['metrics'][metric]
        thr = block['threshold']
        for model_key, model_name in [('mean_model', 'mean'), ('learnable_model', 'learnable(best)')]:
            m = block[model_key]
            rs = m['reach_step_summary']
            re = m['reach_epoch_summary']
            aucs = m['auc_norm_summary']
            fin = m['final_value_summary']
            rs_txt = 'N/A' if rs is None else f"{rs['mean']:.1f}±{rs['std']:.1f}"
            re_txt = 'N/A' if re is None else f"{re['mean']:.1f}±{re['std']:.1f}"
            lines.append(
                f"| {metric.replace('@[20]','@20')} | {thr:.6f} | {model_name} | {rs_txt} | {re_txt} | {aucs['mean']:.6f}±{aucs['std']:.6f} | {fin['mean']:.6f}±{fin['std']:.6f} |"
            )

    lines.append('')
    lines.append('## Key Findings')
    for metric in METRICS:
        b = out['metrics'][metric]
        mr = b['mean_model']['reach_step_summary']
        lr = b['learnable_model']['reach_step_summary']
        ma = b['mean_model']['auc_norm_summary']
        la = b['learnable_model']['auc_norm_summary']
        mr_txt = 'N/A' if mr is None else f"{mr['mean']:.1f}"
        lr_txt = 'N/A' if lr is None else f"{lr['mean']:.1f}"
        lines.append(
            f"- {metric.replace('@[20]','@20')}: mean reaches threshold at step {mr_txt} (vs {lr_txt} for learnable), and has normalized AUC {ma['mean']:.6f} (vs {la['mean']:.6f})."
        )

    with open('Convergence_Efficiency_Quantification_Yelp2018.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')

    print('saved Convergence_Efficiency_Quantification_Yelp2018.json')
    print('saved Convergence_Efficiency_Quantification_Yelp2018.md')


if __name__ == '__main__':
    main()
