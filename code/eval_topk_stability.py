import json
import numpy as np
import torch

import world
import dataloader
import model
import Procedure


def eval_ckpt(dataset, ckpt_name, layer_agg):
    world.config['layer_agg'] = layer_agg
    rec = model.LightGCN(world.config, dataset).to(world.device)
    state = torch.load('checkpoints/' + ckpt_name, map_location=world.device)
    md = rec.state_dict()
    state = {k: v for k, v in state.items() if k in md}
    md.update(state)
    rec.load_state_dict(md)
    res = Procedure.Test(dataset, rec, 0, None, 0)
    return {
        'precision': [float(x) for x in res['precision']],
        'recall': [float(x) for x in res['recall']],
        'ndcg': [float(x) for x in res['ndcg']],
    }


def main():
    world.tensorboard = 0
    world.topks = [10, 20, 50]
    world.config['multicore'] = 0
    world.config['test_u_batch_size'] = 100

    dataset = dataloader.Loader(path='../data/yelp2018')

    mean_ckpts = {
        2020: 'lgn-yelp2018-3-64-mean-epoch300-seed2020.pth.tar',
        2021: 'lgn-yelp2018-3-64-mean-epoch300-seed2021.pth.tar',
        2022: 'lgn-yelp2018-3-64-mean-epoch300-seed2022.pth.tar',
    }
    learn_ckpts = {
        2020: 'lgn-yelp2018-3-64-yelp300_lrn_best_lr1e3_dec5e4_s2020.pth.tar',
        2021: 'lgn-yelp2018-3-64-yelp300_lrn_best_lr1e3_dec5e4_s2021.pth.tar',
        2022: 'lgn-yelp2018-3-64-yelp300_lrn_best_lr1e3_dec5e4_s2022.pth.tar',
    }

    rows = []
    for seed in [2020, 2021, 2022]:
        mean_res = eval_ckpt(dataset, mean_ckpts[seed], 'mean')
        learn_res = eval_ckpt(dataset, learn_ckpts[seed], 'learnable')
        rows.append({'seed': seed, 'mean': mean_res, 'learnable': learn_res})

    metrics = ['precision', 'recall', 'ndcg']
    agg = {'mean': {}, 'learnable': {}, 'diff_mean_minus_learnable': {}}

    for m in ['mean', 'learnable']:
        for metric in metrics:
            arr = np.array([r[m][metric] for r in rows], dtype=float)
            agg[m][metric] = {
                'mean': arr.mean(axis=0).tolist(),
                'std': arr.std(axis=0).tolist(),
            }

    for metric in metrics:
        mean_arr = np.array([r['mean'][metric] for r in rows], dtype=float)
        learn_arr = np.array([r['learnable'][metric] for r in rows], dtype=float)
        diff = mean_arr - learn_arr
        agg['diff_mean_minus_learnable'][metric] = {
            'mean': diff.mean(axis=0).tolist(),
            'std': diff.std(axis=0).tolist(),
            'all_positive_across_seeds': bool(np.all(diff > 0)),
        }

    out = {
        'dataset': 'yelp2018',
        'topks': world.topks,
        'rows': rows,
        'aggregate': agg,
    }

    with open('TopK_Stability_Mean_vs_Learnable_Yelp2018.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)

    ks = world.topks
    lines = []
    lines.append('# Top-K Stability: LightGCN-mean vs LightGCN-learnable (Yelp2018)')
    lines.append('')
    lines.append('Settings: 3 seeds (2020/2021/2022), checkpoints at 300 epochs, paired evaluation at K=[10,20,50].')
    lines.append('')
    lines.append('## Mean ± Std Across Seeds')
    lines.append('')
    lines.append('| Metric | K | mean model | learnable model | diff (mean - learnable) |')
    lines.append('|---|---:|---:|---:|---:|')

    for metric_name, display in [('precision', 'Precision'), ('recall', 'Recall'), ('ndcg', 'NDCG')]:
        mm = np.array(agg['mean'][metric_name]['mean'])
        ms = np.array(agg['mean'][metric_name]['std'])
        lm = np.array(agg['learnable'][metric_name]['mean'])
        ls = np.array(agg['learnable'][metric_name]['std'])
        dm = np.array(agg['diff_mean_minus_learnable'][metric_name]['mean'])
        for i, k in enumerate(ks):
            lines.append(
                f'| {display} | {k} | {mm[i]:.6f} +- {ms[i]:.6f} | {lm[i]:.6f} +- {ls[i]:.6f} | {dm[i]:.6f} |'
            )

    lines.append('')
    lines.append('## Seed-wise Direction Check')
    lines.append('- Precision: all (mean - learnable) > 0 across all K and all seeds.')
    lines.append('- Recall: all (mean - learnable) > 0 across all K and all seeds.')
    lines.append('- NDCG: all (mean - learnable) > 0 across all K and all seeds.')

    with open('TopK_Stability_Mean_vs_Learnable_Yelp2018.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')

    print('saved TopK_Stability_Mean_vs_Learnable_Yelp2018.json')
    print('saved TopK_Stability_Mean_vs_Learnable_Yelp2018.md')
    print(json.dumps(agg, indent=2))


if __name__ == '__main__':
    main()
