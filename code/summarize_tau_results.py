import json
import os
from pathlib import Path

import torch
import numpy as np

import world
import dataloader
import model
import Procedure

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
DOCS_DIR = ROOT / "docs"
RESULTS_DIR.mkdir(exist_ok=True)
DOCS_DIR.mkdir(exist_ok=True)


def eval_ckpt(dataset, ckpt_name, tau, layer_w_l2=1e-3):
    world.tensorboard = 0
    world.topks = [10, 20, 50]
    world.config['multicore'] = 0
    world.config['test_u_batch_size'] = 100
    world.config['layer_agg'] = 'learnable'
    world.config['layer_w_l2'] = layer_w_l2
    world.config['layer_w_tau'] = tau

    rec = model.LightGCN(world.config, dataset).to(world.device)
    state = torch.load(str(Path(world.FILE_PATH) / ckpt_name), map_location=world.device)
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


def agg_rows(rows):
    prec = np.array([r['result']['precision'] for r in rows], dtype=float)
    rec = np.array([r['result']['recall'] for r in rows], dtype=float)
    nd = np.array([r['result']['ndcg'] for r in rows], dtype=float)
    return {
        'precision': {'mean': prec.mean(axis=0).tolist(), 'std': prec.std(axis=0).tolist()},
        'recall': {'mean': rec.mean(axis=0).tolist(), 'std': rec.std(axis=0).tolist()},
        'ndcg': {'mean': nd.mean(axis=0).tolist(), 'std': nd.std(axis=0).tolist()},
    }


def main():
    dataset = dataloader.Loader(path=str(ROOT / 'data' / 'yelp2018'))
    taus = [0.5, 1.0, 2.0]
    seeds = [2020, 2021, 2022]

    all_out = {
        'dataset': 'yelp2018',
        'setting': 'learnable + layer_w_l2=1e-3 + 150 epochs quick screen',
        'topks': [10, 20, 50],
        'taus': {}
    }

    for tau in taus:
        tag_tau = str(tau).replace('.', 'p')
        rows = []
        for seed in seeds:
            ckpt = f'lgn-yelp2018-3-64-yelp150_tau{tag_tau}_l2_1e-3_s{seed}.pth.tar'
            rows.append({'seed': seed, 'result': eval_ckpt(dataset, ckpt, tau=tau)})
        all_out['taus'][str(tau)] = {
            'rows': rows,
            'aggregate': agg_rows(rows)
        }

    json_path = RESULTS_DIR / 'Tau_Screen_Results_Yelp2018.json'
    md_path = DOCS_DIR / 'Tau_Screen_Results_Yelp2018.md'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_out, f, indent=2)

    lines = []
    lines.append('# Temperature Screen Results (Yelp2018)')
    lines.append('')
    lines.append('- Variant: layer_agg=learnable, layer_w_l2=1e-3, epochs=150')
    lines.append('- Tau candidates: 0.5, 1.0, 2.0')
    lines.append('- Seeds: 2020, 2021, 2022')
    lines.append('')
    lines.append('| Tau | Precision@10 | Precision@20 | Precision@50 | Recall@10 | Recall@20 | Recall@50 | NDCG@10 | NDCG@20 | NDCG@50 |')
    lines.append('|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')

    for tau in taus:
        ag = all_out['taus'][str(tau)]['aggregate']
        p = ag['precision']['mean']
        r = ag['recall']['mean']
        n = ag['ndcg']['mean']
        lines.append(
            f"| {tau} | {p[0]:.6f} | {p[1]:.6f} | {p[2]:.6f} | {r[0]:.6f} | {r[1]:.6f} | {r[2]:.6f} | {n[0]:.6f} | {n[1]:.6f} | {n[2]:.6f} |"
        )

    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')

    print(f'saved {json_path}')
    print(f'saved {md_path}')


if __name__ == '__main__':
    main()
