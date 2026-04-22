import json
import torch

import world
import dataloader
import model
import Procedure


def eval_ckpt(dataset, ckpt_name, layer_agg='learnable', layer_w_l2=1e-3):
    world.tensorboard = 0
    world.topks = [10, 20, 50]
    world.config['multicore'] = 0
    world.config['test_u_batch_size'] = 100
    world.config['layer_agg'] = layer_agg
    world.config['layer_w_l2'] = layer_w_l2

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
    dataset = dataloader.Loader(path='../data/yelp2018')

    ckpts = {
        2020: 'lgn-yelp2018-3-64-yelp300_lrn_l2_1e-3_s2020.pth.tar',
        2021: 'lgn-yelp2018-3-64-yelp300_lrn_l2_1e-3_s2021.pth.tar',
        2022: 'lgn-yelp2018-3-64-yelp300_lrn_l2_1e-3_s2022.pth.tar',
    }

    rows = []
    for seed in [2020, 2021, 2022]:
        rows.append({'seed': seed, 'result': eval_ckpt(dataset, ckpts[seed])})

    import numpy as np
    prec = np.array([r['result']['precision'] for r in rows], dtype=float)
    rec = np.array([r['result']['recall'] for r in rows], dtype=float)
    nd = np.array([r['result']['ndcg'] for r in rows], dtype=float)

    out = {
        'dataset': 'yelp2018',
        'variant': 'learnable + layer_w_l2=1e-3',
        'topks': [10, 20, 50],
        'rows': rows,
        'aggregate': {
            'precision': {'mean': prec.mean(axis=0).tolist(), 'std': prec.std(axis=0).tolist()},
            'recall': {'mean': rec.mean(axis=0).tolist(), 'std': rec.std(axis=0).tolist()},
            'ndcg': {'mean': nd.mean(axis=0).tolist(), 'std': nd.std(axis=0).tolist()},
        }
    }

    with open('L2_Resume_Results_Yelp2018.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)

    ks = out['topks']
    lines = []
    lines.append('# L2-Regularized Learnable Results (Yelp2018)')
    lines.append('')
    lines.append('- Variant: layer_agg=learnable, layer_w_l2=1e-3')
    lines.append('- Seeds: 2020, 2021, 2022 (seed2021 resumed + seed2022 fresh)')
    lines.append('')
    lines.append('## Per-seed Results')
    lines.append('')
    lines.append('| Seed | Precision@10 | Precision@20 | Precision@50 | Recall@10 | Recall@20 | Recall@50 | NDCG@10 | NDCG@20 | NDCG@50 |')
    lines.append('|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for r in rows:
        p = r['result']['precision']
        rc = r['result']['recall']
        n = r['result']['ndcg']
        lines.append(
            f"| {r['seed']} | {p[0]:.6f} | {p[1]:.6f} | {p[2]:.6f} | {rc[0]:.6f} | {rc[1]:.6f} | {rc[2]:.6f} | {n[0]:.6f} | {n[1]:.6f} | {n[2]:.6f} |"
        )

    lines.append('')
    lines.append('## 3-seed Mean ± Std')
    lines.append('')
    lines.append('| Metric | @10 | @20 | @50 |')
    lines.append('|---|---:|---:|---:|')
    ag = out['aggregate']
    lines.append(
        f"| Precision | {ag['precision']['mean'][0]:.6f} +- {ag['precision']['std'][0]:.6f} | {ag['precision']['mean'][1]:.6f} +- {ag['precision']['std'][1]:.6f} | {ag['precision']['mean'][2]:.6f} +- {ag['precision']['std'][2]:.6f} |"
    )
    lines.append(
        f"| Recall | {ag['recall']['mean'][0]:.6f} +- {ag['recall']['std'][0]:.6f} | {ag['recall']['mean'][1]:.6f} +- {ag['recall']['std'][1]:.6f} | {ag['recall']['mean'][2]:.6f} +- {ag['recall']['std'][2]:.6f} |"
    )
    lines.append(
        f"| NDCG | {ag['ndcg']['mean'][0]:.6f} +- {ag['ndcg']['std'][0]:.6f} | {ag['ndcg']['mean'][1]:.6f} +- {ag['ndcg']['std'][1]:.6f} | {ag['ndcg']['mean'][2]:.6f} +- {ag['ndcg']['std'][2]:.6f} |"
    )

    with open('L2_Resume_Results_Yelp2018.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')

    print('saved L2_Resume_Results_Yelp2018.json')
    print('saved L2_Resume_Results_Yelp2018.md')


if __name__ == '__main__':
    main()
