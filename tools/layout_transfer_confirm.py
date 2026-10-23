"""research/09: 3-seed confirmation of layout transfer at matched compute, with same-architecture oracles.

    python tools/layout_transfer_confirm.py
"""
import json
import os
import sys

import numpy as np
import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tools'))
import layout_transfer_eval as E  # noqa: E402
from yolic_bench import heads  # noqa: E402
from yolic_bench import metrics as M  # noqa: E402
from yolic_bench.data import CellCache, GpuAugment, batches  # noqa: E402

DEV, PV, SEEDS = 'cuda', [0, 1], (0, 1, 2)
L1 = 'cityscapes'
TARGETS = ['cityscapes', 'cityscapes_union', 'cityscapes_shift', 'cityscapes_fine', 'cityscapes_poly']
HARD = ['cityscapes_shift', 'cityscapes_fine', 'cityscapes_poly']


def seeds_of(name):
    d = os.path.join(ROOT, 'runs', name)
    return [s for s in SEEDS if os.path.exists(os.path.join(d, f'seed{s}', 'test_probs.npy'))]


@torch.no_grad()
def transfer_probs(name, seed, cache, idx, lays):
    """{layout: (N, n_cells, 4)} for an L1-trained MIL model pointed at every layout."""
    d = os.path.join(ROOT, 'runs', name, f'seed{seed}')
    cfg = yaml.safe_load(open(os.path.join(d, 'config.yaml')))
    m = heads.build(cfg['model'], lays[L1], 4, pretrained=False).to(DEV)
    m.load_state_dict(torch.load(os.path.join(d, 'best.pt'), map_location=DEV))
    m.eval()
    aug = GpuAugment(None, 'none', None, True, resize=cfg.get('resize'))
    out = {n: np.zeros((len(idx), len(l), 4), np.float32) for n, l in lays.items()}
    for imgs, _, _, _, pos in batches(cache, idx, 64, shuffle=False):
        with torch.autocast('cuda', dtype=torch.float16):
            f = m.features(aug.to_input(imgs[0].to(DEV)))
        for n, l in lays.items():
            out[n][pos] = torch.sigmoid(m.head.set_layout(l)(f.float())).cpu().numpy()
    return out


def oracle_name(prefix, target):
    return 'cityscapes_b1' if (prefix == 'cityscapes_oracle' and target == L1) else f'{prefix}_{target.split("_")[1]}'


def main():
    cfg = yaml.safe_load(open(os.path.join(ROOT, 'configs', 'cityscapes_b1.yaml')))
    lays = {n: E.layout(n) for n in TARGETS}
    caches = {n: CellCache(os.path.expanduser(cfg['cache']), labels=None if n == L1 else n) for n in TARGETS}
    te = caches[L1].indices(os.path.join(ROOT, cfg['split'], 'test.txt'))
    gt = {n: (caches[n].labels[te], caches[n].valid[te]) for n in TARGETS}

    P = {}  # (method, target) -> (S, N, n_cells, 4)
    a = [np.load(os.path.join(ROOT, 'runs', 'cityscapes_b1', f'seed{s}', 'test_probs.npy')).astype(np.float32)
         for s in SEEDS]
    for n in TARGETS:
        P[('overlap (A)', n)] = np.stack([E.overlap_probs(x, lays[L1], lays[n]) for x in a])
        for label, prefix in (('O_A (YOLIC retrained)', 'cityscapes_oracle'), ('O_T (T retrained)', 'cityscapes_oracleT')):
            if n == L1 and prefix == 'cityscapes_oracleT':
                continue  # T on L1 is its own oracle
            name = oracle_name(prefix, n)
            ss = seeds_of(name)
            if ss:
                P[(label, n)] = np.stack([np.load(os.path.join(ROOT, 'runs', name, f'seed{s}', 'test_probs.npy'))
                                          .astype(np.float32) for s in ss])
    for label, name in (('T (MIL-max s16, 168)', 'cityscapes_mil_max_s16_r168'), ('MIL-max s16, 224', 'cityscapes_mil_max_s16')):
        per = [transfer_probs(name, s, caches[L1], te, lays) for s in seeds_of(name)]
        for n in TARGETS:
            P[(label, n)] = np.stack([p[n] for p in per])
        print('scored', label, len(per), 'seeds', flush=True)

    def f1(p, n):
        return [round(100 * float(M.macro_f1(M.counts(gt[n][0], x > 0.5, gt[n][1]), PV)), 2) for x in p]

    table = {n: {k[0]: {'seeds': f1(v, n), 'mean': round(float(np.mean(f1(v, n))), 2)}
                 for k, v in P.items() if k[1] == n} for n in TARGETS}
    T = 'T (MIL-max s16, 168)'
    checks = {}
    for n in HARD:
        t, ot = table[n][T]['mean'], table[n]['O_T (T retrained)']['mean']
        d, lo, hi = M.paired_bootstrap(gt[n][0], P[('overlap (A)', n)] > 0.5, P[(T, n)] > 0.5, PV,
                                       n_boot=1000, seed=0, valid=gt[n][1])
        checks[n] = {'T': t, 'O_T': ot, 'T/O_T': round(t / ot, 3), 'O_A': table[n]['O_A (YOLIC retrained)']['mean'],
                     'T-overlap': round(100 * d, 2), 'ci95': [round(100 * lo, 2), round(100 * hi, 2)],
                     'O_T seeds': len(table[n]['O_T (T retrained)']['seeds']),
                     'C1': bool(t >= 0.9 * ot and 100 * d >= 3.0 and lo > 0)}
    c2 = bool(table[L1][T]['mean'] >= table[L1]['overlap (A)']['mean'] - 1.0)
    n_c1 = sum(c['C1'] for c in checks.values())
    verdict = 'CONFIRMED' if n_c1 == 3 and c2 else ('PARTIAL' if n_c1 >= 2 or (n_c1 == 3 and not c2) else 'NOT CONFIRMED')
    res = {'table (P+V macro-F1, test)': table, 'checks': checks,
           'C2': {'T on L1': table[L1][T]['mean'], 'A on L1': table[L1]['overlap (A)']['mean'], 'pass': c2},
           'verdict': verdict}
    print(json.dumps(res, indent=1))
    json.dump(res, open(os.path.join(ROOT, 'runs', 'logs', 'layout_transfer_confirm.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
