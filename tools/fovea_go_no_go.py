"""Apply the decision rule of research/05_fovea_gonogo.md (written before any run) to the run directories.

    python tools/fovea_go_no_go.py
"""
import json
import os
import sys

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from yolic_bench import metrics as M  # noqa: E402
from yolic_bench.data import CellCache  # noqa: E402

RUNS = {'A': 'cityscapes_b1', 'F': 'cityscapes_f_fovea', 'R': 'cityscapes_r_336x168'}
PRIMARY = [0, 1]


def seeds(name):
    d = os.path.join(ROOT, 'runs', name)
    return sorted(s for s in os.listdir(d) if os.path.exists(os.path.join(d, s, 'metrics.json'))) if os.path.isdir(d) else []


def main():
    got = {k: seeds(v) for k, v in RUNS.items()}
    if any(len(v) < 3 for v in got.values()):
        print(json.dumps({'status': 'incomplete', 'seeds': {k: len(v) for k, v in got.items()}}))
        return
    m = {k: [json.load(open(os.path.join(ROOT, 'runs', RUNS[k], s, 'metrics.json'))) for s in got[k]] for k in RUNS}
    f1 = {k: [100 * x['primary_macro_f1'] for x in v] for k, v in m.items()}
    mean = {k: float(np.mean(v)) for k, v in f1.items()}

    cfg = yaml.safe_load(open(os.path.join(ROOT, 'configs', RUNS['A'] + '.yaml')))
    cache = CellCache(os.path.expanduser(cfg['cache']))
    te = cache.indices(os.path.join(ROOT, cfg['split'], 'test.txt'))
    gt, valid = cache.labels[te], cache.valid[te]

    def preds(k):
        return np.stack([np.load(os.path.join(ROOT, 'runs', RUNS[k], s, 'test_probs.npy')).astype(np.float32) > 0.5
                         for s in got[k]])

    P = {k: preds(k) for k in RUNS}
    boot = {}
    for a, b in (('A', 'F'), ('R', 'F'), ('A', 'R')):
        d, lo, hi = M.paired_bootstrap(gt, P[a], P[b], PRIMARY, n_boot=1000, seed=0, valid=valid)
        boot[f'{b}-{a}'] = [round(100 * d, 2), round(100 * lo, 2), round(100 * hi, 2)]

    fa, fr, ra = mean['F'] - mean['A'], mean['F'] - mean['R'], mean['R'] - mean['A']
    sep = min(f1['F']) > max(f1['A']) and min(f1['F']) > max(f1['R'])
    ci_ok = boot['F-A'][1] > 0 and boot['F-R'][1] > 0
    n = min(len(v) for v in f1.values())
    if n >= 5:
        verdict = 'GO (5-seed)' if fa >= 1.5 and fr >= 1.0 and ci_ok else 'FAIL'
    elif fa >= 2.0 and fr >= 1.0 and sep and ci_ok:
        verdict = 'GO'
    elif ra >= 2.0 and fr < 1.0:
        verdict = 'RESOLUTION ONLY'
    elif 1.0 <= fa < 2.0 or (fa >= 2.0 and not sep):
        verdict = 'AMBIGUOUS: add 2 seeds to F, A and R'
    else:
        verdict = 'FAIL'

    def grp(k, g):
        return round(100 * float(np.mean([x['by_group_primary'][g]['object_macro_f1'] for x in m[k]])), 2)

    print(json.dumps({
        'primary_macro_f1': {k: [round(x, 2) for x in v] for k, v in f1.items()},
        'means': {k: round(v, 2) for k, v in mean.items()},
        'F-A': round(fa, 2), 'F-R': round(fr, 2), 'R-A': round(ra, 2),
        'F_seeds_above_A_and_R': sep, 'bootstrap_delta_ci': boot,
        'small_cells': {k: grp(k, 'small_64x32') for k in RUNS},
        'large_cells': {k: grp(k, 'large_128x64') for k in RUNS},
        'all_f1': {k: round(100 * float(np.mean([x['all_paper']['f1'] for x in m[k]])), 2) for k in RUNS},
        'MMACs': {k: round(m[k][0]['macs'] / 1e6) for k in RUNS},
        'verdict': verdict,
    }, indent=1))


if __name__ == '__main__':
    main()
