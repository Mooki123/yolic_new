"""research/13: the 2x2 (YOLIC head / Align head) x (224 input / foveated input) on Cityscapes, and its verdict.

    python tools/align_fovea_verdict.py
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from yolic_bench import metrics as M  # noqa: E402
from yolic_bench.data import CellCache  # noqa: E402

RUNS = {'A': 'cityscapes_b1', 'AL': 'cityscapes_align', 'F': 'cityscapes_f_fovea_pvbn', 'F+AL': 'cityscapes_f_align'}
PV = [0, 1]


def load(name):
    d = os.path.join(ROOT, 'runs', name)
    ss = [s for s in range(3) if os.path.exists(os.path.join(d, f'seed{s}', 'metrics.json'))]
    ms = [json.load(open(os.path.join(d, f'seed{s}', 'metrics.json'))) for s in ss]
    pr = np.stack([np.load(os.path.join(d, f'seed{s}', 'test_probs.npy')).astype(np.float32) > 0.5 for s in ss])
    return ms, pr


def main():
    c = CellCache(os.path.expanduser('~/data/yolic_cache/cityscapes'))
    te = c.indices(os.path.join(ROOT, 'splits', 'cityscapes', 'test.txt'))
    gt, valid = c.labels[te], c.valid[te]
    R = {k: load(v) for k, v in RUNS.items()}
    table = {}
    for k, (ms, _) in R.items():
        mean = lambda f: round(100 * float(np.mean([f(m) for m in ms])), 2)
        table[k] = {'seeds': [round(100 * m['primary_macro_f1'], 2) for m in ms], 'P+V': mean(lambda m: m['primary_macro_f1']),
                    'far': mean(lambda m: m['by_group_primary']['small_64x32']['object_macro_f1']),
                    'near': mean(lambda m: m['by_group_primary']['large_128x64']['object_macro_f1']),
                    'All': mean(lambda m: m['all_paper']['f1']), 'params_M': round(ms[0]['params'] / 1e6, 3),
                    'MACs_x_A': round(ms[0]['macs'] / R['A'][0][0]['macs'], 3)}

    def delta(a, b):
        d, lo, hi = M.paired_bootstrap(gt, R[a][1], R[b][1], PV, n_boot=1000, seed=0, valid=valid)
        return [round(100 * d, 2), round(100 * lo, 2), round(100 * hi, 2)]

    q1, q2 = delta('A', 'AL'), delta('F', 'F+AL')
    ratio = table['F+AL']['params_M'] / table['F']['params_M']
    q2_yes = (q2[0] >= 1.0 and q2[1] > 0) or (abs(q2[0]) <= 0.5 and ratio <= 0.6)
    res = {'table': table, 'AL - A': q1, 'F+AL - F': q2, 'F+AL - A': delta('A', 'F+AL'),
           'params F+AL / F': round(ratio, 3),
           'Q1 (Align helps alone)': bool(q1[0] >= 1.0 and q1[1] > 0), 'Q2 (combination worth keeping)': bool(q2_yes)}
    print(json.dumps(res, indent=1))
    json.dump(res, open(os.path.join(ROOT, 'runs', 'logs', 'align_fovea_verdict.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
