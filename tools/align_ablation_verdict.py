"""research/15: Align ablations on F+AL — each variant vs the full model, and the pre-registered reading.

    python tools/align_ablation_verdict.py
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from yolic_bench import metrics as M  # noqa: E402
from yolic_bench.data import CellCache  # noqa: E402

FULL = 'cityscapes_f_align'
VARIANTS = {'no-lse (mean only)': 'cityscapes_f_align_no_lse', 'no-mean (soft max only)': 'cityscapes_f_align_no_mean',
            'no-lateral': 'cityscapes_f_align_no_lateral', 'no-cellbias': 'cityscapes_f_align_no_cellbias'}
PV = [0, 1]


def load(name):
    d = os.path.join(ROOT, 'runs', name)
    ss = [s for s in range(3) if os.path.exists(os.path.join(d, f'seed{s}', 'metrics.json'))]
    ms = [json.load(open(os.path.join(d, f'seed{s}', 'metrics.json'))) for s in ss]
    return ms, np.stack([np.load(os.path.join(d, f'seed{s}', 'test_probs.npy')).astype(np.float32) > 0.5 for s in ss])


def reading(d, lo, hi):
    if d <= -1.0 and hi < 0:
        return 'NEEDED'
    if abs(d) <= 0.5:
        return 'NOT NEEDED'
    if d >= 0.5 and lo > 0:
        return 'BETTER WITHOUT IT'
    return 'MINOR'


def main():
    c = CellCache(os.path.expanduser('~/data/yolic_cache/cityscapes'))
    te = c.indices(os.path.join(ROOT, 'splits', 'cityscapes', 'test.txt'))
    gt, valid = c.labels[te], c.valid[te]
    fm, fp = load(FULL)
    summary = lambda ms: {'seeds': [round(100 * m['primary_macro_f1'], 2) for m in ms],
                          'P+V': round(100 * float(np.mean([m['primary_macro_f1'] for m in ms])), 2),
                          'far': round(100 * float(np.mean([m['by_group_primary']['small_64x32']['object_macro_f1'] for m in ms])), 2),
                          'near': round(100 * float(np.mean([m['by_group_primary']['large_128x64']['object_macro_f1'] for m in ms])), 2),
                          'params_M': round(ms[0]['params'] / 1e6, 3), 'MMACs': round(ms[0]['macs'] / 1e6, 1)}
    res = {'F+AL (full)': summary(fm)}
    for label, name in VARIANTS.items():
        ms, pr = load(name)
        d, lo, hi = (100 * x for x in M.paired_bootstrap(gt, fp, pr, PV, n_boot=1000, seed=0, valid=valid))
        res[label] = summary(ms) | {'delta vs full': [round(d, 2), round(lo, 2), round(hi, 2)], 'reading': reading(d, lo, hi)}
    print(json.dumps(res, indent=1))
    json.dump(res, open(os.path.join(ROOT, 'runs', 'logs', 'align_ablation_verdict.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
