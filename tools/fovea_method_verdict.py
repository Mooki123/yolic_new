"""Apply the verdict of research/06_fovea_method_test.md (thresholds written before the runs).

M1 (Cityscapes test, People+Vehicle macro-F1): V5 - A >= 3.0 with CI excluding 0, and V5 - R >= 1.5.
M2 (outdoor grouped split, object macro-F1 over 11 classes): bands - B1 >= 2.0 with CI excluding 0.

    python tools/fovea_method_verdict.py                     # the original (shared-BN, invalid) runs
    python tools/fovea_method_verdict.py --cityscapes cityscapes_f_v5_pvbn --outdoor outdoor_f_bands_grouped_pvbn
"""
import argparse
import json
import os
import sys

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from yolic_bench import metrics as M  # noqa: E402
from yolic_bench.data import CellCache  # noqa: E402


def load(name):
    d = os.path.join(ROOT, 'runs', name)
    seeds = sorted(s for s in os.listdir(d) if os.path.exists(os.path.join(d, s, 'metrics.json')))
    ms = [json.load(open(os.path.join(d, s, 'metrics.json'))) for s in seeds]
    preds = np.stack([np.load(os.path.join(d, s, 'test_probs.npy')).astype(np.float32) > 0.5 for s in seeds])
    return ms, preds


def compare(base, new, classes, key):
    cfg = yaml.safe_load(open(os.path.join(ROOT, 'configs', base + '.yaml')))
    cache = CellCache(os.path.expanduser(cfg['cache']))
    te = cache.indices(os.path.join(ROOT, cfg['split'], 'test.txt'))
    gt = cache.labels[te]
    valid = cache.valid[te] if cache.valid is not None else None
    (mb, pb), (mn, pn) = load(base), load(new)
    d, lo, hi = M.paired_bootstrap(gt, pb, pn, classes, n_boot=1000, seed=0, valid=valid)
    f = lambda ms: [round(100 * m[key], 2) for m in ms]
    return {'base_seeds': f(mb), 'new_seeds': f(mn), 'base_mean': round(float(np.mean(f(mb))), 2),
            'new_mean': round(float(np.mean(f(mn))), 2), 'delta': round(100 * d, 2),
            'ci95': [round(100 * lo, 2), round(100 * hi, 2)],
            'groups': {g: [round(100 * float(np.mean([m['by_group_primary' if 'by_group_primary' in m else 'by_group'][g]['object_macro_f1'] or 0 for m in ms])), 2)
                           for ms in (mb, mn)] for g in mb[0].get('by_group_primary', mb[0]['by_group'])},
            'macs_ratio': round(next(m['macs'] for m in mn if 'macs' in m) / next(m['macs'] for m in mb if 'macs' in m), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cityscapes', default='cityscapes_f_v5', help='run name of the stage-2 winner')
    ap.add_argument('--outdoor', default='outdoor_f_bands_grouped', help='run name of the stage-3 outdoor model')
    args = ap.parse_args()
    cs_a = compare('cityscapes_b1', args.cityscapes, [0, 1], 'primary_macro_f1')
    cs_r = compare('cityscapes_r_336x168', args.cityscapes, [0, 1], 'primary_macro_f1')
    od = compare('outdoor_b1_grouped', args.outdoor, list(range(11)), 'object_macro_f1')
    m1 = bool(cs_a['delta'] >= 3.0 and cs_a['ci95'][0] > 0 and cs_r['delta'] >= 1.5)
    m2 = bool(od['delta'] >= 2.0 and od['ci95'][0] > 0)
    verdict = {(True, True): 'REAL METHOD', (False, True): 'REAL BUT MODEST (M2 only)',
               (True, False): 'CITYSCAPES-SPECIFIC (M1 only)', (False, False): 'TOPS OUT'}[(m1, m2)]
    print(json.dumps({'M1_cityscapes_V5_vs_A': cs_a, 'M1_cityscapes_V5_vs_R': cs_r, 'M2_outdoor_bands_vs_B1': od,
                      'M1': m1, 'M2': m2, 'verdict': verdict}, indent=1))


if __name__ == '__main__':
    main()
