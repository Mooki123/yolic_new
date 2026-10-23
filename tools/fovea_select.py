"""research/06 stage 1: pick the foveation split s on Cityscapes val' only (seed 0).

Reads only best_val_primary_macro_f1 (val' People+Vehicle macro-F1 at the selected epoch) and never prints test
numbers, so the choice stays blind to the test set.
"""
import argparse
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VARIANTS = {'V1': ('cityscapes_f_fovea', 0.55), 'V2': ('cityscapes_f_v2', 0.75), 'V3': ('cityscapes_f_v3', 0.85),
            'V4': ('cityscapes_f_v4', 0.92), 'V5': ('cityscapes_f_v5', None)}
REFS = {'A (YOLIC)': 'cityscapes_b1', 'R (336x168)': 'cityscapes_r_336x168'}

ap = argparse.ArgumentParser()
ap.add_argument('--suffix', default='', help="run-name suffix of the variants, e.g. _pvbn (per-view BatchNorm)")
suffix = ap.parse_args().suffix
rows = {}
for k, (name, s) in {**VARIANTS, **{k: (v, None) for k, v in REFS.items()}}.items():
    name = name + suffix if k in VARIANTS else name
    p = os.path.join(ROOT, 'runs', name, 'seed0', 'metrics.json')
    rows[k] = round(100 * json.load(open(p))['best_val_primary_macro_f1'], 2) if os.path.exists(p) else None
done = {k: v for k, v in rows.items() if k in VARIANTS and v is not None}
print(json.dumps({'val_prime_P+V_seed0': rows,
                  'winner': max(done, key=done.get) if len(done) == len(VARIANTS) else 'incomplete',
                  's': {k: s for k, (_, s) in VARIANTS.items()}}, indent=1))
