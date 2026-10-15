"""Apply the go/no-go decision rules of plan 04 section 3.3 (thresholds frozen 2026-10-02) to run dirs.

    python tools/go_no_go.py            # prints both arms; an arm whose runs are missing says so

Arm L: leakage is material if random-split object macro-F1 - grouped-split object macro-F1 >= 3.0 points.
Arm H: Delta = (best compute-matched of B, C) - A on Cityscapes People+Vehicle macro-F1 at 0.5, in points,
       mean over seeds; GO / ambiguous / REVERSE / FAIL as in the plan. The rules leave two cases open
       (-2 < Delta <= -1, and |Delta| >= 2 with overlapping seeds once 5 seeds exist); these are reported
       as 'not covered by section 3.3' instead of inventing a threshold after the fact.
"""
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

LEAKAGE_POINTS = 3.0
GATE = (0.8202, 0.030)
ARM_H = {'A': 'cityscapes_b1', 'B': 'cityscapes_b4_cellpool_s32', 'C': 'cityscapes_b8_lraspp_s',
         'D': 'cityscapes_b5_cellpool_s16'}
MATCHED_MACS = 1.2


def runs(name):
    out = []
    for p in sorted(glob.glob(os.path.join(ROOT, 'runs', name, 'seed*', 'metrics.json'))):
        with open(p) as f:
            out.append(json.load(f))
    return out


def arm_l():
    r, g = runs('outdoor_b1_random'), runs('outdoor_b1_grouped')
    if not r or not g:
        return {'status': 'incomplete', 'random_runs': len(r), 'grouped_runs': len(g)}
    fr = 100 * np.mean([m['object_macro_f1'] for m in r])
    fg = 100 * np.mean([m['object_macro_f1'] for m in g])
    gap = fr - fg
    return {
        'random_object_macro_f1': round(fr, 2), 'grouped_object_macro_f1': round(fg, 2),
        'gap_points': round(gap, 2), 'threshold_points': LEAKAGE_POINTS,
        'leakage_material': bool(gap >= LEAKAGE_POINTS),
        'secondary': {k: {'random': round(100 * np.mean([m[k] for m in r]), 2),
                          'grouped': round(100 * np.mean([m[k] for m in g]), 2)} for k in ('mAP',)}
                     | {'tuned_object_macro_f1': {'random': round(100 * np.mean([m['tuned']['object_macro_f1'] for m in r]), 2),
                                                  'grouped': round(100 * np.mean([m['tuned']['object_macro_f1'] for m in g]), 2)}},
        'seeds': {'random': len(r), 'grouped': len(g)},
    }


def pv(m):
    return 100 * m['primary_macro_f1']


def arm_h():
    got = {k: runs(v) for k, v in ARM_H.items()}
    if not got['A'] or not (got['B'] or got['C']):
        return {'status': 'incomplete', 'runs': {k: len(v) for k, v in got.items()}}
    a = [pv(m) for m in got['A']]
    gate_a = abs(np.mean([m['all_paper']['f1'] for m in got['A']]) - GATE[0]) <= GATE[1]
    cands = {}
    for k in ('B', 'C'):
        if not got[k]:
            continue
        macs_ok = all(m.get('macs', 0) <= MATCHED_MACS * got['A'][0].get('macs', np.inf) for m in got[k])
        if macs_ok:
            cands[k] = [pv(m) for m in got[k]]
    if not cands:
        return {'status': 'no compute-matched alternative', 'runs': {k: len(v) for k, v in got.items()}}
    win = max(cands, key=lambda k: np.mean(cands[k]))
    w = cands[win]
    delta = np.mean(w) - np.mean(a)
    separated = min(w) > max(a)
    reversed_sep = max(max(v) for v in cands.values()) < min(a)
    n = min(len(a), len(w))
    if n >= 5:
        verdict = ('GO (5-seed)' if delta >= 1.5 else 'FAIL') + ' - also requires the paired-bootstrap CI to exclude 0'
    elif delta >= 2.0 and separated:
        verdict = 'GO'
    elif 1.0 <= delta < 2.0 or (delta >= 2.0 and not separated):
        verdict = 'AMBIGUOUS: add 2 seeds to A and ' + win
    elif all(np.mean(a) - np.mean(v) >= 2.0 for v in cands.values()) and reversed_sep:
        verdict = 'REVERSE'
    elif abs(delta) < 1.0:
        verdict = 'FAIL'
    else:
        verdict = 'not covered by section 3.3 - decide explicitly'
    return {'gate_A_all_f1_within_0.8202+-0.03': bool(gate_a), 'A_mean': round(np.mean(a), 2),
            'candidates': {k: round(np.mean(v), 2) for k, v in cands.items()}, 'winner': win,
            'delta_points': round(delta, 2), 'winner_seeds_all_above_A': bool(separated),
            'seeds': {k: len(v) for k, v in got.items()}, 'verdict': verdict,
            'note': 'B0 (untouched cityscapes_yolic.py) gate is checked by hand from its own eval output'}


if __name__ == '__main__':
    print(json.dumps({'arm_L': arm_l(), 'arm_H': arm_h()}, indent=1))
