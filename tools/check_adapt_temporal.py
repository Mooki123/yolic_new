"""research/07 quick checks on the outdoor grouped split with the existing B1 checkpoints (no retraining).

A: re-estimate BatchNorm statistics per test session (session-wide, or online in temporal order).
B: temporal smoothing of cell probabilities within a session (causal EMA chosen on val, or a centred window).

    python tools/check_adapt_temporal.py
"""
import json
import os
import sys

import numpy as np
import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from yolic_bench import heads  # noqa: E402
from yolic_bench import metrics as M  # noqa: E402
from yolic_bench.data import CellCache, GpuAugment  # noqa: E402
from yolic_bench.layout import Layout  # noqa: E402

RUN, SEEDS, OBJ, BATCH = 'outdoor_b1_grouped', (0, 1, 2), list(range(11)), 16
ALPHAS = (0.3, 0.5, 0.7, 0.9)
DEV = 'cuda'


def ordered(cache, split_file):
    """Split indices sorted by (session, frame counter), plus each frame's session and counter."""
    sessions = open(os.path.join(cache.path, 'sessions.txt')).read().split()
    idx = cache.indices(split_file)
    key = [(sessions[i], int(os.path.splitext(cache.names[i])[0][-8:])) for i in idx]
    order = sorted(range(len(idx)), key=lambda k: key[k])
    return idx[order], [key[k][0] for k in order], np.array([key[k][1] for k in order])


def bns(model):
    return [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]


def reset_bn(model):
    for m in bns(model):
        m.reset_running_stats()
        m.momentum = None  # cumulative average over the batches seen since the reset


@torch.no_grad()
def forward(model, cache, rows, aug):
    x = aug.to_input(torch.from_numpy(cache.images[np.sort(rows)][np.argsort(np.argsort(rows))]).to(DEV))
    return torch.sigmoid(model(x).float()).cpu().numpy()


@torch.no_grad()
def predict(model, state, cache, idx, sess, aug, mode):
    """mode: 'base' (training BN statistics), 'session' (per-session AdaBN), 'online' (causal per-session BN)."""
    out = np.zeros((len(idx),) + cache.labels.shape[1:], dtype=np.float32)
    bounds = np.flatnonzero(np.r_[True, np.array(sess[1:]) != np.array(sess[:-1]), True])
    model.load_state_dict(state)
    for a, b in zip(bounds[:-1], bounds[1:]):
        if mode != 'base':
            model.load_state_dict(state)
            reset_bn(model)
        if mode == 'session':
            model.train()
            for s in range(a, b, 64):
                forward(model, cache, idx[s:min(s + 64, b)], aug)
        for s in range(a, b, BATCH):
            e = min(s + BATCH, b)
            if mode == 'online':
                model.train()
                forward(model, cache, idx[s:e], aug)  # update the cumulative statistics with this batch
            model.eval()
            out[s:e] = forward(model, cache, idx[s:e], aug)
    return out


def ema(probs, sess, counter, a):
    out = probs.copy()
    for t in range(1, len(probs)):
        same = sess[t] == sess[t - 1]
        if same:
            step = counter[t] - counter[t - 1]
            same = step <= 10 * MEDIAN_STEP[sess[t]]
        if same:
            out[t] = a * probs[t] + (1 - a) * out[t - 1]
    return out


def window(probs, sess):
    out = probs.copy()
    s = np.array(sess)
    for t in range(len(probs)):
        nb = [u for u in (t - 1, t, t + 1) if 0 <= u < len(probs) and s[u] == s[t]]
        out[t] = probs[nb].mean(0)
    return out


def f1(gt, p):
    return 100 * float(M.macro_f1(M.counts(gt, p > 0.5), OBJ))


def main():
    global MEDIAN_STEP
    cfg = yaml.safe_load(open(os.path.join(ROOT, 'configs', RUN + '.yaml')))
    cache = CellCache(os.path.expanduser(cfg['cache']))
    layout = Layout.load(os.path.join(ROOT, cfg['layout']))
    split = os.path.join(ROOT, cfg['split'])
    va, va_s, va_c = ordered(cache, os.path.join(split, 'val.txt'))
    te, te_s, te_c = ordered(cache, os.path.join(split, 'test.txt'))
    MEDIAN_STEP = {}
    for s, c in ((va_s, va_c), (te_s, te_c)):
        for name in set(s):
            steps = np.diff(c[np.array(s) == name])
            MEDIAN_STEP[name] = float(np.median(steps)) if len(steps) else 1.0
    gt_va, gt_te = cache.labels[va], cache.labels[te]
    aug = GpuAugment(None, 'none', None, True)
    model = heads.build(cfg['model'], layout, 12, pretrained=False).to(DEV)

    P = {}  # (variant, split) -> (S, N, C, K) probabilities
    for seed in SEEDS:
        state = torch.load(os.path.join(ROOT, 'runs', RUN, f'seed{seed}', 'best.pt'), map_location=DEV)
        for mode in ('base', 'session', 'online'):
            for name, idx, sess in (('val', va, va_s), ('test', te, te_s)):
                P.setdefault((mode, name), []).append(predict(model, state, cache, idx, sess, aug, mode))
        print(f'seed {seed} done', flush=True)
    P = {k: np.stack(v) for k, v in P.items()}

    def pick_alpha(mode):
        return max(ALPHAS, key=lambda a: np.mean([f1(gt_va, ema(p, va_s, va_c, a)) for p in P[(mode, 'val')]]))

    res = {}
    variants = {'B1 (as trained)': P[('base', 'test')], 'A-session (upper bound)': P[('session', 'test')],
                'A-online': P[('online', 'test')]}
    a_base, a_on = pick_alpha('base'), pick_alpha('online')
    variants[f'B-causal (a={a_base} from val)'] = np.stack([ema(p, te_s, te_c, a_base) for p in P[('base', 'test')]])
    variants['B-window (upper bound)'] = np.stack([window(p, te_s) for p in P[('base', 'test')]])
    variants[f'A-online + B-causal (a={a_on} from val)'] = np.stack([ema(p, te_s, te_c, a_on) for p in P[('online', 'test')]])
    base = variants['B1 (as trained)'] > 0.5
    for k, v in variants.items():
        seeds = [round(f1(gt_te, p), 2) for p in v]
        row = {'seeds': seeds, 'mean': round(float(np.mean(seeds)), 2)}
        if k != 'B1 (as trained)':
            d, lo, hi = M.paired_bootstrap(gt_te, base, v > 0.5, OBJ, n_boot=1000, seed=0)
            row['gain_vs_B1'] = [round(100 * d, 2), round(100 * lo, 2), round(100 * hi, 2)]
        res[k] = row
    res['val_check'] = {'B1 val': round(float(np.mean([f1(gt_va, p) for p in P[('base', 'val')]])), 2),
                        'A-online val': round(float(np.mean([f1(gt_va, p) for p in P[('online', 'val')]])), 2)}
    res['test_sessions'], res['test_frames'] = len(set(te_s)), int(len(te))
    print(json.dumps(res, indent=1))
    json.dump(res, open(os.path.join(ROOT, 'runs', 'logs', 'check_adapt_temporal.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
