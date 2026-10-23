"""research/10 part 2: does conformal miss-rate control survive session shift? (existing predictions, no training)

Cell-level hazard FNR: a cell is a hazard if any of the 11 classes is present, its score is the max class probability.
Per image, loss(lambda) = share of its hazard cells scored below lambda. 500 random calibration/test draws over the 68
sessions B1-grouped never trained on, by frame (leaky control) or by whole session (deployment).

    python tools/crc_sessions.py
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from yolic_bench.data import CellCache  # noqa: E402

CACHE = os.path.expanduser('~/data/yolic_cache/outdoor')
GRID = np.linspace(0, 1, 1001)
ALPHAS = (0.05, 0.10, 0.20)
DRAWS = 500


def main():
    c = CellCache(CACHE)
    sessions = np.array(open(os.path.join(CACHE, 'sessions.txt')).read().split())
    run = os.path.join(ROOT, 'runs', 'outdoor_b1_grouped', 'seed0')
    idx, probs = [], []
    for part in ('val', 'test'):
        idx.append(c.indices(os.path.join(ROOT, 'splits', 'outdoor_grouped', f'{part}.txt')))
        probs.append(np.load(os.path.join(run, f'{part}_probs.npy')).astype(np.float32))
    idx, probs = np.concatenate(idx), np.concatenate(probs)
    hazard = c.labels[idx][..., :11].any(-1)  # (N, cells)
    score = probs[..., :11].max(-1)
    sess = sessions[idx]

    # loss[i, g] = share of image i's hazard cells with score < GRID[g]; flagged[i, g] = share of cells flagged
    n_h = hazard.sum(1)
    loss = np.zeros((len(idx), len(GRID)))
    for i in np.flatnonzero(n_h):
        loss[i] = np.searchsorted(np.sort(score[i][hazard[i]]), GRID, side='left') / n_h[i]
    flagged = 1 - np.stack([np.searchsorted(np.sort(s), GRID, side='left') / len(s) for s in score])

    def thresholds(cal):
        risk, n = loss[cal].mean(0), len(cal)
        out = {}
        for a in ALPHAS:
            ok_crc = np.flatnonzero(n / (n + 1) * risk + 1 / (n + 1) <= a)
            ok_naive = np.flatnonzero(risk <= a)
            out[a] = (ok_crc.max() if len(ok_crc) else 0, ok_naive.max() if len(ok_naive) else 0)
        return out

    rng = np.random.default_rng(0)
    uniq = np.unique(sess)
    res = {}
    for scheme in ('frame-level (leaky control)', 'session-level (deployment)'):
        fnr = {a: {'crc': [], 'naive': []} for a in ALPHAS}
        flag = {a: [] for a in ALPHAS}
        for _ in range(DRAWS):
            if scheme.startswith('frame'):
                perm = rng.permutation(len(idx))
                cal, test = perm[:len(idx) // 2], perm[len(idx) // 2:]
            else:
                pick = set(rng.choice(uniq, len(uniq) // 2, replace=False))
                m = np.array([s in pick for s in sess])
                cal, test = np.flatnonzero(m), np.flatnonzero(~m)
            for a, (g_crc, g_naive) in thresholds(cal).items():
                fnr[a]['crc'].append(loss[test, g_crc].mean())
                fnr[a]['naive'].append(loss[test, g_naive].mean())
                flag[a].append(flagged[test, g_crc].mean())
        res[scheme] = {f'alpha={a}': {
            'CRC mean FNR': round(float(np.mean(fnr[a]['crc'])), 4),
            'CRC P(FNR > alpha+0.05)': round(float(np.mean(np.array(fnr[a]['crc']) > a + 0.05)), 3),
            'CRC FNR 5-95%': np.percentile(fnr[a]['crc'], [5, 95]).round(3).tolist(),
            'naive mean FNR': round(float(np.mean(fnr[a]['naive'])), 4),
            'CRC flagged-cell rate': round(float(np.mean(flag[a])), 3)} for a in ALPHAS}
    g05 = np.searchsorted(GRID, 0.5)
    res['fixed 0.5 threshold'] = {'FNR': round(float(loss[:, g05].mean()), 4), 'flagged-cell rate': round(float(flagged[:, g05].mean()), 3)}
    res['data'] = {'images': int(len(idx)), 'images with a hazard': int((n_h > 0).sum()), 'sessions': int(len(uniq))}
    a = 0.10
    s, f = res['session-level (deployment)'][f'alpha={a}'], res['frame-level (leaky control)'][f'alpha={a}']
    material = (s['CRC mean FNR'] >= a + 0.02 or s['CRC P(FNR > alpha+0.05)'] >= 0.20) and f['CRC P(FNR > alpha+0.05)'] <= 0.05
    res['verdict (alpha = 0.10)'] = 'MATERIAL' if material else 'NOT MATERIAL'
    print(json.dumps(res, indent=1))
    json.dump(res, open(os.path.join(ROOT, 'runs', 'logs', 'crc_sessions.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
