"""research/10 part 1: diagnose the cross-session gap on the outdoor grouped split (existing predictions only).

    python tools/diagnose_sessions.py
"""
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from yolic_bench import metrics as M  # noqa: E402
from yolic_bench.data import CellCache  # noqa: E402

OBJ = list(range(11))
CACHE = os.path.expanduser('~/data/yolic_cache/outdoor')


def probs(run, seed, part):
    return np.load(os.path.join(ROOT, 'runs', run, f'seed{seed}', f'{part}_probs.npy')).astype(np.float32)


def main():
    c = CellCache(CACHE)
    names = np.array(c.names)
    sessions = np.array(open(os.path.join(CACHE, 'sessions.txt')).read().split())
    hashes = np.load(os.path.join(CACHE, 'phash.npy'))
    classes = json.load(open(os.path.join(CACHE, 'meta.json')))['classes'][:11]
    tr, te = (c.indices(os.path.join(ROOT, 'splits', 'outdoor_grouped', f'{p}.txt')) for p in ('train', 'test'))
    gt = c.labels[te]
    pred = np.stack([probs('outdoor_b1_grouped', s, 'test') > 0.5 for s in range(3)])
    out = {}

    # 1. per-session F1 (pooled over seeds), classes present in the session
    rows = []
    for s in np.unique(sessions[te]):
        m = sessions[te] == s
        cnt = M.counts(np.concatenate([gt[m]] * 3), np.concatenate(list(pred[:, m])))
        present = [k for k in OBJ if cnt['tp'][k] + cnt['fn'][k] > 0]
        if present:
            rows.append((s, int(m.sum()), float(M.macro_f1(cnt, present)), len(present)))
    f1s = np.array([r[2] for r in rows])
    w = np.array([r[1] for r in rows])
    out['per_session'] = {'n_sessions': len(rows), 'F1 quartiles': np.percentile(100 * f1s, [0, 25, 50, 75, 100]).round(1).tolist(),
                          'frame-weighted mean': round(float(100 * np.average(f1s, weights=w)), 1),
                          'share of test frames in sessions with F1 < 40': round(float(w[f1s < 0.4].sum() / w.sum()), 3)}

    # 2. class shift and per-class F1, random vs grouped
    rate = lambda idx: c.labels[idx][..., :11].sum((0, 1)) / len(idx)
    r_tr, r_te = rate(tr), rate(te)
    ce = M.prf(M.counts(np.concatenate([gt] * 3), np.concatenate(list(pred))))[2]
    rte = c.indices(os.path.join(ROOT, 'splits', 'outdoor_random', 'test.txt'))
    cr = M.prf(M.counts(c.labels[rte], probs('outdoor_b1_random', 0, 'test') > 0.5))[2]
    out['per_class'] = {k: {'pos/frame train': round(float(r_tr[i]), 2), 'pos/frame test': round(float(r_te[i]), 2),
                            'F1 random split': round(float(100 * cr[i]), 1), 'F1 grouped': round(float(100 * ce[i]), 1)}
                        for i, k in enumerate(classes)}

    # 3. visual distance to training vs session F1
    tr_h = hashes[tr]
    dist = {}
    for s, n, f, _ in rows:
        q = hashes[te[sessions[te] == s]]
        d = np.bitwise_count(np.bitwise_xor(q[:, None], tr_h[None, :])).min(1)
        dist[s] = float(np.median(d))
    rho, p = spearmanr([dist[r[0]] for r in rows], f1s)
    out['distance_vs_F1'] = {'spearman_rho': round(float(rho), 3), 'p': round(float(p), 4),
                             'median NN distance quartiles': np.percentile(list(dist.values()), [0, 25, 50, 75, 100]).tolist()}

    # 4. label agreement of near-identical frames, same vs different session (all 20k frames)
    lab = c.labels[..., :11].astype(bool)
    agree = {'same': [], 'diff': []}
    for i in range(0, len(hashes), 1024):
        d = np.bitwise_count(np.bitwise_xor(hashes[i:i + 1024, None], hashes[None, :]))
        a, b = np.nonzero(d <= 4)
        a = a + i
        keep = a < b
        for x, y in zip(a[keep], b[keep]):
            hz = lab[x].any(1) | lab[y].any(1)  # cells where either frame has a hazard
            if hz.any():
                agree['same' if sessions[x] == sessions[y] else 'diff'].append(float((lab[x][hz] == lab[y][hz]).all(1).mean()))
    out['near_duplicate_label_agreement'] = {k: {'pairs': len(v), 'mean cell agreement': round(float(np.mean(v)), 3) if v else None}
                                             for k, v in agree.items()}
    print(json.dumps(out, indent=1))
    json.dump(out, open(os.path.join(ROOT, 'runs', 'logs', 'diagnose_sessions.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
