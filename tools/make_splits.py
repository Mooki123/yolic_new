"""Freeze the Outdoor Hazard splits (plan 04, section 2.1) and write the leakage evidence.

(a) splits/outdoor_random: the paper's two train_test_split(random_state=2) calls, applied to the
    SORTED file list (the released script uses unsorted os.listdir, outdoor_yolic.py:131-135).
(b) splits/outdoor_grouped: grouped 70/10/20 by recording session. The file names encode the session
    (YYMMDD_HHMMSS + an 8-digit frame counter), so the session is the group and pHash is the check.
    One session holds 39% of the frames, so a plain GroupShuffleSplit (which splits by group count)
    cannot land near 70/10/20 by frames. Instead the sessions are shuffled with numpy seed 0 and filled
    greedily: a session goes to test if it fits under 20% of the frames, else to val if it fits under
    10%, else to train.

splits/outdoor_leakage.json: for each split, the Hamming distance from every test frame's 64-bit pHash
to its nearest train frame (histogram + share within 0 / 4 / 6 / 8 / 12 bits), plus the number of
cross-session near-duplicate pairs (pHash Hamming <= 8) that the session grouping cannot see.

Run:  python tools/make_splits.py --cache ~/data/yolic_cache/outdoor
"""
import argparse
import json
import os

import numpy as np
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
THRESHOLDS = (0, 4, 6, 8, 12)


def random_split(names):
    train, rest = train_test_split(sorted(names), test_size=0.3, random_state=2)
    val, test = train_test_split(rest, test_size=0.6666, random_state=2)
    return {'train': train, 'val': val, 'test': test}


def grouped_split(names, groups, seed=0, fractions=(0.7, 0.1, 0.2)):
    sizes = {}
    for g in groups:
        sizes[g] = sizes.get(g, 0) + 1
    order = sorted(sizes)
    np.random.default_rng(seed).shuffle(order)
    n = len(names)
    cap = {'test': fractions[2] * n, 'val': fractions[1] * n}
    fill = {'test': 0, 'val': 0}
    where = {}
    for g in order:
        for part in ('test', 'val'):
            if fill[part] + sizes[g] <= cap[part]:
                fill[part] += sizes[g]
                where[g] = part
                break
        else:
            where[g] = 'train'
    out = {'train': [], 'val': [], 'test': []}
    for name, g in zip(names, groups):
        out[where[g]].append(name)
    return out


def nn_hamming(query, ref, chunk=512):
    """Nearest-neighbour Hamming distance from each query hash to the reference set."""
    best = np.full(len(query), 64, dtype=np.int64)
    for i in range(0, len(query), chunk):
        x = np.bitwise_xor(query[i:i + chunk, None], ref[None, :])
        best[i:i + chunk] = np.bitwise_count(x).min(axis=1)
    return best


def cross_session_pairs(hashes, groups, thr=8, chunk=512):
    groups = np.asarray(groups)
    pairs = 0
    for i in range(0, len(hashes), chunk):
        d = np.bitwise_count(np.bitwise_xor(hashes[i:i + chunk, None], hashes[None, :]))
        diff = groups[i:i + chunk, None] != groups[None, :]
        pairs += int(((d <= thr) & diff).sum())
    return pairs // 2


def leakage(split, index, hashes):
    tr = hashes[[index[x] for x in split['train']]]
    te = hashes[[index[x] for x in split['test']]]
    d = nn_hamming(te, tr)
    return {
        'n_test': int(len(te)),
        'median_nn_hamming': float(np.median(d)),
        'share_within': {str(t): float((d <= t).mean()) for t in THRESHOLDS},
        'histogram_0_to_32': np.bincount(np.minimum(d, 32), minlength=33).tolist(),
    }


def write(split, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for part, items in split.items():
        with open(os.path.join(out_dir, part + '.txt'), 'w') as f:
            f.write('\n'.join(sorted(items)) + '\n')


def cityscapes_val_cities(root, fraction=0.15, seed=0):
    """Held-out train cities for val' (plan 04, section 2.1): cities shuffled with numpy seed 0, taken
    whole until they hold >= 15% of the train frames. The official val set stays the test set."""
    train = os.path.join(root, 'leftImg8bit', 'train')
    sizes = {c: len(os.listdir(os.path.join(train, c))) for c in sorted(os.listdir(train))}
    order = sorted(sizes)
    np.random.default_rng(seed).shuffle(order)
    total, picked, n = sum(sizes.values()), [], 0
    for c in order:
        if n >= fraction * total:
            break
        picked.append(c)
        n += sizes[c]
    return sorted(picked), n, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default=os.path.expanduser('~/data/yolic_cache/outdoor'))
    ap.add_argument('--out', default=os.path.join(ROOT, 'splits'))
    ap.add_argument('--cityscapes', help='Cityscapes root: only write splits/cityscapes_val_cities.txt')
    args = ap.parse_args()

    if args.cityscapes:
        cities, n, total = cityscapes_val_cities(os.path.expanduser(args.cityscapes))
        path = os.path.join(args.out, 'cityscapes_val_cities.txt')
        if os.path.exists(path) and open(path).read().split() != cities:
            raise SystemExit(f'{path} already exists with a different list; it is frozen, not overwriting')
        with open(path, 'w') as f:
            f.write('\n'.join(cities) + '\n')
        print(f"val' cities {cities}: {n}/{total} train frames ({n / total:.1%})")
        return

    names = open(os.path.join(args.cache, 'names.txt')).read().split()
    groups = open(os.path.join(args.cache, 'sessions.txt')).read().split()
    hashes = np.load(os.path.join(args.cache, 'phash.npy'))
    index = {x: i for i, x in enumerate(names)}

    splits = {'outdoor_random': random_split(names), 'outdoor_grouped': grouped_split(names, groups)}
    report = {'n_frames': len(names), 'n_sessions': len(set(groups)),
              'cross_session_pairs_hamming_le_8': cross_session_pairs(hashes, groups)}
    session_of = dict(zip(names, groups))
    for key, split in splits.items():
        write(split, os.path.join(args.out, key))
        sess = {p: {session_of[x] for x in v} for p, v in split.items()}
        report[key] = {
            'frames': {p: len(v) for p, v in split.items()},
            'fractions': {p: round(len(v) / len(names), 4) for p, v in split.items()},
            'sessions': {p: len(v) for p, v in sess.items()},
            'sessions_shared_train_test': len(sess['train'] & sess['test']),
            'test_to_train': leakage(split, index, hashes),
        }
    with open(os.path.join(args.out, 'outdoor_leakage.json'), 'w') as f:
        json.dump(report, f, indent=1)
    print(json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items()
                                                             if kk != 'test_to_train'} | {
        'within': v.get('test_to_train', {}).get('share_within'),
        'median': v.get('test_to_train', {}).get('median_nn_hamming')}) for k, v in report.items()}, indent=1))


if __name__ == '__main__':
    main()
