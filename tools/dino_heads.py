"""research/11 idea 1: train cell heads on frozen DINOv2 features (tools/dino_features.py) and apply the go/no-go.

Heads: 'linear' ([CLS, mean patch] -> Linear) and 'spatial' (8x8 patch grid -> 1x1 conv 64 -> flatten -> Linear).
Flip augmentation uses the cached features of the flipped frame with mirror-permuted labels. Adam 1e-3, BCE, 40 epochs,
best epoch by grouped-val object macro-F1; the head type is chosen on val (mean over 3 seeds), then scored on test.

    python tools/dino_heads.py
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from yolic_bench import metrics as M  # noqa: E402
from yolic_bench.data import CellCache  # noqa: E402
from yolic_bench.layout import Layout  # noqa: E402

CACHE = os.path.expanduser('~/data/yolic_cache/outdoor')
OBJ, DEV, EPOCHS, BATCH = list(range(11)), 'cuda', 40, 256


class Head(nn.Module):
    def __init__(self, kind, n_cells=104, n_bits=12):
        super().__init__()
        self.kind, self.n_cells, self.n_bits = kind, n_cells, n_bits
        if kind == 'linear':
            self.net = nn.Sequential(nn.Dropout(0.2), nn.Linear(768, n_cells * n_bits))
        else:
            self.net = nn.Sequential(nn.Conv2d(384, 64, 1), nn.BatchNorm2d(64), nn.ReLU(), nn.Flatten(),
                                     nn.Dropout(0.2), nn.Linear(64 * 64, n_cells * n_bits))

    def forward(self, cls, grid):
        x = torch.cat([cls, grid.mean((2, 3))], 1) if self.kind == 'linear' else grid
        return self.net(x).view(-1, self.n_cells, self.n_bits)


def f1(gt, p):
    return 100 * float(M.macro_f1(M.counts(gt, p > 0.5), OBJ))


def train(kind, seed, feats, labels, perm, tr, va):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    head = Head(kind).to(DEV)
    opt = torch.optim.Adam(head.parameters(), 1e-3, weight_decay=1e-4)
    best, best_state = -1, None
    cls, grid = feats
    for _ in range(EPOCHS):
        head.train()
        order = torch.randperm(len(tr), generator=g).numpy()
        for s in range(0, len(order), BATCH):
            b = tr[order[s:s + BATCH]]
            flip = (torch.rand(len(b), generator=g) < 0.5).numpy().astype(int)
            c = torch.from_numpy(cls[flip, b].astype(np.float32)).to(DEV)
            x = torch.from_numpy(grid[flip, b].astype(np.float32)).to(DEV)
            y = labels[b].copy()
            y[flip == 1] = y[flip == 1][:, perm]
            loss = nn.functional.binary_cross_entropy_with_logits(head(c, x), torch.from_numpy(y).float().to(DEV))
            opt.zero_grad()
            loss.backward()
            opt.step()
        v = f1(labels[va], predict(head, feats, va))
        if v > best:
            best, best_state = v, {k: t.clone() for k, t in head.state_dict().items()}
    head.load_state_dict(best_state)
    return head, best


@torch.no_grad()
def predict(head, feats, idx):
    head.eval()
    out = []
    for s in range(0, len(idx), 1024):
        b = idx[s:s + 1024]
        c = torch.from_numpy(feats[0][0, b].astype(np.float32)).to(DEV)
        x = torch.from_numpy(feats[1][0, b].astype(np.float32)).to(DEV)
        out.append(torch.sigmoid(head(c, x)).cpu().numpy())
    return np.concatenate(out)


def main():
    c = CellCache(CACHE)
    perm = Layout.load(os.path.join(ROOT, 'layouts', 'outdoor.json')).mirror_permutation()
    tr, va, te = (c.indices(os.path.join(ROOT, 'splits', 'outdoor_grouped', f'{p}.txt')) for p in ('train', 'val', 'test'))
    feats = (np.load(os.path.join(CACHE, 'dino_cls.npy')), np.load(os.path.join(CACHE, 'dino_grid.npy'), mmap_mode='r'))
    feats = (feats[0], np.ascontiguousarray(feats[1]))  # about 2 GB in RAM; random batch access from a memmap is slow
    labels = c.labels
    res, test_preds = {}, {}
    for kind in ('linear', 'spatial'):
        vals, tests, preds = [], [], []
        for seed in range(3):
            head, v = train(kind, seed, feats, labels, perm, tr, va)
            p = predict(head, feats, te)
            vals.append(round(v, 2))
            tests.append(round(f1(labels[te], p), 2))
            preds.append(p > 0.5)
            print(kind, 'seed', seed, 'val', vals[-1], 'test', tests[-1], flush=True)
        res[kind] = {'val': vals, 'val_mean': round(float(np.mean(vals)), 2), 'test': tests,
                     'test_mean': round(float(np.mean(tests)), 2)}
        test_preds[kind] = np.stack(preds)
    pick = max(res, key=lambda k: res[k]['val_mean'])
    base = np.stack([np.load(os.path.join(ROOT, 'runs', 'outdoor_b1_grouped', f'seed{s}', 'test_probs.npy'))
                     .astype(np.float32) > 0.5 for s in range(3)])
    d, lo, hi = M.paired_bootstrap(labels[te], base, test_preds[pick], OBJ, n_boot=1000, seed=0)
    gain = 100 * d
    verdict = 'GO' if gain >= 5.0 and lo > 0 else ('WEAK' if gain >= 2.0 else 'NO')
    res['B1 test'] = [round(f1(labels[te], b.astype(np.float32)), 2) for b in base]
    res['selected head (on val)'] = pick
    res['gain vs B1'] = [round(gain, 2), round(100 * lo, 2), round(100 * hi, 2)]
    res['verdict'] = verdict
    print(json.dumps(res, indent=1))
    json.dump(res, open(os.path.join(ROOT, 'runs', 'logs', 'dino_heads.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
