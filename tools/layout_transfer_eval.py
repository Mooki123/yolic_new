"""research/08: score every transfer method on every target layout and apply the go/no-go rule.

Methods: overlap (YOLIC A, max over overlapping L1 cells), dense mean (CellPool-s16 D, re-wrapped as a MIL mean head with
its per-class mean bias), MIL max / LSE (trained on L1), the LR-ASPP segmenter (ceiling with masks), and the oracle
retrained on each layout. People+Vehicle macro-F1 at 0.5, void cells masked, seed 0.

    python tools/layout_transfer_eval.py
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
from yolic_bench.data import CellCache, GpuAugment, batches  # noqa: E402
from yolic_bench.layout import Layout  # noqa: E402

DEV, PV, SEED = 'cuda', [0, 1], 0
L1 = 'cityscapes'
TARGETS = ['cityscapes', 'cityscapes_union', 'cityscapes_shift', 'cityscapes_fine']
DENSE = {'mean (D, CellPool-s16)': 'cityscapes_b5_cellpool_s16', 'max (MIL)': 'cityscapes_mil_max_s16',
         'lse (MIL)': 'cityscapes_mil_lse_s16'}


def layout(name):
    return Layout.load(os.path.join(ROOT, 'layouts', name + '.json'))


def run_dir(name):
    return os.path.join(ROOT, 'runs', name, f'seed{SEED}')


def load_model(name, lay1):
    cfg = yaml.safe_load(open(os.path.join(run_dir(name), 'config.yaml')))
    m = heads.build(cfg['model'], lay1, 4, pretrained=False).to(DEV)
    m.load_state_dict(torch.load(os.path.join(run_dir(name), 'best.pt'), map_location=DEV))
    if cfg['model']['head'] == 'cellpool':  # re-wrap D as a layout-agnostic mean-pooling MIL head
        mil = heads.MilPool(1280, lay1, 4, pool='mean').to(DEV)
        mil.conv.load_state_dict(m.head.conv.state_dict())
        mil.bias.data = m.head.bias.data.mean(0)  # the per-(cell, class) bias cannot transfer: use its class mean
        m.head = mil
    return m.eval()


@torch.no_grad()
def dense_probs(model, cache, idx, lays):
    """{layout: (N, n_cells, 4)} cell probabilities, the backbone run once per batch."""
    aug = GpuAugment(None, 'none', None, True)
    out = {n: np.zeros((len(idx), len(l), 4), np.float32) for n, l in lays.items()}
    for imgs, _, _, _, pos in batches(cache, idx, 64, shuffle=False):
        x = aug.to_input(imgs[0].to(DEV))
        with torch.autocast('cuda', dtype=torch.float16):
            f = model.features(x)
        for n, l in lays.items():
            out[n][pos] = torch.sigmoid(model.head.set_layout(l)(f.float())).cpu().numpy()
    return out


@torch.no_grad()
def seg_probs(cache, idx, lays):
    name = 'cityscapes_b8_lraspp_s'
    cfg = yaml.safe_load(open(os.path.join(run_dir(name), 'config.yaml')))
    m = heads.build(cfg['model'], layout(L1), 4, pretrained=False).to(DEV)
    m.load_state_dict(torch.load(os.path.join(run_dir(name), 'best.pt'), map_location=DEV))
    m.eval()
    aug = GpuAugment(None, 'none', None, True)
    out = {n: np.zeros((len(idx), len(l), 4), np.float32) for n, l in lays.items()}
    maps = {n: torch.as_tensor(l.index_map(224, 224).astype(np.int64)).flatten().to(DEV) for n, l in lays.items()}
    for imgs, _, _, _, pos in batches(cache, idx, 64, shuffle=False):
        with torch.autocast('cuda', dtype=torch.float16):
            logits = m(aug.to_input(imgs[0].to(DEV)))
        for n, im in maps.items():
            m.pix_cell, m.pix_in, m.n_cells = im.clamp_min(0), im >= 0, len(lays[n])
            out[n][pos] = m.cell_probs(logits).cpu().numpy()
    return out


def overlap_probs(p1, lay1, lay2):
    """Max of the L1 cells each L2 cell overlaps (by at least one full-resolution pixel)."""
    a, b = lay1.index_map(), lay2.index_map()
    both = (a >= 0) & (b >= 0)
    pairs = np.unique(np.stack([b[both], a[both]], 1), axis=0)
    out = np.zeros((p1.shape[0], len(lay2), 4), np.float32)
    for j in range(len(lay2)):
        src = pairs[pairs[:, 0] == j, 1]
        if len(src):
            out[:, j, :3] = p1[:, src, :3].max(1)
    out[:, :, 3] = 1 - out[:, :, :3].max(2)
    return out


def f1(gt, p, valid):
    return 100 * float(M.macro_f1(M.counts(gt, p > 0.5, valid), PV))


def main():
    cfg = yaml.safe_load(open(os.path.join(ROOT, 'configs', 'cityscapes_b1.yaml')))
    lays = {n: layout(n) for n in TARGETS}
    caches = {n: CellCache(os.path.expanduser(cfg['cache']), labels=None if n == L1 else n) for n in TARGETS}
    split = os.path.join(ROOT, cfg['split'])
    idx = {s: caches[L1].indices(os.path.join(split, f'{s}.txt')) for s in ('val', 'test')}
    gt = {(n, s): (caches[n].labels[idx[s]], caches[n].valid[idx[s]]) for n in TARGETS for s in idx}

    P = {}  # (method, layout, split) -> probs
    a_dir = run_dir('cityscapes_b1')
    a = {'val': np.load(os.path.join(a_dir, 'val_probs.npy')).astype(np.float32),
         'test': np.load(os.path.join(a_dir, 'test_probs.npy')).astype(np.float32)}
    for n in TARGETS:
        for s in idx:
            P[('overlap (A)', n, s)] = overlap_probs(a[s], lays[L1], lays[n])
            o = 'cityscapes_b1' if n == L1 else f'cityscapes_oracle_{n.split("_")[1]}'
            P[('oracle', n, s)] = np.load(os.path.join(run_dir(o), f'{s}_probs.npy')).astype(np.float32)
    for label, name in DENSE.items():
        model = load_model(name, lays[L1])
        for s in idx:
            for n, p in dense_probs(model, caches[L1], idx[s], lays).items():
                P[(label, n, s)] = p
        print('scored', label, flush=True)
    for s in idx:
        for n, p in seg_probs(caches[L1], idx[s], lays).items():
            P[('segmenter (B8, masks)', n, s)] = p

    table = {n: {} for n in TARGETS}
    for (method, n, s), p in P.items():
        table[n].setdefault(method, {})[s] = round(f1(*((gt[(n, s)][0], p, gt[(n, s)][1]))), 2)
    sel = max(DENSE, key=lambda k: np.mean([table[n][k]['val'] for n in ('cityscapes_shift', 'cityscapes_fine')]))

    checks = {}
    for n in ('cityscapes_shift', 'cityscapes_fine'):
        g, v = gt[(n, 'test')]
        d, lo, hi = M.paired_bootstrap(g, P[('overlap (A)', n, 'test')] > 0.5, P[(sel, n, 'test')] > 0.5, PV,
                                       n_boot=1000, seed=0, valid=v)
        t, o = table[n][sel]['test'], table[n]['oracle']['test']
        checks[n] = {'T': t, 'oracle': o, 'T/oracle': round(t / o, 3), 'T-overlap': round(100 * d, 2),
                     'ci95': [round(100 * lo, 2), round(100 * hi, 2)],
                     'pass': bool(t >= 0.9 * o and 100 * d >= 3.0 and lo > 0)}
    passes = sum(c['pass'] for c in checks.values())
    print(json.dumps({'P+V macro-F1 (val\' / test)': table, 'selected dense pooling (on val\')': sel,
                      'checks': checks, 'verdict': {2: 'GO', 1: 'PARTIAL', 0: 'NO-GO'}[passes]}, indent=1))


if __name__ == '__main__':
    main()
