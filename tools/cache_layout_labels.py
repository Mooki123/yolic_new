"""Cell labels for another layout on every row of the Cityscapes cache (research/08 layout transfer).

Uses the full-resolution masks and the v1 rule of tools/cache_cityscapes.py, row order of that cache (rows 0..N-1 plain,
then each train image's 8 rescale-crop states at aug.npy's rows). Writes labels_<name>.npy and valid_<name>.npy.
A spot check recomputes the cache's own layout on some rows and compares with labels.npy.

    python tools/cache_layout_labels.py --layouts cityscapes_union cityscapes_shift cityscapes_fine
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tools'))
import cache_cityscapes as C  # noqa: E402
from yolic_bench.layout import Layout  # noqa: E402

CTX = {}


def init(maps):
    CTX['maps'] = maps  # name -> (index_map, areas, n_cells)


def process(job):
    lab_path, augment = job
    ids = Image.open(lab_path)
    states = [np.asarray(ids, dtype=np.uint8)]
    if augment:
        big = np.asarray(ids.resize(C.BIG, Image.NEAREST), dtype=np.uint8)
        states += [big[y:y + 1024, x:x + 2048] for x, y in C.OFFSETS]
    out = {}
    for name, (index_map, areas, n) in CTX['maps'].items():
        labs, valids = [], []
        for s in states:
            lab, valid = C.v1_labels(C.cell_counts(C.V1_GROUP[s], index_map, n), areas)
            labs.append(lab)
            valids.append(valid)
        out[name] = (np.stack(labs), np.stack(valids))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=os.path.expanduser('~/data/cityscapes'))
    ap.add_argument('--cache', default=os.path.expanduser('~/data/yolic_cache/cityscapes'))
    ap.add_argument('--layouts', nargs='+', required=True)
    ap.add_argument('--workers', type=int, default=6)
    args = ap.parse_args()

    meta = json.load(open(os.path.join(args.cache, 'meta.json')))
    names = open(os.path.join(args.cache, 'names.txt')).read().split()
    aug = np.load(os.path.join(args.cache, 'aug.npy'))
    val_cities = set(open(os.path.join(ROOT, 'splits', 'cityscapes_val_cities.txt')).read().split())
    all_train = C.listing(args.root, 'train')
    parts = {'train': [x for x in all_train if x[0].split('/')[1] not in val_cities],
             'val': [x for x in all_train if x[0].split('/')[1] in val_cities],
             'test': C.listing(args.root, 'val')}
    items = [(p, *x) for p, xs in parts.items() for x in xs]
    assert [x[1] for x in items] == names, 'item order differs from the cache'

    maps = {}
    for name in args.layouts + ['cityscapes']:  # 'cityscapes' = the cache's own layout, for the alignment check
        lay = Layout.load(os.path.join(ROOT, 'layouts', name + '.json'))
        im = lay.index_map()
        maps[name] = (im, np.bincount(im[im >= 0], minlength=len(lay)).astype(np.float64), len(lay))
    out = {n: (np.zeros((meta['rows'], maps[n][2], 4), np.uint8), np.zeros((meta['rows'], maps[n][2]), bool))
           for n in maps}
    ref = np.load(os.path.join(args.cache, 'labels.npy'))
    jobs = [(lab, part == 'train') for part, _, _, lab in items]
    with ProcessPoolExecutor(args.workers, initializer=init, initargs=(maps,)) as ex:
        for i, res in enumerate(ex.map(process, jobs, chunksize=8)):
            rows = [i] + ([int(r) for r in aug[i]] if aug[i, 0] >= 0 else [])
            for n, (lab, valid) in res.items():
                out[n][0][rows], out[n][1][rows] = lab, valid
            if i % 500 == 0:
                print(f'{i}/{len(items)}', flush=True)
    assert np.array_equal(out['cityscapes'][0], ref), 'row alignment check failed against labels.npy'
    for n in args.layouts:
        np.save(os.path.join(args.cache, f'labels_{n}.npy'), out[n][0])
        np.save(os.path.join(args.cache, f'valid_{n}.npy'), out[n][1])
        print(n, 'cells', maps[n][2], 'invalid share', round(float(1 - out[n][1][:len(items)].mean()), 4),
              'positives P/V/O/R', out[n][0][:len(items)][out[n][1][:len(items)]].sum(0).tolist())
    print('done; alignment check passed on all', meta['rows'], 'rows')


if __name__ == '__main__':
    main()
