"""Add extra image views to the Cityscapes cache (research/05_fovea_gonogo.md).

Each view is a crop of the 2048x1024 frame (of each augmentation state) resized to its own size. Rows are written
in exactly the order tools/cache_cityscapes.py uses, so labels.npy / valid.npy / aug.npy apply unchanged; a
spot check recomputes the labels of some rows and compares them with labels.npy.

Run:  python tools/cache_views.py --root ~/data/cityscapes --cache ~/data/yolic_cache/cityscapes
Writes images_<view>.npy (rows, h, w, 3) uint8 per view.
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

# name: (crop box in the 2048x1024 frame, output width, output height)
VIEWS = {
    'g224x112': ((0, 0, 2048, 1024), 224, 112),
    'fov320x96': ((512, 320, 1536, 640), 320, 96),
    'r336x168': ((0, 0, 2048, 1024), 336, 168),
    # research/06 stage 1: budget split between the context view and the fovea
    'g160x80': ((0, 0, 2048, 1024), 160, 80),
    'g128x64': ((0, 0, 2048, 1024), 128, 64),
    'g96x48': ((0, 0, 2048, 1024), 96, 48),
    'fov352x112': ((512, 320, 1536, 640), 352, 112),
    'fov384x120': ((512, 320, 1536, 640), 384, 120),
    'fov416x128': ((512, 320, 1536, 640), 416, 128),
    'near256x48': ((0, 640, 2048, 1024), 256, 48),  # the large-cell band (two-band control V5)
}


def render(frame, views):
    return [np.asarray(frame.crop(box).resize((w, h), Image.BILINEAR), dtype=np.uint8) for box, w, h in views]


def process(job):
    img_path, lab_path, augment, views, check = job
    img = Image.open(img_path).convert('RGB')
    states = [img]
    ids_states = [np.asarray(Image.open(lab_path), dtype=np.uint8)] if check else None
    if augment:
        big = img.resize(C.BIG, Image.LANCZOS)
        states += [big.crop((x, y, x + 2048, y + 1024)) for x, y in C.OFFSETS]
        if check:
            big_ids = np.asarray(Image.open(lab_path).resize(C.BIG, Image.NEAREST), dtype=np.uint8)
            ids_states += [big_ids[y:y + 1024, x:x + 2048] for x, y in C.OFFSETS]
    out = [render(s, views) for s in states]
    return out, ids_states


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=os.path.expanduser('~/data/cityscapes'))
    ap.add_argument('--cache', default=os.path.expanduser('~/data/yolic_cache/cityscapes'))
    ap.add_argument('--views', nargs='+', default=list(VIEWS))
    ap.add_argument('--workers', type=int, default=6)
    args = ap.parse_args()

    meta = json.load(open(os.path.join(args.cache, 'meta.json')))
    names = open(os.path.join(args.cache, 'names.txt')).read().split()
    labels = np.load(os.path.join(args.cache, 'labels.npy'))
    aug = np.load(os.path.join(args.cache, 'aug.npy'))
    val_cities = set(open(os.path.join(ROOT, 'splits', 'cityscapes_val_cities.txt')).read().split())
    all_train = C.listing(args.root, 'train')
    parts = {'train': [x for x in all_train if x[0].split('/')[1] not in val_cities],
             'val': [x for x in all_train if x[0].split('/')[1] in val_cities],
             'test': C.listing(args.root, 'val')}
    items = [(p, *x) for p, xs in parts.items() for x in xs]
    assert [x[1] for x in items] == names and len(items) == meta['n'], 'item order differs from the cache'

    layout = Layout.load(os.path.join(ROOT, 'layouts', 'cityscapes.json'))
    index_map = layout.index_map()
    areas = np.bincount(index_map[index_map >= 0], minlength=len(layout)).astype(np.float64)
    specs = [VIEWS[v] for v in args.views]
    arrays = [np.lib.format.open_memmap(os.path.join(args.cache, f'images_{v}.npy'), 'w+', np.uint8,
                                        (meta['rows'], h, w, 3)) for v, (_, w, h) in zip(args.views, specs)]
    jobs = [(img, lab, part == 'train', specs, i % 97 == 0) for i, (part, _, img, lab) in enumerate(items)]
    checked = 0
    with ProcessPoolExecutor(args.workers) as ex:
        for i, (states, ids_states) in enumerate(ex.map(process, jobs, chunksize=8)):
            rows = [i] + ([int(r) for r in aug[i]] if len(states) > 1 else [])
            assert len(rows) == len(states)
            for row, views in zip(rows, states):
                for arr, im in zip(arrays, views):
                    arr[row] = im
            if ids_states is not None:  # row alignment check against the stored labels
                for row, ids in zip(rows, ids_states):
                    lab, _ = C.v1_labels(C.cell_counts(C.V1_GROUP[ids], index_map, len(layout)), areas)
                    assert np.array_equal(lab, labels[row]), f'label mismatch at row {row}'
                    checked += 1
            if i % 250 == 0:
                print(f'{i}/{len(items)}', flush=True)
    for arr in arrays:
        arr.flush()
    meta.setdefault('views', {}).update({v: {'crop': list(b), 'size': [w, h]} for v, (b, w, h) in zip(args.views, specs)})
    json.dump(meta, open(os.path.join(args.cache, 'meta.json'), 'w'), indent=1)
    print(f'done; {checked} rows label-checked against labels.npy')


if __name__ == '__main__':
    main()
