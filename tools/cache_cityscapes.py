"""Build the Cityscapes cache for arm H (plan 04, section 4.1, row cityscapes.py:149-177).

Rows 0..N-1 hold every image in its plain state (train, val' and the official val = test). For each
train image 8 more rows hold the released rescale-crop augmentation (cityscapes.py:159-171: enlarge to
2198x1099, crop 2048x1024) at 8 FIXED offsets instead of a random one. The x offsets are symmetric
about the middle of the 150 px range, so a horizontal flip (done on the GPU with the layout's mirror
permutation) commutes with the crop set. aug.npy (N, 8) gives those rows, -1 for non-train images.

Cell labels are computed from the full-resolution (2048x1024) mask of each state, as the released
encode_cell does, with the v1 rule:
  * train ids from the released custom table, except polegroup -> pole (5, 'Other') (D11);
  * object group bit = at least min_pixels (1) pixel of the group in the cell (D13 parameter);
  * background (Road) bit = no People / Vehicle / Other pixel in the cell (the released meaning,
    without the index-216 special case, D12);
  * valid = void (train id 255) fraction of the cell <= TAU_VOID (0.5); invalid cells are masked
    out of loss and metric. TAU_VOID was fixed on 2026-10-08 before any Cityscapes run.
labels_released.npy holds the exact released encode_cell labels (polegroup -> Vehicle, 216 rule) for
the plain state, for the reproduction gate and the protocol ladder.

Also stored: 224x224 group masks (0 People, 1 Vehicle, 2 Other, 3 Road, 255 void; nearest) for the
LR-ASPP segmenter baseline (B8).

Run:  python tools/cache_cityscapes.py --root ~/data/cityscapes --out ~/data/yolic_cache/cityscapes
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
from yolic_bench.layout import Layout  # noqa: E402

SIZE = 224
TAU_VOID = 0.5
MIN_PIXELS = 1
OFFSETS = [(x, y) for x in (0, 50, 100, 150) for y in (0, 75)]  # into the 2198x1099 enlargement
BIG = (2198, 1099)
CLASSES = ['People', 'Vehicle', 'Other', 'Road']

# Released custom table (cityscapes.py:classes), label id 0..33 -> train id; license plate (-1) unused.
RELEASED_TRAIN_ID = np.array([255, 255, 255, 255, 255, 25, 24, 0, 1, 23, 22, 2, 3, 4, 21, 20, 19, 5, 18, 6, 7,
                              8, 9, 10, 11, 12, 13, 14, 15, 13, 13, 16, 17, 18], dtype=np.uint8)
V1_TRAIN_ID = RELEASED_TRAIN_ID.copy()
V1_TRAIN_ID[18] = 5  # polegroup -> pole
# cityscapes_yolic.py:interested_classes
GROUPS = [(11, 12), (13, 14, 15, 16, 17, 18), (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 25, 21, 20, 19), (0, 23, 22, 24)]
TRAIN_ID_TO_GROUP = np.full(256, 4, dtype=np.uint8)  # 4 = void
for g, ids in enumerate(GROUPS):
    TRAIN_ID_TO_GROUP[list(ids)] = g


def lut(table):
    """label id (uint8 PNG value, 0..255) -> group, through a train-id table."""
    out = np.full(256, 4, dtype=np.uint8)
    out[:len(table)] = TRAIN_ID_TO_GROUP[table]
    return out


V1_GROUP = lut(V1_TRAIN_ID)


def cell_counts(groups, index_map, n_cells):
    """(n_cells, 5) pixel counts of each group (People, Vehicle, Other, Road, void) per cell."""
    m = index_map >= 0
    flat = index_map[m].astype(np.int64) * 5 + groups[m]
    return np.bincount(flat, minlength=n_cells * 5).reshape(n_cells, 5)


def v1_labels(counts, areas):
    obj = counts[:, :3] >= MIN_PIXELS
    lab = np.concatenate([obj, ~obj.any(axis=1, keepdims=True)], axis=1).astype(np.uint8)
    valid = counts[:, 4] / areas <= TAU_VOID
    return lab, valid


def released_labels(train_ids, cell_list):
    """cityscapes.py:Cityscapes.encode_cell, vectorised per cell but with identical logic."""
    out = np.zeros((len(cell_list), 4), dtype=np.uint8)
    for num, ((x1, y1), (x2, y2)) in enumerate(cell_list):
        value = np.unique(train_ids[y1:y2, x1:x2])
        for j, sub in enumerate(GROUPS[:3]):
            out[num, j] = any(s in value for s in sub)
        if out[num, :3].any():
            out[num, 3] = 0
        elif np.size(value) == 1 and value[0] == 255 and num <= 216:
            out[num, 3] = 0
        else:
            out[num, 3] = 1
    return out


def small_mask(groups):
    m = Image.fromarray(np.where(groups == 4, 255, groups).astype(np.uint8))
    return np.asarray(m.resize((SIZE, SIZE), Image.NEAREST), dtype=np.uint8)


def process(job):
    img_path, lab_path, augment, cell_list = job
    index_map, areas = CTX['index_map'], CTX['areas']
    n = len(cell_list)
    img = Image.open(img_path).convert('RGB')
    ids = Image.open(lab_path)
    ids_np = np.asarray(ids, dtype=np.uint8)
    states = [(img, ids_np)]
    if augment:
        big_img = img.resize(BIG, Image.LANCZOS)
        big_ids = np.asarray(ids.resize(BIG, Image.NEAREST), dtype=np.uint8)
        for x, y in OFFSETS:
            states.append((big_img.crop((x, y, x + 2048, y + 1024)), big_ids[y:y + 1024, x:x + 2048]))
    imgs, masks, labs, valids = [], [], [], []
    for im, idmap in states:
        groups = V1_GROUP[idmap]
        lab, valid = v1_labels(cell_counts(groups, index_map, n), areas)
        imgs.append(np.asarray(im.resize((SIZE, SIZE), Image.BILINEAR), dtype=np.uint8))
        masks.append(small_mask(groups))
        labs.append(lab)
        valids.append(valid)
    rel = released_labels(RELEASED_TRAIN_ID[ids_np], cell_list)
    return np.stack(imgs), np.stack(masks), np.stack(labs), np.stack(valids), rel


CTX = {}


def init(index_map, areas):
    CTX['index_map'], CTX['areas'] = index_map, areas


def listing(root, split, cities=None):
    base = os.path.join(root, 'leftImg8bit', split)
    out = []
    for city in sorted(os.listdir(base)):
        if cities is not None and city not in cities:
            continue
        for f in sorted(os.listdir(os.path.join(base, city))):
            stem = f.split('_leftImg8bit')[0]
            out.append((f'{split}/{city}/{stem}', os.path.join(base, city, f),
                        os.path.join(root, 'gtFine', split, city, stem + '_gtFine_labelIds.png')))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=os.path.expanduser('~/data/cityscapes'))
    ap.add_argument('--out', default=os.path.expanduser('~/data/yolic_cache/cityscapes'))
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--limit', type=int, help='only the first k images per split (smoke test)')
    args = ap.parse_args()

    layout = Layout.load(os.path.join(ROOT, 'layouts', 'cityscapes.json'))
    cell_list = [[[int(c[:, 0].min()), int(c[:, 1].min())], [int(c[:, 0].max()), int(c[:, 1].max())]]
                 for c in layout.cells]
    index_map = layout.index_map()
    areas = np.bincount(index_map[index_map >= 0], minlength=len(layout)).astype(np.float64)

    val_cities = set(open(os.path.join(ROOT, 'splits', 'cityscapes_val_cities.txt')).read().split())
    all_train = listing(args.root, 'train')
    parts = {'train': [x for x in all_train if x[0].split('/')[1] not in val_cities],
             'val': [x for x in all_train if x[0].split('/')[1] in val_cities],
             'test': listing(args.root, 'val')}
    if args.limit:
        parts = {k: v[:args.limit] for k, v in parts.items()}
    items = [(p, *x) for p, xs in parts.items() for x in xs]
    n = len(items)
    n_rows = n + len(OFFSETS) * len(parts['train'])

    os.makedirs(args.out, exist_ok=True)
    images = np.lib.format.open_memmap(os.path.join(args.out, 'images.npy'), 'w+', np.uint8, (n_rows, SIZE, SIZE, 3))
    masks = np.lib.format.open_memmap(os.path.join(args.out, 'masks.npy'), 'w+', np.uint8, (n_rows, SIZE, SIZE))
    labels = np.zeros((n_rows, len(layout), 4), dtype=np.uint8)
    valid = np.zeros((n_rows, len(layout)), dtype=bool)
    released = np.zeros((n, len(layout), 4), dtype=np.uint8)
    aug = np.full((n, len(OFFSETS)), -1, dtype=np.int64)
    jobs = [(img, lab, part == 'train', cell_list) for part, _, img, lab in items]
    nxt = n
    with ProcessPoolExecutor(args.workers, initializer=init, initargs=(index_map, areas)) as ex:
        for i, (im, mk, lb, vd, rel) in enumerate(ex.map(process, jobs, chunksize=8)):
            rows = [i] + list(range(nxt, nxt + len(im) - 1))
            if len(im) > 1:
                aug[i] = rows[1:]
                nxt += len(im) - 1
            images[rows], masks[rows], labels[rows], valid[rows] = im, mk, lb, vd
            released[i] = rel
            if i % 250 == 0:
                print(f'{i}/{n}', flush=True)
    images.flush()
    masks.flush()
    for name, arr in (('labels', labels), ('valid', valid), ('labels_released', released), ('aug', aug)):
        np.save(os.path.join(args.out, name + '.npy'), arr)
    with open(os.path.join(args.out, 'names.txt'), 'w') as f:
        f.write('\n'.join(x[1] for x in items) + '\n')
    split_dir = os.path.join(ROOT, 'splits', 'cityscapes' if not args.limit else 'cityscapes_smoke')
    os.makedirs(split_dir, exist_ok=True)
    for part, xs in parts.items():
        with open(os.path.join(split_dir, part + '.txt'), 'w') as f:
            f.write('\n'.join(x[0] for x in xs) + '\n')
    base = labels[:n]
    meta = dict(n=n, rows=n_rows, size=SIZE, classes=CLASSES, layout='layouts/cityscapes.json', tau_void=TAU_VOID,
                min_pixels=MIN_PIXELS, offsets=OFFSETS, split_sizes={k: len(v) for k, v in parts.items()},
                invalid_cell_share=float(1 - valid[:n].mean()),
                positives_per_class_plain=base[valid[:n]].sum(0).tolist(),
                released_vs_v1_bit_disagreement=float((released != base).mean()))
    with open(os.path.join(args.out, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(meta, indent=1))


if __name__ == '__main__':
    main()
