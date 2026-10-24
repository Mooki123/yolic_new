"""research/14: Cityscapes person/vehicle boxes from gtFine instanceIds, as a YOLO dataset on our splits.

People = person, rider (label ids 24-25); Vehicle = car, truck, bus, caravan, trailer, train, motorcycle, bicycle
(26-33): the v1 cell-label groups. Pixels >= 1000 encode label_id * 1000 + instance: one box per instance. Pixels
< 1000 equal to one of these label ids are crowd regions: one box per 8-connected component (flagged in boxes.json).

Writes ~/data/yolo_cityscapes/{images,labels}/{train,val,test}/ (images at 1024x512, YOLO txt with class cx cy w h
normalised), data.yaml, and boxes.json (all boxes in 2048x1024 source pixels, per split image, for the box oracle and
for rasterising detections). train = train cities minus val', val = val' cities, test = official val.

    python tools/cityscapes_boxes.py
"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image
from scipy import ndimage

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'tools'))
import cache_cityscapes as C  # noqa: E402

PEOPLE, VEHICLE = (24, 25), (26, 27, 28, 29, 30, 31, 32, 33)
CLASS_OF = {i: 0 for i in PEOPLE} | {i: 1 for i in VEHICLE}
OUT = os.path.expanduser('~/data/yolo_cityscapes')
SIZE = (1024, 512)


def boxes_of(inst):
    """[(cls, x1, y1, x2, y2, crowd)] in source pixels; x2/y2 exclusive."""
    out = []
    for v in np.unique(inst):
        lab = v // 1000 if v >= 1000 else v
        if lab not in CLASS_OF:
            continue
        m = inst == v
        comps = [(m, False)] if v >= 1000 else [(c, True) for c in
                                                 (lambda l, n: [l == k for k in range(1, n + 1)])(*ndimage.label(m, np.ones((3, 3))))]
        for mask, crowd in comps:
            ys, xs = np.nonzero(mask)
            out.append((CLASS_OF[lab], int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1, crowd))
    return out


def process(job):
    split, key, img_path, lab_path = job
    inst = np.asarray(Image.open(lab_path.replace('_labelIds.png', '_instanceIds.png')), dtype=np.int32)
    bx = boxes_of(inst)
    name = key.replace('/', '__')
    Image.open(img_path).convert('RGB').resize(SIZE, Image.BILINEAR).save(
        os.path.join(OUT, 'images', split, name + '.jpg'), quality=95)
    with open(os.path.join(OUT, 'labels', split, name + '.txt'), 'w') as f:
        for c, x1, y1, x2, y2, _ in bx:
            f.write(f'{c} {(x1 + x2) / 4096:.6f} {(y1 + y2) / 2048:.6f} {(x2 - x1) / 2048:.6f} {(y2 - y1) / 1024:.6f}\n')
    return key, bx


def main():
    root = os.path.expanduser('~/data/cityscapes')
    val_cities = set(open(os.path.join(ROOT, 'splits', 'cityscapes_val_cities.txt')).read().split())
    all_train = C.listing(root, 'train')
    parts = {'train': [x for x in all_train if x[0].split('/')[1] not in val_cities],
             'val': [x for x in all_train if x[0].split('/')[1] in val_cities],
             'test': C.listing(root, 'val')}
    for split, items in parts.items():  # same order and membership as splits/cityscapes/*.txt
        assert [x[0] for x in items] == open(os.path.join(ROOT, 'splits', 'cityscapes', f'{split}.txt')).read().split()
        for d in ('images', 'labels'):
            os.makedirs(os.path.join(OUT, d, split), exist_ok=True)
    jobs = [(s, *x) for s, xs in parts.items() for x in xs]
    allb = {}
    with ProcessPoolExecutor(6) as ex:
        for i, (key, bx) in enumerate(ex.map(process, jobs, chunksize=8)):
            allb[key] = bx
            if i % 500 == 0:
                print(f'{i}/{len(jobs)}', flush=True)
    json.dump(allb, open(os.path.join(OUT, 'boxes.json'), 'w'))
    with open(os.path.join(OUT, 'data.yaml'), 'w') as f:
        f.write(f'path: {OUT}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: People\n  1: Vehicle\n')
    n = {s: sum(len(allb[x[0]]) for x in xs) for s, xs in parts.items()}
    crowd = sum(b[5] for bx in allb.values() for b in bx)
    print(json.dumps({'images': {s: len(x) for s, x in parts.items()}, 'boxes': n, 'crowd boxes': int(crowd)}))


if __name__ == '__main__':
    main()
