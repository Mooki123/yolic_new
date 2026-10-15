"""Build the Outdoor Hazard cache: 224x224 uint8 image memmap, cell labels, pHash and session keys.

The released training script reads one 1248-integer .txt per frame (yoliclabel/). The Kaggle copy
ships Pascal-VOC XML instead (yolic_xml/), where every object is one (class, cell) pair whose box is
exactly that cell's rectangle. Each box is matched to the layout cell with the same rectangle; the
Road (background) bit is set when no hazard class is present in the cell, which is the released
code's 'normal' cell [0] * 11 + [1] (outdoor_yolic.py:pred_acc).

Images are squashed from 848x480 to 224x224 with bilinear + antialias, which is what
transforms.Resize((224, 224)) does on a PIL image in the released transform.

Run:  python tools/cache_outdoor.py --data ~/dev/yolic/outdoor --out ~/data/yolic_cache/outdoor
Writes: images.u8 (N, 224, 224, 3), labels.npy (N, 104, 12) uint8, names.txt, phash.npy (N,) uint64,
        sessions.txt, meta.json
"""
import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from yolic_bench.layout import Layout  # noqa: E402

CLASSES = ['Bump', 'Column', 'Dent', 'Fence', 'Creature', 'Vehicle', 'Wall', 'Weed', 'ZebraCrossing',
           'TrafficCone', 'TrafficSign']  # bit order of outdoor_eval.py:class_names; bit 11 = Road
SIZE = 224
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def session_of(name):
    """Recording session = the file stem without its 8-digit frame counter (191209_151537|00000510)."""
    stem = os.path.splitext(name)[0]
    return stem[:-8]


def cell_lookup(layout):
    """{(x1, y1, x2, y2): cell index} for a rectangle layout."""
    out = {}
    for i, poly in enumerate(layout.cells):
        key = tuple(int(v) for v in (poly[:, 0].min(), poly[:, 1].min(), poly[:, 0].max(), poly[:, 1].max()))
        out[key] = i
    return out


def parse_labels(xml_path, lookup, n_cells):
    lab = np.zeros((n_cells, len(CLASSES) + 1), dtype=np.uint8)
    unmatched = 0
    for obj in ET.parse(xml_path).getroot().iter('object'):
        b = obj.find('bndbox')
        key = tuple(int(float(b.find(k).text)) for k in ('xmin', 'ymin', 'xmax', 'ymax'))
        cell = lookup.get(key)
        if cell is None:
            unmatched += 1
            continue
        lab[cell, CLASSES.index(obj.find('name').text)] = 1
    lab[:, -1] = lab[:, :-1].max(axis=1) == 0
    return lab, unmatched


def load_one(args):
    img_path, xml_path, lookup, n_cells = args
    import imagehash
    im = Image.open(img_path).convert('RGB')
    ph = int(str(imagehash.phash(im)), 16)
    small = np.asarray(im.resize((SIZE, SIZE), Image.BILINEAR), dtype=np.uint8)
    lab, unmatched = parse_labels(xml_path, lookup, n_cells)
    return small, lab, ph, unmatched, im.size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.expanduser('~/dev/yolic/outdoor'))
    ap.add_argument('--out', default=os.path.expanduser('~/data/yolic_cache/outdoor'))
    ap.add_argument('--workers', type=int, default=6)
    args = ap.parse_args()

    layout = Layout.load(os.path.join(ROOT, 'layouts', 'outdoor.json'))
    lookup = cell_lookup(layout)
    names = sorted(os.listdir(os.path.join(args.data, 'RGB')))
    n = len(names)
    os.makedirs(args.out, exist_ok=True)
    images = np.lib.format.open_memmap(os.path.join(args.out, 'images.npy'), mode='w+', dtype=np.uint8,
                                       shape=(n, SIZE, SIZE, 3))
    labels = np.zeros((n, len(layout), len(CLASSES) + 1), dtype=np.uint8)
    phash = np.zeros(n, dtype=np.uint64)
    unmatched, sizes = 0, set()
    jobs = [(os.path.join(args.data, 'RGB', f), os.path.join(args.data, 'yolic_xml', os.path.splitext(f)[0] + '.xml'),
             lookup, len(layout)) for f in names]
    with ProcessPoolExecutor(args.workers) as ex:
        for i, (small, lab, ph, um, size) in enumerate(ex.map(load_one, jobs, chunksize=64)):
            images[i], labels[i], phash[i] = small, lab, ph
            unmatched += um
            sizes.add(size)
            if i % 2000 == 0:
                print(f'{i}/{n}', flush=True)
    images.flush()
    np.save(os.path.join(args.out, 'labels.npy'), labels)
    np.save(os.path.join(args.out, 'phash.npy'), phash)
    with open(os.path.join(args.out, 'names.txt'), 'w') as f:
        f.write('\n'.join(names) + '\n')
    with open(os.path.join(args.out, 'sessions.txt'), 'w') as f:
        f.write('\n'.join(session_of(x) for x in names) + '\n')
    meta = dict(n=n, size=SIZE, classes=CLASSES + ['Road'], layout='layouts/outdoor.json',
                frame_sizes=sorted(sizes), unmatched_boxes=unmatched,
                positives_per_class=labels.reshape(-1, labels.shape[-1]).sum(0).tolist())
    with open(os.path.join(args.out, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(meta, indent=1))


if __name__ == '__main__':
    main()
