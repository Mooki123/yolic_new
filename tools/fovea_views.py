"""The layout-driven foveation rule of research/06, and an image-view cache for flat datasets (outdoor).

derive(layout, group, s, total_px) -> {'context': (box, w, h), 'fovea': (box, w, h)}:
  fovea   = bounding box of the layout's cell group `group`, given s * total_px pixels;
  context = the full frame, given (1 - s) * total_px pixels;
  each view keeps its source aspect ratio; sizes are rounded to multiples of 8.

    python tools/fovea_views.py derive --layout outdoor --group far_34px --s 0.85 --total 54272
    python tools/fovea_views.py cache --cache ~/data/yolic_cache/outdoor --images ~/dev/yolic/outdoor/RGB \
        --layout outdoor --group far_34px --s 0.85 --total 54272
The cache command writes images_<name>.npy (rows in names.txt order) and records the views in meta.json.
"""
import argparse
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from yolic_bench.layout import Layout  # noqa: E402


def _size(box, px):
    aspect = (box[2] - box[0]) / (box[3] - box[1])
    h = math.sqrt(px / aspect)
    return max(8, int(round(aspect * h / 8)) * 8), max(8, int(round(h / 8)) * 8)


def derive(layout, group, s, total_px):
    pts = np.concatenate([layout.cells[i] for i in layout.groups[group]])
    fovea = tuple(int(v) for v in (pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()))
    if abs((fovea[0] + fovea[2]) - layout.width) > 1:
        raise ValueError(f'fovea {fovea} is not symmetric about the vertical midline; the flip would be wrong')
    context = (0, 0, layout.width, layout.height)
    return {'context': (context, *_size(context, (1 - s) * total_px)),
            'fovea': (fovea, *_size(fovea, s * total_px))}


def _bbox(layout, group):
    pts = np.concatenate([layout.cells[i] for i in layout.groups[group]])
    box = tuple(int(v) for v in (pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()))
    if abs((box[0] + box[2]) - layout.width) > 1:
        raise ValueError(f'band {group} {box} is not symmetric about the vertical midline; the flip would be wrong')
    return box


def derive_bands(layout, s, total_px, min_side=32):
    """The frozen rule of research/06 (addendum): one view per cell group, cropped to the group's bounding box;
    the smallest-cell group gets s of the pixels, the others share 1 - s in proportion to their box areas;
    aspect kept, sides rounded to multiples of 8 and clamped to >= min_side. Returns {group: (box, w, h)}."""
    area = {g: np.mean([np.ptp(layout.cells[i][:, 0]) * np.ptp(layout.cells[i][:, 1]) for i in c])
            for g, c in layout.groups.items()}
    fovea = min(area, key=area.get)
    boxes = {g: _bbox(layout, g) for g in layout.groups}
    rest = {g: (b[2] - b[0]) * (b[3] - b[1]) for g, b in boxes.items() if g != fovea}
    share = {fovea: s} | {g: (1 - s) * a / sum(rest.values()) for g, a in rest.items()}
    out = {}
    for g, b in boxes.items():
        w, h = _size(b, share[g] * total_px)
        out[g] = (b, max(w, min_side), max(h, min_side))
    return out


def _render(job):
    path, views = job
    im = Image.open(path).convert('RGB')
    return [np.asarray(im.crop(box).resize((w, h), Image.BILINEAR), dtype=np.uint8) for box, w, h in views]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['derive', 'cache'])
    ap.add_argument('--rule', choices=['context', 'bands'], default='bands',
                    help='context: full frame + fovea (research/06 stage 1); bands: one view per cell group (addendum)')
    ap.add_argument('--layout', default='outdoor')
    ap.add_argument('--group', default='far_34px')
    ap.add_argument('--s', type=float, default=0.76)
    ap.add_argument('--total', type=int, default=51712)
    ap.add_argument('--cache', default=os.path.expanduser('~/data/yolic_cache/outdoor'))
    ap.add_argument('--images', default=os.path.expanduser('~/dev/yolic/outdoor/RGB'))
    ap.add_argument('--workers', type=int, default=6)
    args = ap.parse_args()

    layout = Layout.load(os.path.join(ROOT, 'layouts', args.layout + '.json'))
    if args.rule == 'bands':
        named = {f'{g}_{w}x{h}': (b, w, h) for g, (b, w, h) in derive_bands(layout, args.s, args.total).items()}
    else:
        views = derive(layout, args.group, args.s, args.total)
        named = {('ctx' if k == 'context' else 'fov') + f'{w}x{h}': (b, w, h) for k, (b, w, h) in views.items()}
    print(json.dumps({k: {'crop': list(b), 'size': [w, h], 'pixels': w * h} for k, (b, w, h) in named.items()}),
          f'total {sum(w * h for _, w, h in named.values())}')
    if args.cmd == 'derive':
        return
    names = open(os.path.join(args.cache, 'names.txt')).read().split()
    arrays = {k: np.lib.format.open_memmap(os.path.join(args.cache, f'images_{k}.npy'), 'w+', np.uint8, (len(names), h, w, 3))
              for k, (_, w, h) in named.items()}
    specs = list(named.values())
    with ProcessPoolExecutor(args.workers) as ex:
        for i, out in enumerate(ex.map(_render, [(os.path.join(args.images, n), specs) for n in names], chunksize=64)):
            for arr, im in zip(arrays.values(), out):
                arr[i] = im
            if i % 4000 == 0:
                print(f'{i}/{len(names)}', flush=True)
    for arr in arrays.values():
        arr.flush()
    meta_path = os.path.join(args.cache, 'meta.json')
    meta = json.load(open(meta_path))
    meta.setdefault('views', {}).update({k: {'crop': list(b), 'size': [w, h]} for k, (b, w, h) in named.items()})
    json.dump(meta, open(meta_path, 'w'), indent=1)
    print('done', list(named))


if __name__ == '__main__':
    main()
