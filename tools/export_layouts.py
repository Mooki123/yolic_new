"""Write layouts/{cityscapes,outdoor,indoor}.json from the cell geometry in the released scripts.

Run from the repository root:  python tools/export_layouts.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from yolic_bench import released  # noqa: E402


def rect_to_poly(rect):
    (x1, y1), (x2, y2) = rect
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def flat_to_poly(flat):
    return [[flat[i], flat[i + 1]] for i in range(0, len(flat), 2)]


LAYOUTS = {
    'cityscapes': dict(
        frame=[2048, 1024], source='cityscapes_yolic.py: cell_list',
        cells=lambda: [rect_to_poly(r) for r in released.cityscapes_cells()],
        # 160 small 64x32 cells in the central band, 96 large 128x64 cells over the full width.
        groups={'small_64x32': list(range(0, 160)), 'large_128x64': list(range(160, 256))}),
    'outdoor': dict(
        frame=[848, 480], source='outdoor_pred.py: points_list / cell_list',
        cells=lambda: [rect_to_poly(r) for r in released.outdoor_cells()],
        groups={'far_34px': list(range(0, 32)), 'near_53px': list(range(32, 96)),
                'top_60px': list(range(96, 104))}),
    'indoor': dict(
        frame=[848, 480], source='indoor_pred.py: polygonList',
        cells=lambda: [flat_to_poly(p) for p in released.indoor_polygons()],
        groups={'row1': list(range(0, 8)), 'row2': list(range(8, 16)), 'row3': list(range(16, 22)),
                'row4': list(range(22, 26)), 'row5': list(range(26, 30))}),
}


def build(name):
    spec = LAYOUTS[name]
    return {'name': name, 'frame': spec['frame'], 'source': spec['source'],
            'cells': spec['cells'](), 'groups': spec['groups']}


def main():
    os.makedirs('layouts', exist_ok=True)
    for name in LAYOUTS:
        d = build(name)
        with open(os.path.join('layouts', name + '.json'), 'w') as f:
            # One cell per line keeps the files diffable.
            f.write('{\n')
            for key in ('name', 'frame', 'source', 'groups'):
                f.write(f'  {json.dumps(key)}: {json.dumps(d[key])},\n')
            f.write('  "cells": [\n')
            f.write(',\n'.join('    ' + json.dumps(c) for c in d['cells']))
            f.write('\n  ]\n}\n')
        print(f'layouts/{name}.json: {len(d["cells"])} cells')


if __name__ == '__main__':
    main()
