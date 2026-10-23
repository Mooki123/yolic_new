"""Write the research/08 target layouts for layout transfer: layouts/cityscapes_{union,shift,fine}.json.

    python tools/make_l2_layouts.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def grid(x0, y0, w, h, nx, ny, y_max=None):
    cells = []
    for r in range(ny):
        for c in range(nx):
            x1, y1, x2, y2 = x0 + c * w, y0 + r * h, x0 + (c + 1) * w, y0 + (r + 1) * h
            if y_max is not None:
                y2 = min(y2, y_max)
            cells.append([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
    return cells


def perspective(vp=(1024, 300), rows=(400, 440, 490, 550, 620, 700, 800, 910, 1024), cols=12, width=2048):
    """Trapezoid cells along a road receding to the vanishing point vp; boundaries radiate from vp, so the layout is
    irregular (no edge is shared with the L1 rectangles) but left-right symmetric. Rows 0-3 far, 4-7 near."""
    def x_at(xb, y):  # the line through vp and (xb, 1024), evaluated at height y
        return vp[0] + (xb - vp[0]) * (y - vp[1]) / (1024 - vp[1])
    cells = []
    for y1, y2 in zip(rows[:-1], rows[1:]):
        for k in range(cols):
            xa, xb = k * width / cols, (k + 1) * width / cols
            cells.append([[x_at(xa, y1), y1], [x_at(xb, y1), y1], [x_at(xb, y2), y2], [x_at(xa, y2), y2]])
    half = len(cells) // 2
    return cells[:half], cells[half:]


LAYOUTS = {
    # 2x2 merges of the L1 cells
    'union': (grid(512, 320, 128, 64, 8, 5), grid(0, 640, 256, 128, 8, 3)),
    # half-cell offset in x and y, symmetric about x = 1024 (one column fewer per band)
    'shift': (grid(544, 336, 64, 32, 15, 10), grid(64, 672, 128, 64, 15, 6, y_max=1024)),
    # 2x2 splits of the L1 cells
    'fine': (grid(512, 320, 32, 16, 32, 20), grid(0, 640, 64, 32, 32, 12)),
    # irregular perspective trapezoids (research/08 round 2); groups are far rows / near rows
    'poly': perspective(),
}

for name, (small, large) in LAYOUTS.items():
    d = {'name': f'cityscapes_{name}', 'frame': [2048, 1024], 'source': 'tools/make_l2_layouts.py (research/08)',
         'groups': {'small_band': list(range(len(small))),
                    'large_band': list(range(len(small), len(small) + len(large)))},
         'cells': small + large}
    with open(os.path.join(ROOT, 'layouts', f'cityscapes_{name}.json'), 'w') as f:
        json.dump(d, f)
    print(name, len(small), '+', len(large), '=', len(small) + len(large))
