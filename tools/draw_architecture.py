"""Draw the F+AL architecture figure as an editable SVG in a two-panel publication style
((a) overall architecture, (b) detailed proposed block), following the visual conventions of
github.com/Ztsdut/ml-architecture-diagram-skill (white blocks with coloured header bands, stacked tensor cards,
orthogonal arrows, fusion circles, small vector illustrations). Shapes are the measured Cityscapes ones.

    python tools/draw_architecture.py          # writes docs/architecture/f_al_architecture.svg
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'docs', 'architecture', 'f_al_architecture.svg')

INK, ARROW = '#1F2933', '#36454F'
STROKE = '#2F3B45'
BLUE, MINT, SAND, SAGE, PALE = '#DCEAF7', '#DCEFEA', '#F3D7A7', '#DDEBD8', '#E8F0E4'
ORANGE, GREY = '#C66A12', '#9AA5AE'
FONT = 'Arial, Helvetica, sans-serif'
W, H = 1320, 1095

parts = []


def add(s):
    parts.append(s)


def text(x, y, s, size=13.9, weight=400, anchor='middle', opacity=1.0, color=INK, italic=False):
    st = ' font-style="italic"' if italic else ''
    add(f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT}" font-size="{size}" font-weight="{weight}" '
        f'text-anchor="{anchor}" fill="{color}" fill-opacity="{opacity}"{st}>{s}</text>')


def block(x, y, w, h, title, sub=None, color=SAND, tag=None, dashed=False, sub2=None):
    dash = ' stroke-dasharray="5 3"' if dashed else ''
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="3" fill="#FFFFFF" stroke="{STROKE}" stroke-width="1.41"{dash}/>')
    add(f'<rect x="{x + 0.7}" y="{y + 0.7}" width="{w - 1.4}" height="7" fill="{color}"/>')
    add(f'<rect x="{x + 0.7}" y="{y + 7.7}" width="{w - 1.4}" height="3" fill="{color}" fill-opacity="0.45"/>')
    text(x + w / 2, y + 32, title, size=16.5, weight=650)
    if sub:
        text(x + w / 2, y + 51, sub, size=13.2, opacity=0.68)
    if sub2:
        text(x + w / 2, y + 68, sub2, size=13.2, opacity=0.68)
    if tag:
        text(x + w - 8, y + 26, tag, size=13.9, weight=700, anchor='end', color=ORANGE)


def card(x, y, w, h, color, n=4, grid=(6, 4)):
    """Stacked translucent tensor slices with a faint grid (front slice at (x, y))."""
    for i in reversed(range(n)):
        o = 3.4 * i
        op = 0.90 - 0.14 * i
        add(f'<rect x="{x + o:.1f}" y="{y - o:.1f}" width="{w}" height="{h}" rx="2" fill="{color}" '
            f'fill-opacity="{op:.2f}" stroke="{STROKE}" stroke-width="0.8" stroke-opacity="0.55"/>')
    gx, gy = grid
    for i in range(1, gx):
        add(f'<line x1="{x + w * i / gx:.1f}" y1="{y}" x2="{x + w * i / gx:.1f}" y2="{y + h}" stroke="{STROKE}" '
            f'stroke-width="0.63" stroke-opacity="0.23"/>')
    for j in range(1, gy):
        add(f'<line x1="{x}" y1="{y + h * j / gy:.1f}" x2="{x + w}" y2="{y + h * j / gy:.1f}" stroke="{STROKE}" '
            f'stroke-width="0.63" stroke-opacity="0.23"/>')


def arrow(pts, width=1.9, dashed=False, color=ARROW, marker=True):
    d = 'M' + ' L'.join(f'{x:.1f},{y:.1f}' for x, y in pts)
    dash = ' stroke-dasharray="5 3"' if dashed else ''
    m = ' marker-end="url(#arrow)"' if marker else ''
    add(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linecap="round" '
        f'stroke-linejoin="round"{dash}{m}/>')


def fusion(cx, cy, sym, r=18.5):
    add(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{SAGE}" fill-opacity="0.70" stroke="{STROKE}" stroke-width="1.41"/>')
    text(cx, cy + 6.5, sym, size=18, weight=650)


# ---------- small vector illustrations ----------
def frame_illustration(x, y, w, h):
    """Street scene: horizon, road receding to a vanishing point, the far band marked as the fovea."""
    hy = y + h * 0.42
    add(f'<line x1="{x + 4}" y1="{hy:.1f}" x2="{x + w - 4}" y2="{hy:.1f}" stroke="{STROKE}" stroke-width="0.7" stroke-opacity="0.5"/>')
    vx = x + w / 2
    for bx in (x + w * 0.12, x + w * 0.88):
        add(f'<line x1="{vx:.1f}" y1="{hy:.1f}" x2="{bx:.1f}" y2="{y + h - 3}" stroke="{STROKE}" stroke-width="0.8" stroke-opacity="0.55"/>')
    for i, cx in enumerate((x + w * 0.38, x + w * 0.58)):
        add(f'<rect x="{cx:.1f}" y="{hy - 7 + i * 3:.1f}" width="{7 - i}" height="{5 - i}" fill="{STROKE}" fill-opacity="0.55"/>')
    add(f'<rect x="{x + w * 0.25:.1f}" y="{y + h * 0.31:.1f}" width="{w * 0.5:.1f}" height="{h * 0.31:.1f}" fill="none" '
        f'stroke="{ORANGE}" stroke-width="1.3" stroke-dasharray="3 2"/>')


def pyramid(cx, cy):
    """Feature pyramid: three shrinking slabs (stride 4, 8, 16)."""
    for i, (w, h) in enumerate(((30, 22), (22, 16), (15, 11))):
        x = cx - 34 + i * 26
        add(f'<rect x="{x}" y="{cy - h / 2:.1f}" width="{w}" height="{h}" rx="1.5" fill="{SAND}" fill-opacity="{0.5 + 0.18 * i:.2f}" '
            f'stroke="{STROKE}" stroke-width="0.7" stroke-opacity="0.7"/>')


def mask_grid(x, y, cols=6, rows=3, s=11):
    """Cell mask over a feature grid: one cell's bilinear weights (darker = larger)."""
    weights = {(2, 1): 0.95, (3, 1): 0.75, (2, 0): 0.35, (3, 0): 0.3, (2, 2): 0.3, (3, 2): 0.25, (1, 1): 0.2, (4, 1): 0.15}
    for r in range(rows):
        for c in range(cols):
            a = weights.get((c, r), 0.0)
            add(f'<rect x="{x + c * s}" y="{y + r * s}" width="{s - 1.5}" height="{s - 1.5}" fill="{SAND}" '
                f'fill-opacity="{0.12 + 0.85 * a:.2f}" stroke="{STROKE}" stroke-width="0.45" stroke-opacity="0.5"/>')


def cell_output(x, y, cols=10, rows=4, s=9):
    hot = {(3, 1): BLUE, (4, 1): BLUE, (6, 2): SAND, (7, 2): SAND, (6, 3): SAND, (1, 3): BLUE}
    for r in range(rows):
        for c in range(cols):
            col = hot.get((c, r), '#FFFFFF')
            add(f'<rect x="{x + c * s}" y="{y + r * s}" width="{s - 1}" height="{s - 1}" fill="{col}" '
                f'stroke="{STROKE}" stroke-width="0.45" stroke-opacity="0.55"/>')


def build():
    add(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
    add('<defs><marker id="arrow" markerWidth="7" markerHeight="7" refX="6.2" refY="3.5" orient="auto" '
        f'markerUnits="userSpaceOnUse"><path d="M0,0 L7,3.5 L0,7 z" fill="{ARROW}"/></marker></defs>')
    add(f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>')

    # ================= (a) overall architecture =================
    text(18, 32, '(a) Overall architecture: F+AL (foveated YOLIC-Align)', size=18, weight=650, anchor='start')

    # input frame
    text(105, 128, 'Camera frame', size=16.5, weight=620)
    card(40, 150, 130, 82, BLUE)
    frame_illustration(40, 150, 130, 82)
    text(105, 258, '3 × 1024 × 2048', opacity=0.68)
    text(105, 276, 'fovea = far band', size=12.5, opacity=0.68, color=ORANGE)

    # the two views
    text(305, 74, 'Context view', size=15.5, weight=620)
    card(255, 90, 100, 50, BLUE, grid=(8, 4))
    text(305, 162, 'whole frame', size=12.8, opacity=0.68)
    text(305, 178, '3 × 112 × 224', size=12.8, opacity=0.68)
    text(305, 236, 'Fovea view', size=15.5, weight=620)
    card(255, 252, 100, 42, BLUE, grid=(10, 3))
    text(305, 316, 'far band (512,320)–(1536,640)', size=12.8, opacity=0.68)
    text(305, 332, '3 × 96 × 320', size=12.8, opacity=0.68)
    arrow([(175, 175), (212, 175), (212, 115), (249, 115)])
    arrow([(175, 205), (212, 205), (212, 273), (249, 273)])

    # trunks
    for i, (y, shape) in enumerate(((62, '1408 × 7 × 14'), (226, '1408 × 6 × 20'))):
        block(410, y, 200, 112, 'MobileNetV2 trunk', 'stride 16 + lateral fusion', SAND,
              sub2=('view-0 BatchNorm' if i == 0 else 'view-1 BatchNorm'))
        pyramid(510, y + 92)
        text(510, y + 134, shape, size=12.8, opacity=0.68)
    arrow([(365, 115), (404, 115)])
    arrow([(365, 273), (404, 273)])
    add(f'<line x1="618" y1="128" x2="618" y2="236" stroke="{GREY}" stroke-width="1.3" stroke-dasharray="4 3"/>')
    text(626, 176, 'shared conv', size=12, anchor='start', opacity=0.75)
    text(626, 191, 'weights', size=12, anchor='start', opacity=0.75)

    # readouts (the proposed block)
    for i, (y, shape) in enumerate(((62, '256 cells × 4'), (226, '160 far cells × 4'))):
        block(700, y, 200, 112, 'Align cell readout', 'mask-weighted mean', SAND,
              sub2='over the cells in view')
        mask_grid(767, y + 78)
        text(800, y + 134, shape, size=12.8, opacity=0.68)
    arrow([(610, 118), (694, 118)])
    arrow([(610, 282), (694, 282)])

    # the proposed branch (detailed in panel b), one per view
    add(f'<rect x="398" y="50" width="514" height="318" rx="6" fill="none" stroke="{ORANGE}" stroke-width="1.3" '
        f'stroke-dasharray="6 4"/>')
    text(906, 44, '(b) Align branch, one per view', size=13.9, weight=700, anchor='end', color=ORANGE)

    # per-cell view blend + output
    fusion(985, 200, 'Σ')
    arrow([(900, 118), (985, 118), (985, 175)])
    arrow([(900, 282), (985, 282), (985, 225)])
    text(997, 262, 'per-cell view blend', size=13, opacity=0.75, anchor='start')
    text(997, 279, 'a = softmax(view_logit)', size=12, opacity=0.6, anchor='start')
    block(1035, 160, 110, 80, '+ bias', 'sigmoid', SAGE)
    arrow([(1004, 200), (1029, 200)])
    text(1240, 128, 'Cell hazards', size=16.5, weight=620)
    card(1185, 160, 110, 68, PALE, grid=(1, 1))
    cell_output(1194, 176)
    arrow([(1145, 200), (1179, 200)])
    text(1240, 258, '256 × 4', opacity=0.68)
    text(1240, 276, 'People · Vehicle', size=12.5, opacity=0.68)
    text(1240, 292, 'Other · Road', size=12.5, opacity=0.68)

    # summary line
    add(f'<rect x="40" y="372" width="1250" height="44" rx="3" fill="{PALE}" fill-opacity="0.55" stroke="{STROKE}" stroke-width="0.8" stroke-opacity="0.4"/>')
    text(665, 399, '2.28 M parameters (YOLIC 3.54 M) · 359 M MACs (1.14× YOLIC) · 9.8 ms on one CPU thread · '
         'Cityscapes People+Vehicle F1 86.9 (YOLIC 78.0, YOLOv8-N@640 86.9)', size=14, weight=500)

    # ================= (b) detailed proposed block =================
    add(f'<line x1="18" y1="448" x2="{W - 18}" y2="448" stroke="{STROKE}" stroke-width="0.6" stroke-opacity="0.25"/>')
    text(18, 482, '(b) Detailed structure of one Align branch (view v): backbone, lateral fusion and cell readout',
         size=18, weight=650, anchor='start')

    # row 1: features
    text(95, 560, 'View input', size=15.5, weight=620)
    card(45, 582, 100, 60, BLUE, grid=(8, 4))
    text(95, 670, '3 × H × W', size=12.8, opacity=0.68)
    block(200, 558, 210, 100, 'MobileNetV2 0–13', 'stride 16', SAND, sub2='96 × H/16 × W/16')
    arrow([(150, 612), (194, 612)])

    block(480, 510, 200, 90, 'MobileNetV2 14–18', 'stride 32 · 1280 ch', SAND)
    block(735, 510, 170, 90, 'Bilinear ↑2', 'to stride 16', MINT)
    block(480, 640, 200, 100, 'Lateral 1×1 conv', '96 → 128, BN, ReLU6', MINT, sub2='needed: −2.2 F1 without')
    arrow([(410, 590), (445, 590), (445, 555), (474, 555)])
    arrow([(410, 630), (445, 630), (445, 690), (474, 690)])
    arrow([(680, 555), (729, 555)])
    fusion(965, 620, 'C')
    arrow([(905, 555), (965, 555), (965, 595)])
    arrow([(680, 690), (965, 690), (965, 645)])
    text(1005, 600, 'concat', size=13, anchor='start', opacity=0.75)
    text(1005, 616, '1408 × H/16 × W/16', size=12.8, anchor='start', opacity=0.68)

    # row 2: classifier and readout (right to left)
    block(1065, 655, 200, 100, '1×1 classifier', '1408 → 4 bits', SAND, sub2='shared by all views')
    arrow([(984, 620), (1165, 620), (1165, 649)])
    text(855, 800, 'Logit map z', size=15.5, weight=620)
    card(805, 822, 100, 54, SAGE, grid=(8, 4))
    text(855, 902, '4 × P_v  (P_v = H/16 · W/16)', size=12.8, opacity=0.68)
    arrow([(1165, 755), (1165, 849), (911, 849)])

    block(470, 795, 230, 112, 'Mask-weighted mean', 'per cell n, per bit k', SAND,
          sub2='r = Σ_p w_v[n,p] · z[k,p]')
    mask_grid(552, 870, cols=6, rows=2, s=11)
    arrow([(799, 849), (706, 849)])

    block(470, 965, 230, 100, 'Cell masks w_v', 'exact cell coverage → bilinear', MINT,
          sub2='stride-16 grid, from layout + crop')
    arrow([(585, 965), (585, 913)])

    block(190, 965, 230, 100, 'LSE soft max + mix', 'optional (YOLIC-Align)', '#EEF1F3', dashed=True,
          sub2='not needed: +0.2 F1 without')
    arrow([(305, 965), (305, 885), (464, 885)], dashed=True, color=GREY, width=1.4)

    text(110, 800, 'Block output r_v', size=15.5, weight=620)
    card(60, 822, 100, 54, PALE, grid=(10, 3))
    text(110, 902, 'N_v cells × 4', size=12.8, opacity=0.68)
    text(110, 920, '→ per-cell view blend in (a)', size=12, opacity=0.6)
    arrow([(464, 849), (166, 849)])

    add('</svg>')
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as f:
        f.write('\n'.join(parts))
    print('wrote', OUT)


if __name__ == '__main__':
    build()
