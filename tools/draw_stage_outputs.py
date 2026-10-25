"""Stage-by-stage outputs of the trained F+AL model on one Cityscapes test frame, drawn in the same visual style as
docs/architecture/f_al_architecture.svg (tools/draw_architecture.py). Real tensors from runs/cityscapes_f_align/seed0:
the two input views, the stride-16 backbone features and the fused stride-16 map (both shown as their first three
principal components in RGB), the People / Vehicle logit maps, each view's cell readout, and the blended output next
to the ground truth.

    python tools/draw_stage_outputs.py [--key val/munster/munster_000150_000019]
writes docs/architecture/f_al_stage_outputs.svg
"""
import argparse
import base64
import io
import os
import sys

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tools'))
import draw_architecture as A  # noqa: E402  (shared style: text, block, arrow, fusion, colours)
from yolic_bench import heads  # noqa: E402
from yolic_bench import metrics as M  # noqa: E402
from yolic_bench.data import CellCache, GpuAugment  # noqa: E402
from yolic_bench.layout import Layout  # noqa: E402

OUT = os.path.join(ROOT, 'docs', 'architecture', 'f_al_stage_outputs.svg')
RUN = os.path.join(ROOT, 'runs', 'cityscapes_f_align', 'seed0')
PEOPLE = np.array([198, 106, 18]) / 255  # the orange accent of the architecture figure
VEHICLE = np.array([47, 110, 186]) / 255
W, H = 1340, 1150


# ---------------- tensors ----------------
def load_model(lay):
    cfg = yaml.safe_load(open(os.path.join(RUN, 'config.yaml')))
    specs = [((0, 0, 2048, 1024), (224, 112)), ((512, 320, 1536, 640), (320, 96))]
    m = heads.build(cfg['model'], lay, 4, pretrained=False, view_specs=specs).eval()
    m.load_state_dict(torch.load(os.path.join(RUN, 'best.pt'), map_location='cpu'))
    return m, specs


@torch.no_grad()
def stages(model, views):
    out = {}
    for v, x in enumerate(views):
        t = model.trunks[v]
        x16 = t.stride16(x)
        fused = t(x)
        z = model.classifier(fused)
        r = model.readout(v, x) + model.cell_bias
        out[v] = dict(x16=x16[0].numpy(), fused=fused[0].numpy(), z=torch.sigmoid(z)[0].numpy(),
                      r=torch.sigmoid(r)[0].numpy())
    out['final'] = torch.sigmoid(model(*views))[0].numpy()
    return out


# ---------------- rasters ----------------
def pca_rgb(f):
    """(C, h, w) features -> (h, w, 3) in [0, 1] from the first three principal components."""
    c, h, w = f.shape
    x = f.reshape(c, -1).T
    x = x - x.mean(0)
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    p = x @ vt[:3].T
    p = (p - p.min(0)) / np.maximum(p.max(0) - p.min(0), 1e-6)
    return p.reshape(h, w, 3)


def two_class_map(pp, pv):
    """Probability maps -> RGB on white: People in orange, Vehicle in blue."""
    img = np.ones(pp.shape + (3,))
    img = img * (1 - pp[..., None]) + PEOPLE * pp[..., None]
    img = img * (1 - pv[..., None]) + VEHICLE * pv[..., None]
    return img


def to_png(arr, size, nearest=True):
    im = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
    im = im.resize(size, Image.NEAREST if nearest else Image.BILINEAR)
    b = io.BytesIO()
    im.save(b, format='PNG')
    return base64.b64encode(b.getvalue()).decode()


def cell_overlay(base_rgb, lay, probs, crop, size, cells, thr=0.5):
    """Draw cells over an image (crop of the source frame -> size): People / Vehicle cells above thr filled with the
    class colour, opacity = probability; other cells outlined faintly."""
    im = Image.fromarray((base_rgb * 255).astype(np.uint8)).convert('L').convert('RGBA').resize(size, Image.BILINEAR)
    over = Image.new('RGBA', size, (0, 0, 0, 0))
    d = ImageDraw.Draw(over)
    x1, y1, x2, y2 = crop
    sx, sy = size[0] / (x2 - x1), size[1] / (y2 - y1)
    for n in cells:
        c = lay.cells[n]
        box = [(c[:, 0].min() - x1) * sx, (c[:, 1].min() - y1) * sy, (c[:, 0].max() - x1) * sx - 1, (c[:, 1].max() - y1) * sy - 1]
        pp, pv = probs[n, 0], probs[n, 1]
        if max(pp, pv) > thr:
            col = PEOPLE if pp >= pv else VEHICLE
            d.rectangle(box, fill=tuple(int(v * 255) for v in col) + (int(90 + 130 * max(pp, pv)),))
        d.rectangle(box, outline=(47, 59, 69, 70))
    im = Image.alpha_composite(im, over).convert('RGB')
    b = io.BytesIO()
    im.save(b, format='PNG')
    return base64.b64encode(b.getvalue()).decode()


# ---------------- drawing ----------------
def image(x, y, w, h, b64):
    A.add(f'<image x="{x}" y="{y}" width="{w}" height="{h}" preserveAspectRatio="none" href="data:image/png;base64,{b64}"/>')
    A.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="none" stroke="{A.STROKE}" stroke-width="0.8" stroke-opacity="0.6"/>')


def tile(x, y, w, img_h, title, caption, color, b64, img_w=None, caption2=None):
    """Block in the architecture style holding one stage output image."""
    h = 44 + img_h + 46
    A.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="3" fill="#FFFFFF" stroke="{A.STROKE}" stroke-width="1.41"/>')
    A.add(f'<rect x="{x + 0.7}" y="{y + 0.7}" width="{w - 1.4}" height="7" fill="{color}"/>')
    A.add(f'<rect x="{x + 0.7}" y="{y + 7.7}" width="{w - 1.4}" height="3" fill="{color}" fill-opacity="0.45"/>')
    A.text(x + w / 2, y + 31, title, size=15, weight=650)
    iw = img_w or (w - 24)
    image(x + (w - iw) / 2, y + 42, iw, img_h, b64)
    A.text(x + w / 2, y + 42 + img_h + 20, caption, size=12.5, opacity=0.68)
    if caption2:
        A.text(x + w / 2, y + 42 + img_h + 37, caption2, size=12, opacity=0.6)
    return h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--key', default='val/munster/munster_000150_000019')
    args = ap.parse_args()

    lay = Layout.load(os.path.join(ROOT, 'layouts', 'cityscapes.json'))
    model, specs = load_model(lay)
    cache = CellCache(os.path.expanduser('~/data/yolic_cache/cityscapes'), ['g224x112', 'fov320x96'])
    row = cache.index[args.key]
    aug = GpuAugment(None, 'none', None, True)
    views = [aug.to_input(torch.from_numpy(np.array(v[row:row + 1]))) for v in cache.views]
    S = stages(model, views)
    gt, valid = cache.labels[row], cache.valid[row]
    split, city, stem = args.key.split('/')
    frame = np.asarray(Image.open(os.path.expanduser(f'~/data/cityscapes/leftImg8bit/{split}/{city}/{stem}_leftImg8bit.png'))
                       .convert('RGB').resize((1024, 512), Image.BILINEAR)) / 255.0
    view_rgb = [cache.views[0][row] / 255.0, cache.views[1][row] / 255.0]
    pred = S['final'] > 0.5
    f1 = 100 * float(M.macro_f1(M.counts(gt[None], pred[None], valid[None]), [0, 1]))
    pv_counts = (int(gt[valid, 0].sum()), int(gt[valid, 1].sum()))

    A.parts.clear()
    A.add(f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{W}" height="{H}" '
          f'viewBox="0 0 {W} {H}">')
    A.add('<defs><marker id="arrow" markerWidth="7" markerHeight="7" refX="6.2" refY="3.5" orient="auto" '
          f'markerUnits="userSpaceOnUse"><path d="M0,0 L7,3.5 L0,7 z" fill="{A.ARROW}"/></marker></defs>')
    A.add(f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>')
    A.text(18, 32, f'Stage outputs of F+AL on a Cityscapes test frame ({stem})', size=18, weight=650, anchor='start')

    # input frame with the fovea band
    fw, fh = 300, 150
    tile(30, 52, fw + 24, fh, 'Input frame', '3 × 1024 × 2048', A.BLUE, to_png(frame, (fw * 2, fh * 2), nearest=False),
         img_w=fw, caption2='dashed: far band seen by the fovea view')
    A.add(f'<rect x="{42 + fw * 0.25:.1f}" y="{94 + fh * 0.3125:.1f}" width="{fw * 0.5:.1f}" height="{fh * 0.3125:.1f}" '
          f'fill="none" stroke="{A.ORANGE}" stroke-width="1.6" stroke-dasharray="5 3"/>')

    # legend
    lx = 400
    A.text(lx, 80, 'How to read the panels', size=15, weight=650, anchor='start')
    legend = [('Feature maps: first three principal components of the channels, as RGB', None),
              ('People probability', PEOPLE), ('Vehicle probability', VEHICLE),
              ('Cell readouts: cells above 0.5, coloured by the more likely class, opacity = probability', None)]
    for i, (s, col) in enumerate(legend):
        y = 104 + i * 21
        if col is not None:
            A.add(f'<rect x="{lx}" y="{y - 11}" width="14" height="14" fill="rgb{tuple(int(v * 255) for v in col)}"/>')
        A.text(lx + (22 if col is not None else 0), y, s, size=13, anchor='start', opacity=0.8)
    A.text(lx, 210, f'Model: F+AL (runs/cityscapes_f_align/seed0), Cityscapes test (official val), '
           f'ground truth: {pv_counts[0]} People cells, {pv_counts[1]} Vehicle cells', size=12.5, anchor='start', opacity=0.65)

    # two rows: context view and fovea view
    rows = [(340, 0, (224, 112), (200, 100), (0, 0, 2048, 1024), range(256), 'Context view', 'all 256 cells'),
            (610, 1, (320, 96), (220, 66), (512, 320, 1536, 640), range(160), 'Fovea view', '160 far cells')]
    xs = [30, 290, 550, 810, 1070]
    for y, v, (iw, ih), (dw, dh), crop, cells, name, cell_txt in rows:
        s = S[v]
        h16, w16 = s['x16'].shape[1:]
        tile(xs[0], y, 240, dh, name, f'3 × {ih} × {iw}', A.BLUE, to_png(view_rgb[v], (dw * 2, dh * 2), nearest=False), img_w=dw)
        tile(xs[1], y, 240, dh, 'Backbone stride 16', f'96 × {h16} × {w16}', A.SAND, to_png(pca_rgb(s['x16']), (dw, dh)), img_w=dw,
             caption2='MobileNetV2 layers 0–13')
        tile(xs[2], y, 240, dh, 'Fused stride-16 map', f'1408 × {h16} × {w16}', A.MINT, to_png(pca_rgb(s['fused']), (dw, dh)),
             img_w=dw, caption2='upsampled stride 32 + lateral')
        tile(xs[3], y, 240, dh, 'Logit maps z', f'People / Vehicle, {h16} × {w16}', A.SAND,
             to_png(two_class_map(s['z'][0], s['z'][1]), (dw, dh)), img_w=dw, caption2='1×1 classifier, sigmoid')
        tile(xs[4], y, 240, dh, 'Cell readout r_v', cell_txt, A.SAGE,
             cell_overlay(view_rgb[v], lay, s['r'], crop, (dw * 2, dh * 2), cells), img_w=dw,
             caption2='mask-weighted mean + bias')
        cy = y + 44 + dh / 2
        for a, b in zip(xs[:-1], xs[1:]):
            A.arrow([(a + 240, cy), (b - 6, cy)])
        A.text(18, y - 10, f'{"(i)" if v == 0 else "(ii)"} {name} branch', size=15, weight=650, anchor='start', color=A.ORANGE)
    # frame -> views: down into the context view; along the left margin into the fovea view
    A.arrow([(230, 292), (230, 334)])
    A.arrow([(30, 167), (14, 167), (14, 610 + 44 + 33), (24, 610 + 44 + 33)])

    # blend and outputs
    fy = 880
    A.fusion(1190, fy + 90, 'Σ')
    for y, dh in ((340, 100), (610, 66)):
        A.arrow([(1310, y + 44 + dh / 2), (1325, y + 44 + dh / 2), (1325, fy + 90), (1215, fy + 90)])
    A.text(1190, fy + 130, 'per-cell view blend', size=13, opacity=0.75)
    A.text(1190, fy + 147, '+ cell bias, sigmoid', size=12, opacity=0.6)
    ow, oh = 300, 150
    tile(800, fy, ow + 24, oh, 'Final cell output', f'256 cells × 4 · P+V F1 on this frame {f1:.1f}', A.PALE,
         cell_overlay(frame, lay, S['final'], (0, 0, 2048, 1024), (ow * 2, oh * 2), range(256)), img_w=ow)
    A.arrow([(1171, fy + 90), (1130, fy + 90)])
    gt_probs = np.where(valid[:, None], gt.astype(float), 0.0)
    tile(440, fy, ow + 24, oh, 'Ground truth cells', 'from the full-resolution masks (≥ 1 px)', A.BLUE,
         cell_overlay(frame, lay, gt_probs, (0, 0, 2048, 1024), (ow * 2, oh * 2), range(256)), img_w=ow)
    A.text(782, fy + 95, 'vs', size=15, weight=650, opacity=0.7)

    A.add('</svg>')
    with open(OUT, 'w') as f:
        f.write('\n'.join(A.parts))
    print('wrote', OUT, f'| P+V F1 on this frame {f1:.1f}')


if __name__ == '__main__':
    main()
