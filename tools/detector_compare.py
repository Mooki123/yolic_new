"""research/14: cell models vs YOLOv8-N on Cityscapes with the shared per-cell metric, plus latency.

People+Vehicle macro-F1 on test at 0.5 and at per-class thresholds tuned on val' (each model its own thresholds);
paired bootstrap (seeds pooled) on the tuned predictions.

    python tools/detector_compare.py
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from yolic_bench import metrics as M  # noqa: E402
from yolic_bench.data import CellCache  # noqa: E402
from yolic_bench.layout import Layout  # noqa: E402
from yolic_bench.rasterise import BoxRasteriser  # noqa: E402

PV = [0, 1]
MODELS = {'A (YOLIC)': 'runs/cityscapes_b1', 'AL (Align)': 'runs/cityscapes_align', 'F (foveated)': 'runs/cityscapes_f_fovea_pvbn',
          'F+AL': 'runs/cityscapes_f_align', 'YOLOv8-N@224': 'runs/yolo_cells/yolov8n_224',
          'YOLOv8-N@640': 'runs/yolo_cells/yolov8n_640'}


def main():
    c = CellCache(os.path.expanduser('~/data/yolic_cache/cityscapes'))
    te = c.indices(os.path.join(ROOT, 'splits', 'cityscapes', 'test.txt'))
    gt, valid = c.labels[te], c.valid[te]
    f1 = lambda pred: 100 * float(M.macro_f1(M.counts(gt, pred, valid), PV))
    P, table = {}, {}
    for label, d in MODELS.items():
        d = os.path.join(ROOT, d)
        seeds = [s for s in range(3) if os.path.exists(os.path.join(d, f'seed{s}', 'metrics.json'))]
        if not seeds:
            continue
        ms = [json.load(open(os.path.join(d, f'seed{s}', 'metrics.json'))) for s in seeds]
        probs = [np.load(os.path.join(d, f'seed{s}', 'test_probs.npy')).astype(np.float32) for s in seeds]
        tuned = np.stack([p > np.asarray(m['tuned']['thresholds'])[:p.shape[-1]] for p, m in zip(probs, ms)])
        P[label] = tuned
        at05 = [f1(p > 0.5) for p in probs]
        tu = [f1(t) for t in tuned]
        table[label] = {'seeds': len(seeds), 'P+V @0.5': round(float(np.mean(at05)), 2),
                        'P+V tuned': round(float(np.mean(tu)), 2), 'tuned per seed': [round(x, 2) for x in tu],
                        'far (tuned)': round(float(np.mean([100 * M.macro_f1(M.counts(gt, t, valid, np.arange(160)), PV) for t in tuned])), 2),
                        'near (tuned)': round(float(np.mean([100 * M.macro_f1(M.counts(gt, t, valid, np.arange(160, 256)), PV) for t in tuned])), 2)}
        if 'box_mAP50' in ms[0]:
            table[label]['box mAP50 / 50-95'] = [round(100 * float(np.mean([m['box_mAP50'] for m in ms])), 1),
                                                 round(100 * float(np.mean([m['box_mAP50_95'] for m in ms])), 1)]
        if 'params' in ms[0]:
            table[label]['params_M'] = round(ms[0]['params'] / 1e6, 2)
    boxes = json.load(open(os.path.expanduser('~/data/yolo_cityscapes/boxes.json')))
    r = BoxRasteriser(Layout.load(os.path.join(ROOT, 'layouts', 'cityscapes.json')))
    keys = open(os.path.join(ROOT, 'splits', 'cityscapes', 'test.txt')).read().split()
    oracle = np.stack([np.pad(r.scores([(b[0], *b[1:5], 1.0) for b in boxes[k]])[:, :2], ((0, 0), (0, 2))) for k in keys]) > 0.5
    table['box oracle (GT boxes)'] = {'P+V tuned': round(f1(oracle), 2)}

    def delta(a, b):
        d, lo, hi = M.paired_bootstrap(gt, P[a], P[b], PV, n_boot=1000, seed=0, valid=valid)
        return [round(100 * d, 2), round(100 * lo, 2), round(100 * hi, 2)]

    res = {'table': table}
    if 'YOLOv8-N@224' in P:
        res['F+AL - YOLOv8-N@224 (tuned)'] = delta('YOLOv8-N@224', 'F+AL')
        res['A - YOLOv8-N@224 (tuned)'] = delta('YOLOv8-N@224', 'A (YOLIC)')
        res['claim: F+AL beats YOLOv8-N@224'] = bool(res['F+AL - YOLOv8-N@224 (tuned)'][1] > 0)
    if 'YOLOv8-N@640' in P:
        gap = table['F+AL']['P+V tuned'] - table['YOLOv8-N@640']['P+V tuned']
        res['F+AL - YOLOv8-N@640 (tuned, 1 seed)'] = round(gap, 2)
        res['claim: F+AL matches YOLOv8-N@640 (|gap| <= 1)'] = bool(abs(gap) <= 1.0)
    lat = os.path.join(ROOT, 'runs', 'latency', 'latency.json')
    if os.path.exists(lat):
        res['latency (ms, 1 CPU thread)'] = json.load(open(lat))
    print(json.dumps(res, indent=1))
    json.dump(res, open(os.path.join(ROOT, 'runs', 'logs', 'detector_compare.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
