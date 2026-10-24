"""research/14: train YOLOv8-N on Cityscapes boxes and score it with the shared per-cell metric.

    python tools/yolo_baseline.py --imgsz 224 --seed 0      # train (resumes / skips if finished), then score
Writes runs/yolo/yolov8n_<imgsz>_seed<k>/ (Ultralytics run) and runs/yolo_cells/yolov8n_<imgsz>/seed<k>/
{val,test}_probs.npy + metrics.json in the same format as yolic_bench.train, so the comparison tools read both.
"""
import argparse
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

DATA = os.path.expanduser('~/data/yolo_cityscapes')
PV = [0, 1]


def cell_probs(model, keys, split, imgsz, rast):
    out = np.zeros((len(keys), rast.n_cells, 4), dtype=np.float32)
    paths = [os.path.join(DATA, 'images', split, k.replace('/', '__') + '.jpg') for k in keys]
    for i in range(0, len(paths), 32):
        for j, r in enumerate(model.predict(paths[i:i + 32], imgsz=imgsz, conf=0.001, iou=0.7, max_det=300,
                                            verbose=False, stream=False)):
            b = r.boxes
            xyxy = b.xyxy.cpu().numpy() * 2.0  # 1024x512 image -> 2048x1024 source pixels
            s = rast.scores([(int(c), *box, float(p)) for c, box, p in
                             zip(b.cls.cpu().numpy(), xyxy, b.conf.cpu().numpy())])
            out[i + j, :, :2], out[i + j, :, 3] = s[:, :2], s[:, 2]  # People, Vehicle, (Other unused), background
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--imgsz', type=int, required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=150)
    ap.add_argument('--out', default=os.path.join(ROOT, 'runs'), help='root for runs/yolo and runs/yolo_cells (smoke tests)')
    args = ap.parse_args()
    from ultralytics import YOLO

    name = f'yolov8n_{args.imgsz}_seed{args.seed}'
    run = os.path.join(args.out, 'yolo', name)
    best = os.path.join(run, 'weights', 'best.pt')
    if not os.path.exists(best):
        last = os.path.join(run, 'weights', 'last.pt')
        if os.path.exists(last):
            YOLO(last).train(resume=True)
        else:
            YOLO(os.path.join(ROOT, 'runs', 'yolo', 'yolov8n.pt')).train(data=os.path.join(DATA, 'data.yaml'), imgsz=args.imgsz, epochs=args.epochs,
                                     seed=args.seed, deterministic=True, workers=6, plots=False,
                                     project=os.path.join(args.out, 'yolo'), name=name, exist_ok=True)
    out_dir = os.path.join(args.out, 'yolo_cells', f'yolov8n_{args.imgsz}', f'seed{args.seed}')
    if os.path.exists(os.path.join(out_dir, 'metrics.json')):
        print(out_dir, 'already scored')
        return
    os.makedirs(out_dir, exist_ok=True)
    model = YOLO(best)
    box = model.val(data=os.path.join(DATA, 'data.yaml'), split='test', imgsz=args.imgsz, plots=False, verbose=False)

    c = CellCache(os.path.expanduser('~/data/yolic_cache/cityscapes'))
    rast = BoxRasteriser(Layout.load(os.path.join(ROOT, 'layouts', 'cityscapes.json')))
    res = {'name': f'yolov8n_{args.imgsz}', 'seed': args.seed, 'imgsz': args.imgsz,
           'box_mAP50': float(box.box.map50), 'box_mAP50_95': float(box.box.map)}
    probs = {}
    for split in ('val', 'test'):
        keys = open(os.path.join(ROOT, 'splits', 'cityscapes', f'{split}.txt')).read().split()
        probs[split] = cell_probs(model, keys, split, args.imgsz, rast)
        np.save(os.path.join(out_dir, f'{split}_probs.npy'), probs[split].astype(np.float16))
    va, te = (c.indices(os.path.join(ROOT, 'splits', 'cityscapes', f'{s}.txt')) for s in ('val', 'test'))
    gt, valid = c.labels[te], c.valid[te]
    thr = M.tune_thresholds(c.labels[va], probs['val'], PV, valid=c.valid[va])
    res['primary_macro_f1'] = float(M.macro_f1(M.counts(gt, M.binarize(probs['test']), valid), PV))
    res['tuned'] = {'thresholds': thr.tolist(),
                    'primary_macro_f1': float(M.macro_f1(M.counts(gt, M.binarize(probs['test'], thr), valid), PV))}
    for g, cells in Layout.load(os.path.join(ROOT, 'layouts', 'cityscapes.json')).groups.items():
        res.setdefault('by_group_primary', {})[g] = {'object_macro_f1': float(
            M.macro_f1(M.counts(gt, M.binarize(probs['test'], thr), valid, cells), PV))}
    json.dump(res, open(os.path.join(out_dir, 'metrics.json'), 'w'), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
