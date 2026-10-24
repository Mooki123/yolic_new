"""research/14: CPU latency (laptop proxy) of the cell models and YOLOv8-N, ONNX Runtime, 1 thread, batch 1.

50 warm-up + 500 timed runs per model; median and p90 in ms. Models only (detector NMS excluded). Weights: seed 0 runs.

    python tools/latency.py
"""
import json
import os
import sys
import time

import numpy as np
import onnxruntime as ort
import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from yolic_bench import heads  # noqa: E402
from yolic_bench.layout import Layout  # noqa: E402

OUT = os.path.join(ROOT, 'runs', 'latency')
CELL_MODELS = {'A (YOLIC)': 'cityscapes_b1', 'AL (Align)': 'cityscapes_align', 'F (foveated)': 'cityscapes_f_fovea_pvbn',
               'F+AL': 'cityscapes_f_align'}


def time_onnx(path, shapes, warmup=50, runs=500):
    opt = ort.SessionOptions()
    opt.intra_op_num_threads = opt.inter_op_num_threads = 1
    opt.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess = ort.InferenceSession(path, opt, providers=['CPUExecutionProvider'])
    feed = {i.name: np.random.rand(*s).astype(np.float32) for i, s in zip(sess.get_inputs(), shapes)}
    for _ in range(warmup):
        sess.run(None, feed)
    t = []
    for _ in range(runs):
        s = time.perf_counter()
        sess.run(None, feed)
        t.append((time.perf_counter() - s) * 1000)
    return round(float(np.median(t)), 2), round(float(np.percentile(t, 90)), 2)


def cell_model(name):
    run = os.path.join(ROOT, 'runs', name, 'seed0')
    cfg = yaml.safe_load(open(os.path.join(run, 'config.yaml')))
    meta = json.load(open(os.path.join(os.path.expanduser(cfg['cache']), 'meta.json'))).get('views', {})
    views = cfg.get('views') or [None]
    specs = [((tuple(meta[v]['crop']), tuple(meta[v]['size'])) if v else (None, (224, 224))) for v in views]
    m = heads.build(cfg['model'], Layout.load(os.path.join(ROOT, cfg['layout'])), 4, pretrained=False,
                    n_views=len(views), view_specs=specs).eval()
    m.load_state_dict(torch.load(os.path.join(run, 'best.pt'), map_location='cpu'))
    return m, [(1, 3, h, w) for _, (w, h) in specs]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--cells-only', action='store_true', help='skip the YOLOv8-N models (no Ultralytics needed)')
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    prev = os.path.join(OUT, 'latency.json')
    torch.set_num_threads(1)
    res = json.load(open(prev)) if os.path.exists(prev) else {}
    for label, name in CELL_MODELS.items():
        m, shapes = cell_model(name)
        path = os.path.join(OUT, name + '.onnx')
        torch.onnx.export(m, tuple(torch.zeros(s) for s in shapes), path, opset_version=17, dynamo=False,
                          input_names=[f'view{i}' for i in range(len(shapes))])
        res[label] = dict(zip(('median_ms', 'p90_ms'), time_onnx(path, shapes)), input=[s[2:] for s in shapes])
        print(label, res[label], flush=True)
    if args.cells_only:
        json.dump(res, open(prev, 'w'), indent=1)
        return
    from ultralytics import YOLO
    for imgsz, hw in ((224, (128, 224)), (640, (320, 640))):  # letterboxed 2:1 frame, padded to a stride-32 multiple
        weights = os.path.join(ROOT, 'runs', 'yolo', f'yolov8n_{imgsz}_seed0', 'weights', 'best.pt')
        path = YOLO(weights).export(format='onnx', imgsz=list(hw), opset=17, dynamic=False, simplify=False)
        label = f'YOLOv8-N@{imgsz}'
        res[label] = dict(zip(('median_ms', 'p90_ms'), time_onnx(path, [(1, 3, *hw)])), input=[hw])
        print(label, res[label], flush=True)
    json.dump(res, open(os.path.join(OUT, 'latency.json'), 'w'), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
