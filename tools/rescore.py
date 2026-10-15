"""Recompute the per-group breakdown of finished runs from their saved test_probs.npy (no GPU).

    python tools/rescore.py runs/outdoor_b1_random/seed0 [more run dirs ...]
"""
import json
import os
import sys

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from yolic_bench import metrics as M  # noqa: E402
from yolic_bench.data import CellCache  # noqa: E402
from yolic_bench.layout import Layout  # noqa: E402
from yolic_bench.train import by_group  # noqa: E402

for run in sys.argv[1:]:
    cfg = yaml.safe_load(open(os.path.join(run, 'config.yaml')))
    cache = CellCache(os.path.expanduser(cfg['cache']))
    gt = cache.labels[cache.indices(os.path.join(ROOT, cfg['split'], 'test.txt'))]
    probs = np.load(os.path.join(run, 'test_probs.npy')).astype(np.float32)
    layout = Layout.load(os.path.join(ROOT, cfg['layout']))
    path = os.path.join(run, 'metrics.json')
    m = json.load(open(path))
    m['by_group'] = by_group(gt, M.binarize(probs), layout, cfg['object_classes'])
    json.dump(m, open(path, 'w'), indent=1)
    print(run, json.dumps(m['by_group']))
