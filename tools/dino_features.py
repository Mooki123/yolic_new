"""research/11 idea 1: cache frozen DINOv2 ViT-S/14 features for every outdoor frame and its horizontal flip.

Input: the same 224x224 squashed frames as B1 (cache images.npy), ImageNet normalisation.
Writes dino_cls.npy (2, N, 384) and dino_grid.npy (2, N, 384, 8, 8) in float16; index 0 = original, 1 = flipped.
The grid is the 16x16 patch-token map average-pooled to 8x8.

    python tools/dino_features.py
"""
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from yolic_bench.data import CellCache, GpuAugment, batches  # noqa: E402

CACHE = os.path.expanduser('~/data/yolic_cache/outdoor')


@torch.no_grad()
def main():
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', trust_repo=True).cuda().eval()
    c = CellCache(CACHE)
    n = len(c.names)
    cls = np.lib.format.open_memmap(os.path.join(CACHE, 'dino_cls.npy'), 'w+', np.float16, (2, n, 384))
    grid = np.lib.format.open_memmap(os.path.join(CACHE, 'dino_grid.npy'), 'w+', np.float16, (2, n, 384, 8, 8))
    aug = GpuAugment(None, 'none', None, True)
    for imgs, _, _, _, pos in batches(c, np.arange(n), 128, shuffle=False):
        x = aug.to_input(imgs[0].cuda()).contiguous()
        for f, xx in enumerate((x, x.flip(-1))):
            with torch.autocast('cuda', dtype=torch.float16):
                o = model.forward_features(xx)
            p = o['x_norm_patchtokens'].float().transpose(1, 2).reshape(-1, 384, 16, 16)
            cls[f, pos] = o['x_norm_clstoken'].float().cpu().numpy()
            grid[f, pos] = torch.nn.functional.avg_pool2d(p, 2).cpu().numpy()
        if pos[0] % 4096 < 128:
            print(f'{pos[0]}/{n}', flush=True)
    cls.flush()
    grid.flush()
    print('done', n)


if __name__ == '__main__':
    main()
