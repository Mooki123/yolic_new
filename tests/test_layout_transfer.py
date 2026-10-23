"""research/08: the MIL head pools correctly and can be pointed at any of the transfer layouts."""
import os

import numpy as np
import pytest
import torch

from yolic_bench import heads
from yolic_bench.layout import Layout

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAMES = ['cityscapes', 'cityscapes_union', 'cityscapes_shift', 'cityscapes_fine']


@pytest.mark.parametrize('pool', ['max', 'lse', 'mean'])
def test_mil_pooling_matches_brute_force(pool):
    lay = Layout.load(os.path.join(ROOT, 'layouts', 'cityscapes_shift.json'))
    head = heads.MilPool(8, lay, 4, pool=pool, dropout=0.0).eval()
    x = torch.randn(2, 8, 14, 14)
    out = head(x)
    z = torch.nn.functional.interpolate(head.conv(x), size=(112, 112), mode='bilinear', align_corners=False).flatten(2)
    idx = torch.as_tensor(lay.index_map(112, 112).astype(np.int64)).flatten()
    for c in (0, 77, len(lay) - 1):
        v = z[:, :, idx == c]
        ref = {'max': v.max(-1).values, 'mean': v.mean(-1),
               'lse': torch.logsumexp(5.0 * v, -1) / 5.0 - np.log(v.shape[-1]) / 5.0}[pool] + head.bias
        assert torch.allclose(out[:, c], ref, atol=1e-4)


@pytest.mark.parametrize('name', NAMES)
def test_every_layout_reaches_every_cell(name):
    lay = Layout.load(os.path.join(ROOT, 'layouts', name + '.json'))
    head = heads.MilPool(8, Layout.load(os.path.join(ROOT, 'layouts', 'cityscapes.json')), 4)
    head.set_layout(lay)  # raises if a cell has no pixel centre on the grid
    assert head(torch.randn(1, 8, 14, 14)).shape == (1, len(lay), 4)


def test_mil_model_builds_at_stride_16():
    lay = Layout.load(os.path.join(ROOT, 'layouts', 'cityscapes.json'))
    m = heads.build({'backbone': 'mnv2', 'head': 'mil', 'pool': 'lse', 'stride': 16}, lay, 4, pretrained=False).eval()
    assert m(torch.zeros(1, 3, 224, 224)).shape == (1, 256, 4)


@pytest.mark.parametrize('pool', ['max', 'lse', 'mean'])
def test_mil_pooling_is_finite_with_large_logits(pool):
    """Regression: LSE overflowed (inf * 0 = NaN) when a pixel outside every cell had a large logit."""
    lay = Layout.load(os.path.join(ROOT, 'layouts', 'cityscapes.json'))
    head = heads.MilPool(8, lay, 4, pool=pool, dropout=0.0)
    x = torch.randn(2, 8, 14, 14, requires_grad=True)
    out = head(x * 50)
    out.sum().backward()
    assert torch.isfinite(out).all() and torch.isfinite(x.grad).all()
