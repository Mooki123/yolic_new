"""The research/06 foveation rule: reproduces the hand-sized Cityscapes variants and refuses asymmetric foveas."""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'tools'))
import fovea_views as F  # noqa: E402
from yolic_bench.layout import Layout  # noqa: E402


@pytest.mark.parametrize('s,total,ctx,fov', [(0.75, 52224, (160, 80), (352, 112)), (0.85, 54272, (128, 64), (384, 120)),
                                             (0.92, 57856, (96, 48), (416, 128))])
def test_rule_reproduces_cityscapes_variants(s, total, ctx, fov):
    v = F.derive(Layout.load(os.path.join(ROOT, 'layouts', 'cityscapes.json')), 'small_64x32', s, total)
    assert v['context'][1:] == ctx and v['fovea'][1:] == fov
    assert v['fovea'][0] == (512, 320, 1536, 640)


def test_rule_refuses_asymmetric_fovea():
    lay = Layout.load(os.path.join(ROOT, 'layouts', 'outdoor.json'))
    lay.groups['left'] = lay.groups['far_34px'][:4]  # the left end of the first far row only
    with pytest.raises(ValueError):
        F.derive(lay, 'left', 0.8, 50000)


def test_band_rule_reproduces_v5():
    v = F.derive_bands(Layout.load(os.path.join(ROOT, 'layouts', 'cityscapes.json')), 0.76, 51712)
    assert v['small_64x32'] == ((512, 320, 1536, 640), 352, 112)
    assert v['large_128x64'] == ((0, 640, 2048, 1024), 256, 48)
