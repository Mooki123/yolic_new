"""GPU-side augmentation and heads: checked against torchvision and the layout geometry."""
import os

import numpy as np
import pytest
import torch
import torchvision.transforms.functional as TF

from yolic_bench import data as D
from yolic_bench import heads
from yolic_bench.layout import Layout

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope='module')
def outdoor():
    return Layout.load(os.path.join(ROOT, 'layouts', 'outdoor.json'))


def test_hsv_roundtrip_and_hue_matches_torchvision():
    x = torch.rand(4, 3, 16, 16)
    h, s, v = D._rgb_to_hsv(x)
    assert torch.allclose(D._hsv_to_rgb(h, s, v), x, atol=1e-5)
    for shift in (-0.1, 0.03, 0.1):
        ours = D._hsv_to_rgb(torch.fmod(h + shift + 1.0, 1.0), s, v)
        ref = torch.stack([TF.adjust_hue(xi, shift) for xi in x])
        assert torch.allclose(ours, ref, atol=1e-5)


def test_single_ops_match_torchvision():
    x = torch.rand(3, 3, 16, 16)
    f = torch.tensor([0.6, 1.0, 1.4]).view(-1, 1, 1, 1)
    gray = D._gray(x)
    assert torch.allclose((x * f).clamp(0, 1), torch.stack([TF.adjust_brightness(xi, float(fi)) for xi, fi in zip(x, f)]), atol=1e-6)
    ours = (f * x + (1 - f) * gray.mean((2, 3), keepdim=True)).clamp(0, 1)
    assert torch.allclose(ours, torch.stack([TF.adjust_contrast(xi, float(fi)) for xi, fi in zip(x, f)]), atol=1e-5)
    ours = (f * x + (1 - f) * gray).clamp(0, 1)
    assert torch.allclose(ours, torch.stack([TF.adjust_saturation(xi, float(fi)) for xi, fi in zip(x, f)]), atol=1e-5)


def test_hflip_moves_content_with_labels(outdoor):
    """Paint cell 0 white at 224x224; after a forced flip the paint lies in cell perm-partner's footprint."""
    perm = outdoor.mirror_permutation()
    cov = outdoor.rasterize(224, 224)
    img = torch.zeros(1, 224, 224, 3, dtype=torch.uint8)
    img[0][torch.from_numpy(cov[0] > 0.5)] = 255
    lab = torch.zeros(1, len(outdoor), 12)
    lab[0, 0, 0] = 1
    aug = D.GpuAugment(perm, 'hflip', None, normalize=False)
    torch.manual_seed(0)
    aug.g = torch.Generator().manual_seed(0)
    while True:  # draw until the flip fires
        x, y, _, _ = aug.train(img, lab)
        if y[0, 0, 0] == 0:
            break
    j = int(np.nonzero(y[0, :, 0].numpy())[0][0])
    painted = x[0, 0].numpy() > 0.5
    assert painted[cov[j] > 0.5].all() and painted.sum() == (cov[j] > 0.5).sum()


def test_cellpool_equals_cell_average_of_maps(outdoor):
    cov = outdoor.rasterize(7, 7, supersample=heads.COVERAGE_SUPERSAMPLE)
    head = heads.CellPool(8, cov, 12, dropout=0.0)
    x = torch.randn(2, 8, 7, 7)
    out = head(x)
    maps = head.conv(x)
    w = torch.from_numpy(cov / cov.sum(axis=(1, 2), keepdims=True))
    ref = torch.einsum('bkhw,chw->bck', maps, w.float())
    assert out.shape == (2, len(outdoor), 12)
    assert torch.allclose(out, ref, atol=1e-5)


def test_mnv2_stride16_grid():
    f, _ = heads.backbone('mnv2', pretrained=False, stride=16)
    assert f(torch.zeros(1, 3, 224, 224)).shape[-2:] == (14, 14)


@pytest.mark.parametrize('name', ['cityscapes', 'outdoor', 'indoor'])
@pytest.mark.parametrize('side', [7, 14])
def test_cellpool_coverage_reaches_every_cell(name, side):
    """Regression: at supersample 4, 48 Cityscapes small cells had zero coverage on the 7x7 grid."""
    lay = Layout.load(os.path.join(ROOT, 'layouts', name + '.json'))
    cov = lay.rasterize(side, side, supersample=heads.COVERAGE_SUPERSAMPLE)
    area = cov.sum(axis=(1, 2)) * (lay.width / side) * (lay.height / side)  # in source pixels
    true = np.bincount(lay.index_map()[lay.index_map() >= 0], minlength=len(lay))
    assert (cov.sum(axis=(1, 2)) > 0).all()
    assert np.allclose(area, true, rtol=0.05)
    heads.CellPool(8, cov, 4)  # must not raise


def test_views_share_flip_and_jitter(outdoor):
    """Identical views must come out identical: one flip draw and one jitter draw per sample."""
    img = torch.randint(0, 256, (6, 32, 48, 3), dtype=torch.uint8)
    lab = torch.zeros(6, len(outdoor), 12)
    jit = dict(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)
    aug = D.GpuAugment(outdoor.mirror_permutation(), 'hflip', jit, True, torch.Generator().manual_seed(1))
    (a, b), _, _, _ = aug.train([img, img.clone()], lab)
    assert torch.equal(a, b)
    single, _, _, _ = D.GpuAugment(outdoor.mirror_permutation(), 'hflip', jit, True,
                                   torch.Generator().manual_seed(1)).train(img, lab)
    assert torch.equal(single, a)  # a one-view list behaves exactly like the plain tensor path


def test_fovea_gapfc_shape(outdoor):
    m = heads.build({'backbone': 'mnv2', 'head': 'fovea_gapfc'}, outdoor, 12, pretrained=False, n_views=2).eval()
    out = m(torch.zeros(2, 3, 112, 224), torch.zeros(2, 3, 96, 320))
    assert out.shape == (2, len(outdoor), 12)


def test_per_view_bn_shares_everything_but_batchnorm(outdoor):
    m = heads.build({'backbone': 'mnv2', 'head': 'fovea_gapfc'}, outdoor, 12, pretrained=False, n_views=3)
    a, b = m.views[0], m.views[2]
    conv_a, conv_b = a[0][0], b[0][0]
    assert conv_a.weight is conv_b.weight  # convolutions tied
    bn_a, bn_b = a[0][1], b[0][1]
    assert bn_a.weight is not bn_b.weight and bn_a.running_mean is not bn_b.running_mean  # BN separate
    shared = heads.build({'backbone': 'mnv2', 'head': 'fovea_gapfc', 'per_view_bn': False}, outdoor, 12,
                         pretrained=False, n_views=3)
    n = lambda mod: sum(p.numel() for p in mod.parameters())  # parameters() de-duplicates tied weights
    bn = sum(p.numel() for mm in shared.features.modules() if isinstance(mm, torch.nn.BatchNorm2d) for p in mm.parameters())
    assert n(m) == n(shared) + 2 * bn


def test_per_view_bn_keeps_each_views_statistics(outdoor):
    """Each view's first BatchNorm must track only that view's activations (the shared-BN bug mixed them)."""
    torch.manual_seed(0)
    m = heads.build({'backbone': 'mnv2', 'head': 'fovea_gapfc'}, outdoor, 12, pretrained=False, n_views=2)
    xs = (torch.randn(8, 3, 64, 96) * 3 + 2, torch.randn(8, 3, 32, 128) * 0.2 - 1)  # very different statistics
    for mod in m.modules():
        if isinstance(mod, torch.nn.BatchNorm2d):
            mod.momentum = None  # cumulative average: running_mean = batch mean after one pass
    m.train()
    with torch.no_grad():
        m(*xs)
        for net, x in zip(m.views, xs):
            conv, bn = net[0][0], net[0][1]
            assert torch.allclose(bn.running_mean, conv(x).mean(dim=(0, 2, 3)), atol=1e-4)
