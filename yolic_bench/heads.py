"""Backbones and cell-presence heads (plan 04, section 2.4).

Every model maps an image batch (B, 3, H, W) to logits (B, n_cells, n_bits), where n_bits = M + 1
(M object classes plus the background bit), the same layout the released 1248-wide Linear produces
after a reshape.

  gapfc     YOLIC (B1/B2/B7): global average pool -> Dropout(0.2) -> Linear(C, n_cells * n_bits)
  cellpool  B4/B5/B6: Dropout(0.2) -> 1x1 conv to n_bits class maps -> area-weighted mean of each
            map over each cell's footprint on the feature grid -> + a per-(cell, class) bias
"""
import numpy as np
import torch
import torch.nn as nn
from torchvision import models


def _dilate_mnv2_last_stages(features):
    """Output stride 32 -> 16: remove the stride of features[14] and dilate every later 3x3 conv by 2."""
    for idx in range(14, len(features)):
        for m in features[idx].modules():
            if isinstance(m, nn.Conv2d) and m.kernel_size == (3, 3):
                if idx == 14:
                    m.stride = (1, 1)
                else:
                    m.dilation, m.padding = (2, 2), (2, 2)
    return features


def backbone(name, pretrained=True, stride=32):
    """Feature extractor and its channel count."""
    if name == 'mnv2':
        w = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        f = models.mobilenet_v2(weights=w).features
        if stride == 16:
            f = _dilate_mnv2_last_stages(f)
        return f, 1280
    if stride != 32:
        raise ValueError(f'stride {stride} is only implemented for mnv2')
    if name == 'shufflenetv2':
        w = models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.shufflenet_v2_x1_0(weights=w)
        return nn.Sequential(m.conv1, m.maxpool, m.stage2, m.stage3, m.stage4, m.conv5), 1024
    if name == 'mnv3s':
        w = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        return models.mobilenet_v3_small(weights=w).features, 576
    raise ValueError(f'unknown backbone {name!r}')


class GapFc(nn.Module):
    def __init__(self, channels, n_cells, n_bits, dropout=0.2):
        super().__init__()
        self.n_cells, self.n_bits = n_cells, n_bits
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(channels, n_cells * n_bits)

    def forward(self, x):
        x = self.drop(x.mean(dim=(2, 3)))
        return self.fc(x).view(-1, self.n_cells, self.n_bits)


class CellPool(nn.Module):
    """Area-weighted pooling of 1x1-conv class maps over each cell (the CAM identity, review E1).

    coverage: (n_cells, h, w) fraction of each feature pixel covered by each cell, from
    Layout.rasterize(w, h, supersample=COVERAGE_SUPERSAMPLE). Rows are normalised to sum to 1.
    A cell with zero coverage would be scored from its bias alone, so that is an error.
    """

    def __init__(self, channels, coverage, n_bits, dropout=0.2):
        super().__init__()
        cov = torch.as_tensor(np.asarray(coverage), dtype=torch.float32)
        empty = torch.nonzero(cov.flatten(1).sum(1) == 0).flatten().tolist()
        if empty:
            raise ValueError(f'{len(empty)} cells have no coverage on the {tuple(cov.shape[1:])} grid: {empty[:10]}')
        cov = cov / cov.flatten(1).sum(1).clamp_min(1e-12)[:, None, None]
        self.register_buffer('weights', cov.flatten(1))  # (n_cells, h * w)
        self.grid = tuple(cov.shape[1:])
        self.drop = nn.Dropout(dropout)
        self.conv = nn.Conv2d(channels, n_bits, 1)
        self.bias = nn.Parameter(torch.zeros(cov.shape[0], n_bits))

    def forward(self, x):
        if tuple(x.shape[2:]) != self.grid:
            raise ValueError(f'feature grid {tuple(x.shape[2:])} != rasterised grid {self.grid}')
        maps = self.conv(self.drop(x)).flatten(2)  # (B, n_bits, h * w)
        return torch.einsum('bkp,cp->bck', maps, self.weights.to(maps.dtype)) + self.bias


# Sub-samples per feature pixel and axis for CellPool coverage. Plan 04 says 4, but at stride 32 a
# 4x4 grid samples a 2048x1024 frame every 73x37 px, so 48 of the 160 64x32 Cityscapes cells got no
# sample at all (zero weight: bias-only logits). 64 samples every ~5x2 px (cell areas within 2.2%).
COVERAGE_SUPERSAMPLE = 64


class CellNet(nn.Module):
    def __init__(self, features, head):
        super().__init__()
        self.features, self.head = features, head

    def forward(self, x):
        return self.head(self.features(x))


class SegCells(nn.Module):
    """B8: LR-ASPP on a dilated MobileNetV3 (torchvision's _lraspp_mobilenetv3 recipe, any size).

    forward() gives pixel logits (B, n_groups, H, W) for a softmax over the object groups plus the
    background group (void pixels are ignored in the loss). cell_probs() reads cells out: an object
    group's cell score is the max of its pixel probability inside the cell (the '>= 1 px' rule), and
    the background bit is 1 - the largest object score.
    """
    seg = True

    def __init__(self, size, layout, n_bits, input_size=224, pretrained=True):
        super().__init__()
        from torchvision.models.segmentation.lraspp import LRASPP
        from torchvision.models._utils import IntermediateLayerGetter
        if size == 'small':
            w = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
            bb = models.mobilenet_v3_small(weights=w, dilated=True).features
        else:
            w = models.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
            bb = models.mobilenet_v3_large(weights=w, dilated=True).features
        stage = [0] + [i for i, b in enumerate(bb) if getattr(b, '_is_cn', False)] + [len(bb) - 1]
        low, high = stage[-4], stage[-1]
        bb = IntermediateLayerGetter(bb, return_layers={str(low): 'low', str(high): 'high'})
        self.net = LRASPP(bb, bb[str(low)].out_channels, bb[str(high)].out_channels, n_bits)
        idx = torch.as_tensor(layout.index_map(input_size, input_size).astype(np.int64)).flatten()
        self.register_buffer('pix_cell', idx.clamp_min(0))
        self.register_buffer('pix_in', idx >= 0)
        self.n_cells, self.n_bits = len(layout), n_bits

    def forward(self, x):
        return self.net(x)['out']

    def cell_probs(self, logits):
        p = logits.float().softmax(1).flatten(2)[:, :-1]  # object groups, (B, M, H*W)
        p = p * self.pix_in
        out = torch.zeros(p.shape[0], p.shape[1], self.n_cells, device=p.device)
        out = out.scatter_reduce(2, self.pix_cell.expand_as(p), p, reduce='amax', include_self=True)
        out = out.transpose(1, 2)  # (B, n_cells, M)
        return torch.cat([out, 1 - out.max(dim=2, keepdim=True).values], dim=2)


class FoveaGapFc(nn.Module):
    """F (research/05): one shared backbone runs on every view (e.g. a low-res full frame and a high-res
    crop of the small-cell band); the views' global-average-pooled features are concatenated and one
    Linear maps them to every (cell, bit), exactly like GapFc."""

    def __init__(self, features, channels, n_views, n_cells, n_bits, dropout=0.2):
        super().__init__()
        self.features, self.n_cells, self.n_bits = features, n_cells, n_bits
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(channels * n_views, n_cells * n_bits)

    def forward(self, *views):
        z = torch.cat([self.features(v).mean(dim=(2, 3)) for v in views], dim=1)
        return self.fc(self.drop(z)).view(-1, self.n_cells, self.n_bits)


def build(cfg, layout, n_bits, input_size=224, pretrained=True, n_views=1):
    """cfg: dict with backbone, head and (cellpool) stride. Returns a CellNet, SegCells or FoveaGapFc."""
    if cfg['head'] == 'fovea_gapfc':
        feats, channels = backbone(cfg['backbone'], pretrained)
        return FoveaGapFc(feats, channels, n_views, len(layout), n_bits)
    if n_views != 1:
        raise ValueError(f"head {cfg['head']!r} takes one view, got {n_views}")
    if cfg['head'] == 'lraspp':
        return SegCells(cfg.get('size', 'small'), layout, n_bits, input_size, pretrained)
    stride = int(cfg.get('stride', 32))
    feats, channels = backbone(cfg['backbone'], pretrained, stride)
    if cfg['head'] == 'gapfc':
        head = GapFc(channels, len(layout), n_bits)
    elif cfg['head'] == 'cellpool':
        side = input_size // stride
        head = CellPool(channels, layout.rasterize(side, side, supersample=COVERAGE_SUPERSAMPLE, crop=cfg.get('crop')), n_bits)
    else:
        raise ValueError(f"unknown head {cfg['head']!r}")
    return CellNet(feats, head)
