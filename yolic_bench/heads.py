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


class MilPool(nn.Module):
    """research/08: layout-agnostic MIL head. 1x1 conv class maps, bilinearly upsampled to grid x grid, pooled over the
    pixels whose centres fall in each cell (max, log-sum-exp with sharpness r, or mean), plus a per-class bias only, so
    nothing is tied to a layout. set_layout() points the same trained head at any other layout."""

    def __init__(self, channels, layout, n_bits, pool='max', r=5.0, grid=112, dropout=0.2):
        super().__init__()
        if pool not in ('max', 'lse', 'mean'):
            raise ValueError(f'unknown MIL pooling {pool!r}')
        self.pool, self.r, self.grid, self.n_bits = pool, r, grid, n_bits
        self.drop = nn.Dropout(dropout)
        self.conv = nn.Conv2d(channels, n_bits, 1)
        self.bias = nn.Parameter(torch.zeros(n_bits))
        self.set_layout(layout)

    def set_layout(self, layout):
        idx = torch.as_tensor(layout.index_map(self.grid, self.grid).astype(np.int64)).flatten()
        count = torch.bincount(idx[idx >= 0], minlength=len(layout))
        if (count == 0).any():
            raise ValueError(f'{int((count == 0).sum())} cells contain no pixel centre on the {self.grid}x{self.grid} grid')
        self.n_cells = len(layout)
        self.register_buffer('pix_cell', idx.clamp_min(0), persistent=False)
        self.register_buffer('pix_in', idx >= 0, persistent=False)
        self.register_buffer('count', count.float(), persistent=False)
        return self

    def forward(self, x):
        z = self.conv(self.drop(x)).float()
        z = nn.functional.interpolate(z, size=(self.grid, self.grid), mode='bilinear', align_corners=False).flatten(2)
        b, k, _ = z.shape
        index = self.pix_cell.to(z.device).expand(b, k, -1)
        inside = self.pix_in.to(z.device)
        if self.pool == 'max':
            out = torch.full((b, k, self.n_cells), -1e4, device=z.device)
            out = out.scatter_reduce(2, index, z.masked_fill(~inside, -1e4), reduce='amax', include_self=True)
        elif self.pool == 'mean':
            out = torch.zeros(b, k, self.n_cells, device=z.device).scatter_add(2, index, z * inside)
            out = out / self.count.to(z.device)
        else:  # log-sum-exp, a smooth max: (1/r) log mean_p exp(r z_p)
            m = torch.full((b, k, self.n_cells), -1e4, device=z.device)
            m = m.scatter_reduce(2, index, z.masked_fill(~inside, -1e4), reduce='amax', include_self=True)
            d = (z - m.gather(2, index)).masked_fill(~inside, -30.0)  # mask before exp: outside pixels could overflow
            e = torch.exp(self.r * d) * inside
            s = torch.zeros(b, k, self.n_cells, device=z.device).scatter_add(2, index, e)
            out = m + torch.log(s / self.count.to(z.device)) / self.r
        return out.transpose(1, 2) + self.bias  # (B, n_cells, n_bits)


def _splat_matrix(num_pixels, num_bins):
    """RoIAlign-style bilinear weights (num_bins, num_pixels), as in YOLIC-Align's yolic_align._splat_matrix."""
    centers = (torch.arange(num_pixels, dtype=torch.float64) + 0.5) * num_bins / num_pixels - 0.5
    centers = centers.clamp(0, num_bins - 1)
    bins = torch.arange(num_bins, dtype=torch.float64)
    return (1 - (bins[:, None] - centers[None, :]).abs()).clamp(min=0)


def align_cell_weights(layout, in_w, in_h, feat_w, feat_h, crop=None, supersample=8):
    """Per-cell pooling weights on a feature grid for one view: exact cell coverage at the view's input resolution
    (Layout.rasterize, crop = the view's source box), spread onto the feature grid bilinearly, normalised per cell.
    Returns (weights (n_cells, feat_h * feat_w), covered (n_cells,) bool); uncovered cells get all-zero weights."""
    cov = torch.as_tensor(layout.rasterize(in_w, in_h, supersample=supersample, crop=crop), dtype=torch.float64)
    w = torch.einsum('fh,nhw,gw->nfg', _splat_matrix(in_h, feat_h), cov, _splat_matrix(in_w, feat_w))
    area = w.flatten(1).sum(1)
    covered = area > 1e-9
    w = torch.where(covered[:, None, None], w / area.clamp_min(1e-12)[:, None, None], torch.zeros_like(w))
    return w.flatten(1).float(), covered


class AlignTrunk(nn.Module):
    """YOLIC-Align's feature path: MobileNetV2 split at stride 16, the stride-32 map upsampled to stride 16 and
    concatenated with a 1x1-conv lateral of the stride-16 map."""

    def __init__(self, pretrained=True, lateral_channels=128, lateral=True):
        super().__init__()
        w = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        f = models.mobilenet_v2(weights=w).features
        self.stride16, self.stride32 = f[:14], f[14:]
        self.out_channels = 1280 + (lateral_channels if lateral else 0)
        self.lateral = nn.Sequential(nn.Conv2d(96, lateral_channels, 1, bias=False), nn.BatchNorm2d(lateral_channels),
                                     nn.ReLU6(inplace=True)) if lateral else None

    def forward(self, x):
        x16 = self.stride16(x)
        x32 = nn.functional.interpolate(self.stride32(x16), size=x16.shape[-2:], mode='bilinear', align_corners=False)
        return x32 if self.lateral is None else torch.cat([x32, self.lateral(x16)], dim=1)


class AlignNet(nn.Module):
    """research/13: YOLIC-Align's 'align' head (research/12), for one or several views.

    Per view: trunk (per-view BatchNorm, shared convolutions) -> shared dropout + 1x1 classifier -> per-location logits;
    each cell reads them through its bilinear mask weights in that view as mean + sigmoid(mix) * (LSE_T - mean).
    With several views, a cell's readouts are blended by a learned per-cell softmax over the views that cover it
    (equal at initialisation); then the per-(cell, class) bias is added. With one view this is exactly YOLIC-Align.
    """

    def __init__(self, layout, n_bits, view_specs, pretrained=True, dropout=0.2, lse_temperature=4.0,
                 readout='blend', lateral=True, cell_bias=True):
        """readout: 'blend' (YOLIC-Align), 'mean' or 'lse'; lateral / cell_bias: research/15 ablation switches."""
        super().__init__()
        if readout not in ('blend', 'mean', 'lse'):
            raise ValueError(f'unknown readout {readout!r}')
        self.readout_mode = readout
        trunk = AlignTrunk(pretrained, lateral=lateral)
        self.trunks = _per_view_copies(trunk, len(view_specs)) if len(view_specs) > 1 else nn.ModuleList([trunk])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Conv2d(trunk.out_channels, n_bits, 1)
        self.cell_bias = nn.Parameter(torch.zeros(len(layout), n_bits)) if cell_bias else None
        self.mix = nn.Parameter(torch.zeros(n_bits))
        self.view_logit = nn.Parameter(torch.zeros(len(layout), len(view_specs)))
        self.t = lse_temperature
        covered = []
        with torch.no_grad():
            for v, (crop, (w, h)) in enumerate(view_specs):
                fh, fw = trunk.stride16(torch.zeros(1, 3, h, w)).shape[-2:]
                wts, cov = align_cell_weights(layout, w, h, fw, fh, crop)
                self.register_buffer(f'w{v}', wts, persistent=False)
                self.register_buffer(f'logw{v}', torch.where(wts > 0, wts.log(), torch.zeros_like(wts)), persistent=False)
                self.register_buffer(f'in{v}', wts > 0, persistent=False)
                covered.append(cov)
        covered = torch.stack(covered, 1)
        if not covered.any(1).all():
            raise ValueError(f'{int((~covered.any(1)).sum())} cells are covered by no view')
        self.register_buffer('covered', covered, persistent=False)

    def readout(self, v, x):
        z = self.classifier(self.dropout(self.trunks[v](x))).float().flatten(2)  # (B, K, P)
        w, logw, inside = getattr(self, f'w{v}'), getattr(self, f'logw{v}'), getattr(self, f'in{v}')
        mean = torch.einsum('bkp,np->bnk', z, w)
        if self.readout_mode == 'mean':
            return mean
        s = (self.t * z.unsqueeze(2) + logw).masked_fill(~inside, -1e4)  # (B, K, N, P)
        lse = (torch.logsumexp(s, dim=-1) / self.t).transpose(1, 2)
        if self.readout_mode == 'lse':
            return lse
        return mean + torch.sigmoid(self.mix) * (lse - mean)  # (B, N, K)

    def forward(self, *views):
        r = torch.stack([self.readout(v, x) for v, x in enumerate(views)], dim=-1)  # (B, N, K, V)
        a = torch.softmax(self.view_logit.masked_fill(~self.covered, -1e9), dim=1)  # (N, V)
        out = (r * a[None, :, None, :]).sum(-1)
        return out if self.cell_bias is None else out + self.cell_bias


def _per_view_copies(features, n_views):
    """n_views copies of a backbone that share every weight except BatchNorm (each copy has its own BN).

    Views of different image regions (road band, sky band) have different statistics. With one set of BN
    layers, training normalises each view with its own batch statistics but evaluation uses running
    statistics averaged over all views, so evaluation sees mis-normalised features (research/06 addendum).
    """
    import copy
    copies = [features]
    for _ in range(n_views - 1):
        c = copy.deepcopy(features)
        for (_, src), (_, dst) in zip(features.named_modules(), c.named_modules()):
            if not isinstance(dst, nn.modules.batchnorm._BatchNorm):
                for pname, p in list(dst.named_parameters(recurse=False)):
                    setattr(dst, pname, getattr(src, pname))  # tie to the first copy's Parameter
        copies.append(c)
    return nn.ModuleList(copies)


class FoveaGapFc(nn.Module):
    """F (research/05-06): one backbone runs on every view (e.g. crops of each cell-group band); the views'
    global-average-pooled features are concatenated and one Linear maps them to every (cell, bit), like GapFc.

    per_view_bn=True (default) gives each view its own BatchNorm layers while every other weight is shared.
    per_view_bn=False is the original shared-BN version, kept only to reproduce the invalid research/06 runs.
    """

    def __init__(self, features, channels, n_views, n_cells, n_bits, dropout=0.2, per_view_bn=True):
        super().__init__()
        self.n_cells, self.n_bits, self.per_view_bn = n_cells, n_bits, per_view_bn
        if per_view_bn:
            self.views = _per_view_copies(features, n_views)
        else:
            self.features = features
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(channels * n_views, n_cells * n_bits)

    def forward(self, *views):
        nets = self.views if self.per_view_bn else [self.features] * len(views)
        z = torch.cat([net(v).mean(dim=(2, 3)) for net, v in zip(nets, views)], dim=1)
        return self.fc(self.drop(z)).view(-1, self.n_cells, self.n_bits)


def build(cfg, layout, n_bits, input_size=224, pretrained=True, n_views=1, view_specs=None):
    """cfg: dict with backbone, head and (cellpool) stride. Returns a CellNet, SegCells, FoveaGapFc or AlignNet.
    view_specs: [(crop box in source pixels or None, (width, height))] per input view (needed by 'align')."""
    if cfg['head'] == 'align':
        specs = view_specs or [(None, (input_size, input_size))] * n_views
        return AlignNet(layout, n_bits, specs, pretrained, readout=cfg.get('readout', 'blend'),
                        lateral=cfg.get('lateral', True), cell_bias=cfg.get('cell_bias', True))
    if cfg['head'] == 'fovea_gapfc':
        feats, channels = backbone(cfg['backbone'], pretrained)
        return FoveaGapFc(feats, channels, n_views, len(layout), n_bits, per_view_bn=cfg.get('per_view_bn', True))
    if n_views != 1:
        raise ValueError(f"head {cfg['head']!r} takes one view, got {n_views}")
    if cfg['head'] == 'lraspp':
        return SegCells(cfg.get('size', 'small'), layout, n_bits, input_size, pretrained)
    stride = int(cfg.get('stride', 32))
    feats, channels = backbone(cfg['backbone'], pretrained, stride)
    if cfg['head'] == 'gapfc':
        head = GapFc(channels, len(layout), n_bits)
    elif cfg['head'] == 'mil':
        head = MilPool(channels, layout, n_bits, pool=cfg.get('pool', 'max'), r=float(cfg.get('r', 5.0)))
    elif cfg['head'] == 'cellpool':
        side = input_size // stride
        head = CellPool(channels, layout.rasterize(side, side, supersample=COVERAGE_SUPERSAMPLE, crop=cfg.get('crop')), n_bits)
    else:
        raise ValueError(f"unknown head {cfg['head']!r}")
    return CellNet(feats, head)
