"""Cached cell datasets and GPU-side augmentation (plan 04, sections 2.3 and 4.1).

A cache directory (tools/cache_outdoor.py) holds images.npy (N, S, S, 3) uint8 as a memmap,
labels.npy (N, n_cells, n_bits) uint8, names.txt and optionally valid.npy (N, n_cells) bool for void
masking. Batches are gathered from the memmap by a background thread and augmented on the GPU, so
there are no DataLoader worker processes (the Windows-spawn RAM problem of the released scripts).

Augmentation (training only, never on val/test):
  flip    'hflip': mirror the image left-right and permute cells with Layout.mirror_permutation.
          'compat_vflip': the released bug (D1): image upside-down, labels mirrored left-right.
          'none'.
  jitter  torchvision ColorJitter semantics (brightness/contrast/saturation factors drawn from
          [1 - v, 1 + v], hue shift from [-h, h]) with an independent draw per image. The order of
          the four ops is shuffled per batch rather than per image.
"""
import os
import queue
import threading

import numpy as np
import torch

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class CellCache:
    """views: names of extra image views (images_<view>.npy, tools/cache_views.py); None = images.npy."""

    def __init__(self, path, views=None, labels=None):
        """labels: name of an alternative label set (labels_<name>.npy / valid_<name>.npy, e.g. another layout)."""
        self.path = path
        files = ['images.npy'] if not views else [f'images_{v}.npy' for v in views]
        self.views = [np.load(os.path.join(path, f), mmap_mode='r') for f in files]
        self.images = self.views[0]
        suffix = '' if not labels else f'_{labels}'
        self.labels = np.load(os.path.join(path, f'labels{suffix}.npy'))
        vp = os.path.join(path, f'valid{suffix}.npy')
        self.valid = np.load(vp) if os.path.exists(vp) else None
        mp = os.path.join(path, 'masks.npy')
        self.masks = np.load(mp, mmap_mode='r') if os.path.exists(mp) else None
        ap = os.path.join(path, 'aug.npy')
        self.aug = np.load(ap) if os.path.exists(ap) else None
        self.names = open(os.path.join(path, 'names.txt')).read().split()
        self.index = {n: i for i, n in enumerate(self.names)}

    def indices(self, split_file):
        return np.array([self.index[n] for n in open(split_file).read().split()], dtype=np.int64)

    def sample_states(self, idx, generator):
        """Per epoch: keep each train row's plain state with p = 0.5, else take one of its cached
        rescale-crop rows uniformly (the released p = 0.5 rescale-crop, cityscapes.py:159-171)."""
        if self.aug is None:
            return idx
        alt = self.aug[idx]
        n = len(idx)
        use = (torch.rand(n, generator=generator) < 0.5).numpy() & (alt[:, 0] >= 0)
        pick = torch.randint(0, alt.shape[1], (n,), generator=generator).numpy()
        return np.where(use, alt[np.arange(n), pick], idx)


def batches(cache, idx, batch_size, shuffle, generator=None, drop_last=False, prefetch=4, masks=False):
    """Yield ([uint8 images (B, H, W, 3) per view], labels (B, C, K), valid or None, masks or None, positions).

    positions are indices into idx, so callers can place outputs back in split order.
    """
    order = np.arange(len(idx))
    if shuffle:
        order = torch.randperm(len(idx), generator=generator).numpy()
    chunks = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
    if drop_last and chunks and len(chunks[-1]) < batch_size:
        chunks = chunks[:-1]

    q = queue.Queue(maxsize=prefetch)

    def work():
        for pos in chunks:
            rows = idx[pos]
            srt = np.argsort(rows)  # sequential memmap reads, then restore the batch order
            imgs = []
            for arr in cache.views:
                img = np.empty((len(rows),) + arr.shape[1:], dtype=np.uint8)
                img[srt] = arr[rows[srt]]
                imgs.append(torch.from_numpy(img).pin_memory())
            v = torch.from_numpy(cache.valid[rows]) if cache.valid is not None else None
            mk = None
            if masks:
                mk = np.empty((len(rows),) + cache.masks.shape[1:], dtype=np.uint8)
                mk[srt] = cache.masks[rows[srt]]
                mk = torch.from_numpy(mk).pin_memory()
            q.put((imgs, torch.from_numpy(cache.labels[rows]).pin_memory(), v, mk, pos))
        q.put(None)

    t = threading.Thread(target=work, daemon=True)
    t.start()
    while (item := q.get()) is not None:
        yield item
    t.join()


def _gray(x):
    return (0.2989 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]).unsqueeze(1)


def _rgb_to_hsv(x):
    r, g, b = x.unbind(1)
    maxc, minc = x.max(1).values, x.min(1).values
    eqc = maxc == minc
    cr = maxc - minc
    ones = torch.ones_like(maxc)
    s = cr / torch.where(eqc, ones, maxc)
    crd = torch.where(eqc, ones, cr)
    rc, gc, bc = (maxc - r) / crd, (maxc - g) / crd, (maxc - b) / crd
    hr = (maxc == r) * (bc - gc)
    hg = ((maxc == g) & (maxc != r)) * (2.0 + rc - bc)
    hb = ((maxc != g) & (maxc != r)) * (4.0 + gc - rc)
    h = torch.fmod((hr + hg + hb) / 6.0 + 1.0, 1.0)
    return h, s, maxc


def _hsv_to_rgb(h, s, v):
    i = torch.floor(h * 6.0)
    f = h * 6.0 - i
    i = i.to(torch.int64) % 6
    p = (v * (1.0 - s)).clamp(0, 1)
    q = (v * (1.0 - s * f)).clamp(0, 1)
    t = (v * (1.0 - s * (1.0 - f))).clamp(0, 1)
    mask = (i.unsqueeze(1) == torch.arange(6, device=h.device).view(1, -1, 1, 1)).to(v.dtype)
    a = torch.stack((torch.stack((v, q, p, p, t, v), 1),
                     torch.stack((t, v, v, q, p, p), 1),
                     torch.stack((p, p, t, v, v, q), 1)), 1)  # (B, 3, 6, H, W)
    return torch.einsum('bkhw,bckhw->bchw', mask, a)


def color_jitter(x, brightness, contrast, saturation, hue, generator=None):
    """Per-image ColorJitter on float RGB in [0, 1], shape (B, 3, H, W)."""
    n, dev = x.shape[0], x.device

    def factor(v):
        return (1 - v + 2 * v * torch.rand(n, 1, 1, 1, generator=generator)).to(dev)

    ops = []
    if brightness:
        ops.append(lambda y, f=factor(brightness): (y * f).clamp(0, 1))
    if contrast:
        ops.append(lambda y, f=factor(contrast): (f * y + (1 - f) * _gray(y).mean((2, 3), keepdim=True)).clamp(0, 1))
    if saturation:
        ops.append(lambda y, f=factor(saturation): (f * y + (1 - f) * _gray(y)).clamp(0, 1))
    if hue:
        shift = ((torch.rand(n, 1, 1, generator=generator) * 2 - 1) * hue).to(dev)

        def hue_op(y):
            h, s, v = _rgb_to_hsv(y)
            return _hsv_to_rgb(torch.fmod(h + shift + 1.0, 1.0), s, v)
        ops.append(hue_op)
    for k in torch.randperm(len(ops), generator=generator).tolist():
        x = ops[k](x)
    return x


class GpuAugment:
    def __init__(self, perm, flip='hflip', jitter=None, normalize=True, generator=None, resize=None):
        """resize: optional square side; every view is resized to it (bilinear, antialiased) as the last step."""
        self.perm = torch.as_tensor(perm) if perm is not None else None
        self.flip, self.jitter, self.normalize, self.g = flip, jitter, normalize, generator
        self.resize = resize
        self.mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)

    def to_input(self, imgs):
        """uint8 (B, H, W, 3) on device, or a list of them (one per view) -> normalised float NCHW."""
        if isinstance(imgs, (list, tuple)):
            return [self.to_input(v) for v in imgs]
        x = imgs.permute(0, 3, 1, 2).float().div_(255)
        return self.finish(x)

    def finish(self, x):
        if self.resize:
            x = torch.nn.functional.interpolate(x, size=(self.resize, self.resize), mode='bilinear',
                                                align_corners=False, antialias=True)
        if self.normalize:
            x = (x - self.mean.to(x.device)) / self.std.to(x.device)
        return x.contiguous(memory_format=torch.channels_last)

    def train(self, imgs, labels, valid=None, masks=None):
        """imgs: one uint8 (B, H, W, 3) tensor or a list of views of the same samples.

        All views of a sample share one flip draw and one colour-jitter draw. Mirroring each view on its
        own is only correct when every view's crop is symmetric about the frame's vertical midline
        (true for the full frame and the Cityscapes fovea x 512-1536 of 2048).
        """
        single = not isinstance(imgs, (list, tuple))
        xs = [v.permute(0, 3, 1, 2).float().div_(255) for v in ([imgs] if single else imgs)]
        n, dev = xs[0].shape[0], xs[0].device
        if self.flip != 'none':
            m = (torch.rand(n, generator=self.g) < 0.5).to(dev)
            img_axis = -1 if self.flip == 'hflip' else -2
            xs = [torch.where(m.view(-1, 1, 1, 1), x.flip(img_axis), x) for x in xs]
            perm = self.perm.to(dev)
            labels = torch.where(m.view(-1, 1, 1), labels[:, perm], labels)
            if valid is not None:
                valid = torch.where(m.view(-1, 1), valid[:, perm], valid)
            if masks is not None:
                if self.flip != 'hflip':
                    raise ValueError('pixel masks only support the correct hflip')
                masks = torch.where(m.view(-1, 1, 1), masks.flip(-1), masks)
        if self.jitter:
            gen = self.g if self.g is not None else torch.default_generator
            state = gen.get_state()
            out = []
            for x in xs:
                gen.set_state(state)  # same factors and op order for every view
                out.append(color_jitter(x, generator=gen, **self.jitter))
            xs = out
        xs = [self.finish(x) for x in xs]
        return (xs[0] if single else xs), labels, valid, masks
