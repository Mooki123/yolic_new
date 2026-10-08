"""Cell layouts: geometry of the Cells of Interest, independent of the input resolution.

A layout file (layouts/*.json) holds the source frame size, one polygon per cell in label order
(pixel coordinates of that frame; a rectangle is stored as its 4 corners) and optional named cell
groups for the per-size breakdowns. tools/export_layouts.py writes the files from the released scripts.

Rasterisation tests pixel centres, so the rectangle [[x1, y1], [x2, y2]] covers exactly the pixels
target[y1:y2, x1:x2] that cityscapes.py:encode_cell crops at the source resolution.
"""
import json

import numpy as np


def _inside(poly, xs, ys):
    """Even-odd test of points (xs, ys) (broadcastable arrays) against one polygon (V x 2)."""
    inside = np.zeros(np.broadcast(xs, ys).shape, dtype=bool)
    x0, y0 = poly[-1]
    for x1, y1 in poly:
        if y0 != y1:
            crosses = (y1 > ys) != (y0 > ys)
            x_at = x1 + (ys - y1) * (x0 - x1) / (y0 - y1)
            inside ^= crosses & (xs < x_at)
        x0, y0 = x1, y1
    return inside


class Layout:
    def __init__(self, name, frame, cells, groups=None):
        self.name = name
        self.width, self.height = frame
        self.cells = [np.asarray(c, dtype=np.float64).reshape(-1, 2) for c in cells]
        self.groups = {k: np.asarray(v) for k, v in (groups or {}).items()}

    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        return cls(d['name'], d['frame'], d['cells'], d.get('groups'))

    def __len__(self):
        return len(self.cells)

    def _grid(self, out_w, out_h, s, crop):
        """Source coordinates of the sub-pixel centres of an out_w x out_h output, s x s per pixel.

        The centres are symmetric about the crop's vertical midline, so mirroring a raster is the
        same as rasterising the mirrored geometry.
        """
        x1, y1, x2, y2 = crop if crop is not None else (0, 0, self.width, self.height)
        out_w = int(x2 - x1) if out_w is None else out_w
        out_h = int(y2 - y1) if out_h is None else out_h
        xs = x1 + (np.arange(out_w * s) + 0.5) * (x2 - x1) / (out_w * s)
        ys = y1 + (np.arange(out_h * s) + 0.5) * (y2 - y1) / (out_h * s)
        return out_w, out_h, xs, ys

    def _cell_masks(self, xs, ys):
        """Yield (i, row slice, col slice, bool mask) over each cell's bounding box on the grid."""
        for i, poly in enumerate(self.cells):
            cx = np.nonzero((xs >= poly[:, 0].min()) & (xs <= poly[:, 0].max()))[0]
            cy = np.nonzero((ys >= poly[:, 1].min()) & (ys <= poly[:, 1].max()))[0]
            if len(cx) == 0 or len(cy) == 0:
                continue
            rows, cols = slice(cy[0], cy[-1] + 1), slice(cx[0], cx[-1] + 1)
            yield i, rows, cols, _inside(poly, xs[cols][None, :], ys[rows][:, None])

    def rasterize(self, out_w=None, out_h=None, supersample=1, crop=None):
        """Per-cell coverage at an output resolution: float32 array (n_cells, out_h, out_w).

        Values are the fraction of each output pixel covered by the cell, estimated on a
        supersample x supersample grid of sub-pixel centres (supersample=1 gives a 0/1 mask).
        crop = (x1, y1, x2, y2) in source pixels maps only that window to the output
        (crop-to-cell-union input); by default the whole frame is used.
        Meant for model resolutions (n_cells x out_h x out_w floats); use index_map at full size.
        """
        s = supersample
        out_w, out_h, xs, ys = self._grid(out_w, out_h, s, crop)
        out = np.zeros((len(self.cells), out_h, out_w), dtype=np.float32)
        for i, rows, cols, mask in self._cell_masks(xs, ys):
            fine = np.zeros((out_h * s, out_w * s), dtype=np.float32)
            fine[rows, cols] = mask
            out[i] = fine.reshape(out_h, s, out_w, s).mean(axis=(1, 3))
        return out

    def index_map(self, out_w=None, out_h=None, crop=None):
        """int16 (out_h, out_w) map of the cell index at each pixel centre, -1 outside all cells.

        Raises ValueError if two cells claim the same pixel.
        """
        out_w, out_h, xs, ys = self._grid(out_w, out_h, 1, crop)
        out = np.full((out_h, out_w), -1, dtype=np.int16)
        for i, rows, cols, mask in self._cell_masks(xs, ys):
            view = out[rows, cols]
            if (view[mask] >= 0).any():
                raise ValueError(f'cell {i} overlaps cell {int(view[mask].max())} in layout {self.name!r}')
            view[mask] = i
        return out

    def mirror_permutation(self, min_iou=0.98):
        """perm[i] = the cell that holds, before a horizontal flip, what cell i holds after it.

        This is the seq_list the released random_augmentation expects: new cell i takes the bits
        of old cell perm[i]. Raises ValueError if the layout is not left-right symmetric.
        """
        n = len(self)
        a = self.index_map().astype(np.int64) + 1  # 0 = no cell
        b = a[:, ::-1]  # b at a pixel = the cell whose mirror covers it
        inter = np.bincount((b * (n + 1) + a).ravel(), minlength=(n + 1) ** 2).reshape(n + 1, n + 1)[1:, 1:]
        area = np.bincount(a.ravel(), minlength=n + 1)[1:]
        iou = inter / np.maximum(area[:, None] + area[None, :] - inter, 1)  # (mirror of i, cell j)
        perm = iou.argmax(axis=1)
        worst = iou[np.arange(len(self)), perm].min()
        if worst < min_iou or len(set(perm.tolist())) != len(self):
            raise ValueError(f'layout {self.name!r} is not left-right symmetric (worst IoU {worst:.3f})')
        return perm

    def union_box(self):
        """Bounding box (x1, y1, x2, y2) of all cells, in source pixels."""
        pts = np.concatenate(self.cells)
        return pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()


def flip_cell_labels(labels, perm):
    """Apply a mirror permutation to labels of shape (..., n_cells, n_bits)."""
    return np.asarray(labels)[..., perm, :]
