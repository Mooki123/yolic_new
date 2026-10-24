"""Boxes to per-cell scores (plan 04 section 4.2): a cell's class score is the max score of any box of that class that
overlaps the cell by at least one source pixel. Shared by the detector baselines and the box oracle (research/14)."""
import numpy as np


class BoxRasteriser:
    def __init__(self, layout):
        self.index = layout.index_map()  # (H, W) cell index per source pixel, -1 outside all cells
        self.n_cells = len(layout)

    def cells(self, x1, y1, x2, y2):
        """Indices of the cells overlapping the box [x1, x2) x [y1, y2) (source pixels) by at least one pixel."""
        h, w = self.index.shape
        x1, y1 = max(int(np.floor(x1)), 0), max(int(np.floor(y1)), 0)
        x2, y2 = min(int(np.ceil(x2)), w), min(int(np.ceil(y2)), h)
        if x2 <= x1 or y2 <= y1:
            return np.empty(0, dtype=np.int64)
        u = np.unique(self.index[y1:y2, x1:x2])
        return u[u >= 0]

    def scores(self, boxes, n_classes=2):
        """boxes: iterable of (cls, x1, y1, x2, y2, score). Returns (n_cells, n_classes + 1): class scores and a
        background column = 1 - max class score (the cell-label layout of the other models)."""
        out = np.zeros((self.n_cells, n_classes + 1), dtype=np.float32)
        for c, x1, y1, x2, y2, s in boxes:
            idx = self.cells(x1, y1, x2, y2)
            out[idx, int(c)] = np.maximum(out[idx, int(c)], s)
        out[:, -1] = 1 - out[:, :-1].max(1)
        return out
