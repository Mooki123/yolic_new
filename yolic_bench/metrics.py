"""Shared per-cell metric for every model family (plan 04, section 2.2).

Arrays use the layout (images, cells, bits). The last bit is the background bit (Road on Cityscapes),
so with M object classes there are M + 1 bits per cell, exactly as in the released model output.

Every (cell, bit) pair is scored on its own: a positive prediction on a positive label is a TP, a
positive prediction on a negative label is an FP, and a missed positive is an FN. Pooled over cells
and images, this gives the same per-class precision / recall / F1 as the released
`pred_cm` + `classification_report` code in cityscapes_eval.py; tests/test_metrics.py checks that.
"""
import numpy as np
from sklearn.metrics import average_precision_score


def to_cells(flat, n_cells):
    """Reshape a (N, n_cells * n_bits) model output or target into (N, n_cells, n_bits)."""
    flat = np.asarray(flat)
    return flat.reshape(flat.shape[0], n_cells, -1)


def binarize(probs, threshold=0.5):
    """Threshold probabilities. Strictly greater, as in the released code. threshold may be per bit."""
    return np.asarray(probs) > np.asarray(threshold)


def _weights(shape, valid=None, cells=None):
    """(N, n_cells) weight of 0/1: void-masked cells and cells outside the subset count for nothing."""
    w = np.ones(shape[:2], dtype=bool)
    if valid is not None:
        w &= np.asarray(valid, dtype=bool)
    if cells is not None:
        keep = np.zeros(shape[1], dtype=bool)
        keep[np.asarray(cells)] = True
        w &= keep[None, :]
    return w


def counts(gt, pred, valid=None, cells=None, per_image=False):
    """TP, FP, FN (and TN) per bit, pooled over the selected cells.

    gt, pred: (N, n_cells, n_bits), binary. valid: optional (N, n_cells) mask (void cells -> False).
    cells: optional index array of the cell subset to score. per_image=True keeps the image axis,
    which the bootstrap needs. Returns a dict of int64 arrays of shape (n_bits,) or (N, n_bits).
    """
    gt = np.asarray(gt).astype(bool)
    pred = np.asarray(pred).astype(bool)
    w = _weights(gt.shape, valid, cells)[:, :, None]
    axis = 1 if per_image else (0, 1)
    return {
        'tp': (gt & pred & w).sum(axis=axis),
        'fp': (~gt & pred & w).sum(axis=axis),
        'fn': (gt & ~pred & w).sum(axis=axis),
        'tn': (~gt & ~pred & w).sum(axis=axis),
    }


def _div(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return np.divide(a, b, out=np.zeros(np.broadcast(a, b).shape), where=b > 0)


def prf(c):
    """Precision, recall, F1 per bit from counts(). Zero division gives 0, as sklearn does."""
    p = _div(c['tp'], c['tp'] + c['fp'])
    r = _div(c['tp'], c['tp'] + c['fn'])
    return p, r, _div(2 * p * r, p + r)


def macro_f1(c, classes):
    """Mean F1 over the given bit indices (e.g. People+Vehicle = [0, 1])."""
    return prf(c)[2][..., list(classes)].mean(axis=-1)


def all_score(c, object_classes):
    """The paper's 'All' column (D17): macro P and R over object classes, F1 = their harmonic mean."""
    p, r, _ = prf(c)
    mp = p[..., list(object_classes)].mean(axis=-1)
    mr = r[..., list(object_classes)].mean(axis=-1)
    return mp, mr, _div(2 * mp * mr, mp + mr)


def binary_road(gt, pred, road_bit=-1, valid=None, cells=None):
    """Risk / Road report on the background bit (released code: 0 = Risk, 1 = Road).

    Returns (precision, recall, f1, support), each of shape (2,) in the order [Risk, Road].
    """
    c = counts(gt, pred, valid, cells)
    tp, fp, fn, tn = (int(c[k][road_bit]) for k in ('tp', 'fp', 'fn', 'tn'))
    # For the Risk row the roles flip: Road TN is a Risk TP, Road FN is a Risk FP, Road FP a Risk FN.
    p = np.array([_div(tn, tn + fn), _div(tp, tp + fp)], dtype=np.float64)
    r = np.array([_div(tn, tn + fp), _div(tp, tp + fn)], dtype=np.float64)
    return p, r, _div(2 * p * r, p + r), np.array([tn + fp, tp + fn])


def average_precision(gt, probs, classes, valid=None, cells=None):
    """Per-class AP over all selected (image, cell) pairs. NaN for a class with no positives."""
    gt = np.asarray(gt).astype(bool)
    probs = np.asarray(probs, dtype=np.float64)
    w = _weights(gt.shape, valid, cells)
    out = []
    for k in classes:
        y, s = gt[..., k][w], probs[..., k][w]
        out.append(average_precision_score(y, s) if y.any() else np.nan)
    return np.array(out)


def tune_thresholds(gt, probs, classes, grid=None, valid=None, cells=None):
    """Per-class threshold that maximises F1 on a selection split (val'), never on test."""
    grid = np.linspace(0.05, 0.95, 91) if grid is None else np.asarray(grid)
    thr = np.full(np.asarray(gt).shape[-1], 0.5)
    for k in classes:
        f1s = [prf(counts(gt[..., k:k + 1], binarize(probs[..., k:k + 1], t), valid, cells))[2][0]
               for t in grid]
        thr[k] = grid[int(np.argmax(f1s))]
    return thr


def paired_bootstrap(gt, pred_a, pred_b, classes, n_boot=1000, seed=0, valid=None, cells=None):
    """Bootstrap over test images of macro-F1(B) - macro-F1(A), seeds pooled (plan 04, section 2.2).

    pred_a, pred_b: binary predictions of shape (S, N, n_cells, n_bits) (S seeds; S may differ
    between A and B) or (N, n_cells, n_bits). Each resample draws N images with replacement, the
    same images for both models, and averages macro-F1 over seeds.
    Returns (observed delta, 2.5th percentile, 97.5th percentile).
    """
    def per_image(preds):
        preds = np.asarray(preds)
        preds = preds[None] if preds.ndim == 3 else preds
        cs = [counts(gt, p, valid, cells, per_image=True) for p in preds]
        return {k: np.stack([c[k] for c in cs]) for k in ('tp', 'fp', 'fn')}  # (S, N, n_bits)

    ca, cb = per_image(pred_a), per_image(pred_b)

    def delta(idx):
        fa = macro_f1({k: v[:, idx].sum(axis=1) for k, v in ca.items()}, classes).mean()
        fb = macro_f1({k: v[:, idx].sum(axis=1) for k, v in cb.items()}, classes).mean()
        return fb - fa

    n = np.asarray(gt).shape[0]
    rng = np.random.default_rng(seed)
    boots = np.array([delta(rng.integers(0, n, n)) for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return delta(np.arange(n)), lo, hi
