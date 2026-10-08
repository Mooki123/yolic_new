"""Equivalence test: yolic_bench.metrics must reproduce the released evaluation to 1e-6 (plan 04, R8).

`reference_report` is cityscapes_eval.py:pred_cm copied line for line (with NumCell / NumClass
as arguments and numpy input instead of torch tensors), followed by the same
classification_report calls the script prints.
"""
import numpy as np
import pytest
from sklearn import metrics as skm

from yolic_bench import metrics as M


def reference_report(targets, probs, NumCell, NumClass):
    Gt, Pred, binary_Gt, binary_Pred = [], [], [], []

    def pred_cm(original, predicted):
        orig = original
        pred = predicted
        pred = np.reshape(pred, (NumCell * (NumClass + 1), 1)).flatten()
        orig = np.reshape(orig, (NumCell * (NumClass + 1), 1)).flatten()
        for i in range(0, (NumCell * (NumClass + 1)), (NumClass + 1)):
            pred_out = np.where(pred[i:i + (NumClass + 1)] > 0.5, 1, 0)
            for index, (ground_truth, prediction) in enumerate(zip(orig[i:i + (NumClass + 1)], pred_out)):
                if ground_truth == prediction == 1:
                    Gt.append(index)
                    Pred.append(index)
                if prediction == 1 and ground_truth == 0:
                    Pred.append(index)
                    Gt.append(NumClass + 1)
                if prediction == 0 and ground_truth == 1:
                    Pred.append(NumClass + 1)
                    Gt.append(index)
                if index == NumClass:
                    if prediction == 0 and ground_truth == 0:
                        binary_Pred.append(0)
                        binary_Gt.append(0)
                    if prediction == 1 and ground_truth == 0:
                        binary_Pred.append(1)
                        binary_Gt.append(0)
                    if prediction == 0 and ground_truth == 1:
                        binary_Pred.append(0)
                        binary_Gt.append(1)
                    if prediction == 1 and ground_truth == 1:
                        binary_Pred.append(1)
                        binary_Gt.append(1)

    for t, p in zip(targets, probs):
        pred_cm(t, p)
    labels = list(range(NumClass + 2))
    report = skm.classification_report(Gt, Pred, labels=labels, output_dict=True, zero_division=0)
    binary = skm.classification_report(binary_Gt, binary_Pred, labels=[0, 1], output_dict=True,
                                       zero_division=0)
    return report, binary


def make_data(rng, n_images, n_cells, n_class, pos_rates):
    """Random targets / probabilities, flat as the model emits them, with some exact 0.5 scores."""
    n_bits = n_class + 1
    gt = rng.random((n_images, n_cells, n_bits)) < np.asarray(pos_rates)
    probs = np.clip(gt * 0.35 + rng.random(gt.shape) * 0.65, 0, 1)  # informative but noisy
    probs[rng.random(gt.shape) < 0.02] = 0.5  # the strict '> 0.5' must hold
    return gt.reshape(n_images, -1).astype(np.float32), probs.reshape(n_images, -1).astype(np.float32)


CASES = {
    'cityscapes': (12, 256, 3, [0.05, 0.15, 0.6, 0.4]),
    'outdoor': (8, 104, 11, [0.02, 0.05, 0.1, 0.01, 0.03, 0.2, 0.07, 0.04, 0.02, 0.06, 0.08, 0.5]),
    'never_predicted': (5, 16, 3, [0.0, 0.3, 0.5, 0.5]),  # class 0 absent: zero-division path
}


@pytest.mark.parametrize('case', CASES)
def test_matches_released_eval(case):
    n_images, n_cells, n_class, rates = CASES[case]
    rng = np.random.default_rng(42)
    gt_flat, probs_flat = make_data(rng, n_images, n_cells, n_class, rates)
    if case == 'never_predicted':
        probs_flat.reshape(n_images, n_cells, -1)[..., 0] = 0.1
    ref, ref_bin = reference_report(gt_flat, probs_flat, n_cells, n_class)

    gt = M.to_cells(gt_flat, n_cells)
    pred = M.binarize(M.to_cells(probs_flat, n_cells))
    c = M.counts(gt, pred)
    p, r, f1 = M.prf(c)
    for k in range(n_class + 1):
        row = ref[str(k)]
        assert p[k] == pytest.approx(row['precision'], abs=1e-6)
        assert r[k] == pytest.approx(row['recall'], abs=1e-6)
        assert f1[k] == pytest.approx(row['f1-score'], abs=1e-6)
        assert c['tp'][k] + c['fn'][k] == row['support']

    objects = range(n_class)
    mp, mr, mf = M.all_score(c, objects)
    ref_mp = np.mean([ref[str(k)]['precision'] for k in objects])
    ref_mr = np.mean([ref[str(k)]['recall'] for k in objects])
    assert mp == pytest.approx(ref_mp, abs=1e-6)
    assert mr == pytest.approx(ref_mr, abs=1e-6)
    assert mf == pytest.approx(2 * ref_mp * ref_mr / (ref_mp + ref_mr), abs=1e-6)

    bp, br, bf, bs = M.binary_road(gt, pred)
    for i in (0, 1):
        assert bp[i] == pytest.approx(ref_bin[str(i)]['precision'], abs=1e-6)
        assert br[i] == pytest.approx(ref_bin[str(i)]['recall'], abs=1e-6)
        assert bf[i] == pytest.approx(ref_bin[str(i)]['f1-score'], abs=1e-6)
        assert bs[i] == ref_bin[str(i)]['support']


def test_valid_mask_equals_dropping_cells():
    rng = np.random.default_rng(0)
    gt_flat, probs_flat = make_data(rng, 6, 32, 3, [0.1, 0.2, 0.5, 0.4])
    gt, pred = M.to_cells(gt_flat, 32), M.binarize(M.to_cells(probs_flat, 32))
    cells = np.arange(0, 32, 3)
    a = M.counts(gt, pred, cells=cells)
    b = M.counts(gt[:, cells], pred[:, cells])
    for k in a:
        np.testing.assert_array_equal(a[k], b[k])
    valid = rng.random((6, 32)) < 0.7
    v = M.counts(gt, pred, valid=valid)
    assert v['tp'].sum() == (gt.astype(bool) & pred & valid[..., None]).sum()


def test_bootstrap_identical_models_is_zero():
    rng = np.random.default_rng(1)
    gt_flat, probs_flat = make_data(rng, 30, 16, 3, [0.1, 0.2, 0.5, 0.4])
    gt, pred = M.to_cells(gt_flat, 16), M.binarize(M.to_cells(probs_flat, 16))
    d, lo, hi = M.paired_bootstrap(gt, pred, pred, [0, 1], n_boot=50)
    assert d == lo == hi == 0
    d, lo, hi = M.paired_bootstrap(gt, pred, gt, [0, 1], n_boot=200)  # perfect B beats A
    assert d > 0 and lo > 0 and lo <= d <= hi
