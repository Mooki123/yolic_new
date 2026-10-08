"""Layout tests (plan 04, section 4.1): geometry matches the released scripts, and the flip
permutation derived from the geometry equals the hard-coded seq_list tables. The second check also
shows that the cell order in outdoor_pred.py / indoor_pred.py is the label order the training uses.
"""
import os

import numpy as np
import pytest

from tools.export_layouts import build
from yolic_bench import released
from yolic_bench.layout import Layout, flip_cell_labels

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAMES = ['cityscapes', 'outdoor', 'indoor']


def src(name):
    return os.path.join(ROOT, name)


def load(name):
    return Layout.load(os.path.join(ROOT, 'layouts', name + '.json'))


@pytest.fixture(autouse=True)
def _chdir(monkeypatch):
    monkeypatch.chdir(ROOT)


@pytest.mark.parametrize('name', NAMES)
def test_json_is_up_to_date(name):
    expected = build(name)
    lay = load(name)
    assert len(lay) == len(expected['cells'])
    for got, want in zip(lay.cells, expected['cells']):
        np.testing.assert_array_equal(got, np.asarray(want, dtype=np.float64))


def test_cityscapes_copies_agree():
    ref = released.cityscapes_cells(src('cityscapes_yolic.py'))
    assert released.cityscapes_cells(src('cityscapes_eval.py')) == ref
    assert released.cityscapes_cells(src('cityscapes_pred.py')) == ref


def test_rasterize_matches_encode_cell_crop():
    """At source resolution a cell covers exactly target[y1:y2, x1:x2], as cityscapes.py crops it."""
    lay = load('cityscapes')
    masks = lay.rasterize()
    for i, ((x1, y1), (x2, y2)) in enumerate(released.cityscapes_cells(src('cityscapes_yolic.py'))):
        want = np.zeros((1024, 2048), dtype=np.float32)
        want[y1:y2, x1:x2] = 1
        np.testing.assert_array_equal(masks[i], want)


@pytest.mark.parametrize('name', NAMES)
def test_cells_do_not_overlap(name):
    assert load(name).rasterize().sum(axis=0).max() <= 1


@pytest.mark.parametrize('name', NAMES)
def test_supersampled_coverage_preserves_area(name):
    lay = load(name)
    full = lay.rasterize().sum(axis=(1, 2))
    small = lay.rasterize(56, 56, supersample=8).sum(axis=(1, 2))
    scale = lay.width * lay.height / (56 * 56)
    np.testing.assert_allclose(small * scale, full, rtol=0.05)


@pytest.mark.parametrize('layout_name, scripts', [
    ('outdoor', ['outdoor_yolic.py', 'outdoor_eval.py']),
    ('indoor', ['indoor_yolic.py', 'indoor_eval.py']),
])
def test_mirror_permutation_equals_released_tables(layout_name, scripts):
    perm = load(layout_name).mirror_permutation()
    for script in scripts:
        tables = released.flip_tables(src(script))
        assert len(tables) == 1, script
        assert perm.tolist() == tables[0], script


def test_cityscapes_is_symmetric():
    perm = load('cityscapes').mirror_permutation()
    assert sorted(perm.tolist()) == list(range(256))
    assert (perm[perm] == np.arange(256)).all()


def test_asymmetric_layout_is_refused():
    lay = Layout('lopsided', [100, 50], [[[0, 0], [30, 0], [30, 50], [0, 50]],
                                          [[30, 0], [100, 0], [100, 50], [30, 50]]])
    with pytest.raises(ValueError):
        lay.mirror_permutation()


def test_flip_cell_labels_matches_released_regrouping():
    """flip_cell_labels gives the same label vector as the label half of random_augmentation."""
    def released_regroup(label_list, seq_list):  # outdoor_yolic.py:random_augmentation, minus the image
        n_groups = len(seq_list)
        group_size = len(label_list) // n_groups
        label_groups, start_idx = [], 0
        for _ in seq_list:
            label_groups.append(label_list[start_idx:start_idx + group_size])
            start_idx += group_size
        out = []
        for group_idx in seq_list:
            out.extend(label_groups[group_idx])
        return out

    perm = load('outdoor').mirror_permutation()
    labels = np.random.default_rng(0).integers(0, 2, 104 * 12)
    want = released_regroup(list(labels), perm.tolist())
    got = flip_cell_labels(labels.reshape(104, 12), perm).reshape(-1)
    assert got.tolist() == [int(v) for v in want]
