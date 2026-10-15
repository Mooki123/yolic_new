"""Cityscapes cell labels in tools/cache_cityscapes.py against the released encode_cell (real masks).

Skipped when the dataset is not at ~/data/cityscapes.
"""
import os
import sys

import numpy as np
import pytest
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CS = os.path.expanduser('~/data/cityscapes')
pytestmark = pytest.mark.skipif(not os.path.isdir(os.path.join(CS, 'gtFine')), reason='Cityscapes not found')

sys.path.insert(0, os.path.join(ROOT, 'tools'))
import cache_cityscapes as C  # noqa: E402
from yolic_bench import released  # noqa: E402
from yolic_bench.layout import Layout  # noqa: E402


@pytest.fixture(scope='module')
def setup():
    sys.path.insert(0, ROOT)
    from cityscapes import Cityscapes  # the untouched released dataset code
    layout = Layout.load(os.path.join(ROOT, 'layouts', 'cityscapes.json'))
    cell_list = released.cityscapes_cells(os.path.join(ROOT, 'cityscapes_yolic.py'))
    masks = [x[2] for x in C.listing(CS, 'val')[::97][:5]]
    return Cityscapes, layout, cell_list, masks


def test_released_labels_match_encode_cell(setup):
    Cityscapes, _, cell_list, masks = setup
    for p in masks:
        ids = np.asarray(Image.open(p))
        ref = Cityscapes.encode_cell(Cityscapes.encode_target(ids), cell_list, C.GROUPS).reshape(-1, 4)
        assert np.array_equal(C.released_labels(C.RELEASED_TRAIN_ID[ids], cell_list), ref)


def test_v1_labels_match_slow_reference(setup):
    """Vectorised counts == per-cell crop of the full-res mask, on plain and rescale-crop states."""
    _, layout, cell_list, masks = setup
    index_map = layout.index_map()
    areas = np.bincount(index_map[index_map >= 0], minlength=len(layout)).astype(np.float64)
    for p in masks[:2]:
        ids = Image.open(p)
        big = np.asarray(ids.resize(C.BIG, Image.NEAREST), dtype=np.uint8)
        for state in (np.asarray(ids), big[75:75 + 1024, 50:50 + 2048]):
            groups = C.V1_GROUP[state]
            lab, valid = C.v1_labels(C.cell_counts(groups, index_map, len(layout)), areas)
            for num, ((x1, y1), (x2, y2)) in enumerate(cell_list):
                g = groups[y1:y2, x1:x2]
                obj = [(g == k).any() for k in range(3)]
                assert list(lab[num]) == obj + [not any(obj)]
                assert valid[num] == ((g == 4).mean() <= C.TAU_VOID)


def test_flip_labels_equal_mirror_permutation(setup):
    """Labels of the flipped mask == mirror permutation of the labels (what the GPU flip assumes)."""
    _, layout, _, masks = setup
    index_map = layout.index_map()
    areas = np.bincount(index_map[index_map >= 0], minlength=len(layout)).astype(np.float64)
    perm = layout.mirror_permutation()
    ids = np.asarray(Image.open(masks[0]))
    lab, valid = C.v1_labels(C.cell_counts(C.V1_GROUP[ids], index_map, len(layout)), areas)
    labf, validf = C.v1_labels(C.cell_counts(C.V1_GROUP[ids[:, ::-1]], index_map, len(layout)), areas)
    assert np.array_equal(labf, lab[perm]) and np.array_equal(validf, valid[perm])
