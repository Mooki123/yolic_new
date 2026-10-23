"""research/13: the AlignNet port reproduces YOLIC-Align's original 'align' head (yolic-align branch) exactly."""
import os
import subprocess
import sys
import types

import pytest
import torch

from yolic_bench import heads
from yolic_bench.layout import Layout

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope='module')
def original(tmp_path_factory):
    d = tmp_path_factory.mktemp('yolic_align')
    for f in ('yolic_align.py', 'yolic_cells.py'):
        r = subprocess.run(['git', 'show', f'yolic-align:{f}'], cwd=ROOT, capture_output=True, text=True)
        if r.returncode:
            pytest.skip('yolic-align branch not available')
        (d / f).write_text(r.stdout)
    sys.modules.setdefault('cv2', types.ModuleType('cv2'))  # only polygon layouts need OpenCV
    sys.path.insert(0, str(d))
    import yolic_align
    yield yolic_align
    sys.path.remove(str(d))


def test_cell_weights_match(original):
    import yolic_cells
    ref = original.rect_cell_weights(yolic_cells.CITYSCAPES_CELLS, yolic_cells.CITYSCAPES_SIZE)
    ours, covered = heads.align_cell_weights(Layout.load(os.path.join(ROOT, 'layouts', 'cityscapes.json')), 224, 224, 14, 14)
    assert covered.all()
    assert torch.allclose(ours, ref.flatten(1), atol=1e-6)


def test_forward_matches_original(original):
    torch.manual_seed(0)
    ref = original.build_model('align', 'cityscapes', 256, 4).eval()
    with torch.no_grad():
        ref.cell_bias.normal_()
        ref.mix.normal_()
    ours = heads.build({'head': 'align'}, Layout.load(os.path.join(ROOT, 'layouts', 'cityscapes.json')), 4,
                       pretrained=False).eval()
    state = {('trunks.0.' + k if k.split('.')[0] in ('stride16', 'stride32', 'lateral') else k): v
             for k, v in ref.state_dict().items()}
    missing, unexpected = ours.load_state_dict(state, strict=False)
    assert not unexpected and set(missing) == {'view_logit'}
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        assert torch.allclose(ours(x), ref(x).view(2, 256, 4), atol=1e-4)


def test_foveated_views_cover_every_cell():
    lay = Layout.load(os.path.join(ROOT, 'layouts', 'cityscapes.json'))
    specs = [((0, 0, 2048, 1024), (224, 112)), ((512, 320, 1536, 640), (320, 96))]
    m = heads.build({'head': 'align'}, lay, 4, pretrained=False, view_specs=specs).eval()
    assert m.covered[:, 0].all()                      # the full frame sees every cell
    assert m.covered[:160, 1].all() and not m.covered[160:, 1].any()  # the fovea sees exactly the far cells
    with torch.no_grad():
        out = m(torch.randn(2, 3, 112, 224), torch.randn(2, 3, 96, 320))
    assert out.shape == (2, 256, 4) and torch.isfinite(out).all()
    assert m.trunks[0].stride16[0][0].weight is m.trunks[1].stride16[0][0].weight  # convolutions shared
