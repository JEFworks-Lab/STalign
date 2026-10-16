'''Tests of the public module ``STalign.STalign`` as the tutorials use it.

The public names of the original module, both ways of importing it (``from STalign import
STalign`` and the tutorials' "option B", which puts the STalign folder on sys.path), the
choice of LDDMM backend (the compiled engine and torch agree; 'auto' falls back to torch when
the extension is missing or the problem is not CPU float64), the progress figures, and a
small rasterize -> LDDMM -> transforms workflow.

The numerics of each backend are tested against the oracle in test_lddmm_torch.py and
test_lddmm_cpp.py.
'''
import importlib.util
import inspect
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for path in (REPO, HERE):
    if path not in sys.path:
        sys.path.insert(0, path)

from STalign import STalign  # noqa: E402
from STalign import _lddmm_common as C  # noqa: E402
from STalign import _lddmm_torch  # noqa: E402
import reference as R  # noqa: E402

HAVE_CPP = STalign.backend_info()['cpp']
needs_cpp = pytest.mark.skipif(not HAVE_CPP, reason='the C++ extension is not built')

# every function of the original STalign.py (the tutorials use most of them)
ORIGINAL_FUNCTIONS = [
    'L_T_from_points', 'LDDMM', 'LDDMM_3D_to_slice', 'analyze3Dalign', 'build_transform', 'build_transform3D',
    'calculate_tre', 'clip', 'download_aba_image_labels', 'download_aba_ontology', 'extent_from_x', 'interp',
    'interp3D', 'make_scree', 'normalize', 'plot_brain_regions', 'plot_subset_brain_regions', 'rasterize',
    'rasterizePCA', 'rasterize_with_signal', 'saveRasters', 'to_A', 'to_A_3D', 'transform_image_source_to_target',
    'transform_image_source_with_A', 'transform_image_target_to_source', 'transform_points_source_to_target',
    'transform_points_target_to_source', 'v_to_phii', 'v_to_phii_3D']

# small fits with a stable descent (see test_lddmm_cpp.py); 60 iterations include an E-step
FIT_2D = dict(a=12.0, sigmaM=0.3, sigmaR=5e3, epT=1e-2, epL=1e-5, epV=0.2, niter=60, diffeo_start=10)
FIT_3D = dict(nt=4, a=10.0, expand=1.25, sigmaM=0.5, sigmaR=5e2, epT=1e-2, epL=1e-5, epV=5.0, niter=60,
              L=0.95 * R.rotation_3d(4.0, -3.0), T=[0.5, 0.3, -0.4], prefit=False)
KEYS_2D = ('A', 'v', 'WM', 'WB', 'WA')
KEYS_3D = KEYS_2D + ('Xs',)

# defaults that were changed on purpose (see CHANGES.md): (function, parameter) -> (old, new);
# None: chosen from the data (legacy_defaults=True restores the old values)
CHANGED_DEFAULTS = {('LDDMM_3D_to_slice', 'sigmaR'): (1e8, None), ('LDDMM', 'sigmaR'): (5e5, None),
                    ('LDDMM', 'niter'): (5000, None), ('LDDMM', 'epL'): (2e-8, None),
                    ('LDDMM', 'epT'): (2e-1, None), ('LDDMM', 'epV'): (2e3, None),
                    ('LDDMM_3D_to_slice', 'epV'): (1e3, None)}


def data_2d(points=True, CJ=2, **kw):
    d = R.synthetic_2d(CI=3, CJ=CJ)
    if not points:
        d.pop('pointsI'), d.pop('pointsJ')
    return dict(d, **dict(FIT_2D, **kw))


def data_3d(**kw):
    return dict(R.synthetic_3d(), **dict(FIT_3D, **kw))


def rel(a, b):
    '''max |a - b| relative to max |b|.'''
    a, b = torch.as_tensor(a).double(), torch.as_tensor(b).double()
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-300))


def original_module():
    '''Version 1.0's STalign.py, if available (STALIGN_ORIGINAL or dev/ref/stalign_orig.py).'''
    path = os.environ.get('STALIGN_ORIGINAL') or os.path.join(REPO, 'dev', 'ref', 'stalign_orig.py')
    if not os.path.exists(path):
        pytest.skip('the frozen original STalign.py is not available (set STALIGN_ORIGINAL)')
    spec = importlib.util.spec_from_file_location('stalign_original', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- public names
def test_public_functions():
    for name in ORIGINAL_FUNCTIONS + ['backend_info']:
        f = getattr(STalign, name)
        assert callable(f), name
        # the documentation lists the functions whose __module__ is this module
        assert f.__module__ == STalign.__name__, name
    info = STalign.backend_info()
    assert set(info) == {'auto', 'cpp', 'cpp_error', 'num_threads'}
    assert info['auto'] == ('cpp' if info['cpp'] else 'torch')


# blur=None means blur=1.0 unless the keyword-only blur_um is given (the two exclude each other)
CHANGED_DEFAULTS.update({('rasterize', 'blur'): (1.0, None), ('rasterize_with_signal', 'blur'): (1.0, None)})


def test_signatures_extend_the_original():
    '''Every original parameter keeps its name, position and default (except the documented
    CHANGED_DEFAULTS); new parameters are keyword-only or come last with a default.'''
    S = original_module()
    assert sorted(n for n in dir(S) if inspect.isfunction(getattr(S, n))
                  and getattr(S, n).__module__ == S.__name__) == sorted(ORIGINAL_FUNCTIONS)
    for name in ORIGINAL_FUNCTIONS:
        old = list(inspect.signature(getattr(S, name)).parameters.values())
        new = list(inspect.signature(getattr(STalign, name)).parameters.values())
        assert [p.name for p in new[:len(old)]] == [p.name for p in old], name
        for p, q in zip(old, new):
            assert p.kind == q.kind, (name, p.name)
            if (name, p.name) in CHANGED_DEFAULTS:
                assert (p.default, q.default) == CHANGED_DEFAULTS[name, p.name], (name, p.name)
            elif isinstance(p.default, np.ndarray):
                assert np.array_equal(p.default, q.default), (name, p.name)
            else:
                assert p.default == q.default, (name, p.name)
        assert all(q.default is not inspect.Parameter.empty for q in new[len(old):]), name


def test_import_needs_no_optional_dependencies():
    '''The module imports without plotly, tornado (never used) and pandas, requests, pynrrd
    (only needed by the atlas helpers, which import them when called).'''
    code = ('import sys\n'
            'for m in ("plotly", "tornado", "pandas", "requests", "nrrd"):\n'
            '    sys.modules[m] = None  # makes "import m" fail\n'
            f'sys.path.insert(0, {REPO!r})\n'
            'from STalign import STalign\n'
            'assert callable(STalign.LDDMM)\n')
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr


def test_option_b_import(tmp_path):
    '''The tutorials' "option B": the STalign folder on sys.path and ``import STalign`` gives
    this module, whose LDDMM uses the same backend and gives the same result.'''
    kw = data_2d(niter=20)
    np.savez(tmp_path / 'inputs.npz', **{k: np.asarray(kw[k]) for k in ('I', 'J', 'pointsI', 'pointsJ')},
             xI0=kw['xI'][0], xI1=kw['xI'][1], xJ0=kw['xJ'][0], xJ1=kw['xJ'][1])
    fit = {k: kw[k] for k in FIT_2D if k != 'niter'}
    code = f'''
import sys
sys.path.append({os.path.join(REPO, "STalign")!r})
import numpy as np
import torch
torch.set_num_threads(2)
import STalign
assert STalign.__file__.endswith('STalign.py'), STalign.__file__
missing = [n for n in {ORIGINAL_FUNCTIONS!r} + ['backend_info'] if not callable(getattr(STalign, n, None))]
assert not missing, missing
d = np.load('inputs.npz')
out = STalign.LDDMM([d['xI0'], d['xI1']], d['I'], [d['xJ0'], d['xJ1']], d['J'], d['pointsI'], d['pointsJ'],
                    niter=20, draw=False, **{fit!r})
np.savez('outputs.npz', auto=STalign.backend_info()['auto'], **{{k: out[k].numpy() for k in {KEYS_2D!r}}})
'''
    env = {k: v for k, v in os.environ.items() if k != 'PYTHONPATH'}
    result = subprocess.run([sys.executable, '-c', code], cwd=tmp_path, env=env, capture_output=True, text=True,
                            timeout=300)
    assert result.returncode == 0, result.stderr
    theirs = np.load(tmp_path / 'outputs.npz')
    assert str(theirs['auto']) == STalign.backend_info()['auto']
    ours = STalign.LDDMM(**kw, draw=False)
    for k in KEYS_2D:
        np.testing.assert_array_equal(ours[k].numpy(), theirs[k], err_msg=k)


def test_package_exports_the_module_functions():
    '''With an installed STalign first on sys.path, the tutorials' "option B" (``import STalign``
    with the STalign folder appended to sys.path) imports the package: it has the functions of
    the module ``STalign.STalign``.'''
    code = f'''
import sys
sys.path.insert(0, {REPO!r})
sys.path.append({os.path.join(REPO, "STalign")!r})
import STalign
from STalign import STalign as module
assert STalign.__file__.endswith('__init__.py'), STalign.__file__
assert STalign.STalign is module
names = {ORIGINAL_FUNCTIONS!r} + ['backend_info', 'align_cells']
missing = [n for n in names if getattr(STalign, n, None) is not getattr(module, n)]
assert not missing, missing
'''
    env = {k: v for k, v in os.environ.items() if k != 'PYTHONPATH'}
    result = subprocess.run([sys.executable, '-c', code], env=env, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr


# --------------------------------------------------------------------------- backends
@needs_cpp
@pytest.mark.parametrize('dim,control', [(2, None), (2, True), (3, None), (3, False)],
                         ids=['2d', '2d_step_control', '3d_step_control', '3d'])
def test_backends_agree(dim, control):
    # control=None: the function's default (affine step control off in 2D, on in 3D)
    if dim == 2:
        kw, fit, keys = data_2d(), STalign.LDDMM, KEYS_2D
    else:
        kw, fit, keys = data_3d(), STalign.LDDMM_3D_to_slice, KEYS_3D
    if control is not None:
        kw['affine_step_control'] = control
    cpp = fit(**kw, draw=False, backend='cpp')
    ref = fit(**kw, draw=False, backend='torch')
    auto = fit(**kw, draw=False)
    for k in keys:
        assert cpp[k].dtype == ref[k].dtype == torch.float64 and cpp[k].shape == ref[k].shape, k
        assert rel(cpp[k], ref[k]) < 1e-9, (k, rel(cpp[k], ref[k]))
        assert torch.equal(auto[k], cpp[k]), k  # auto uses the compiled engine for CPU float64
    assert all(torch.equal(x, y) for x, y in zip(cpp['xv'], ref['xv']))


def test_auto_falls_back_when_the_extension_is_missing(monkeypatch):
    # as on an install without a C++ compiler: the compiled module cannot be imported
    monkeypatch.setitem(sys.modules, 'STalign._stalign_cpp', None)
    monkeypatch.delitem(sys.modules, 'STalign._lddmm_cpp', raising=False)
    info = STalign.backend_info()
    assert info['auto'] == 'torch' and not info['cpp'] and info['cpp_error']
    kw = data_2d(niter=12)
    auto = STalign.LDDMM(**kw, draw=False)
    ref = STalign.LDDMM(**kw, draw=False, backend='torch')
    assert all(torch.equal(auto[k], ref[k]) for k in KEYS_2D)
    with pytest.raises(RuntimeError, match="backend='cpp'"):
        STalign.LDDMM(**kw, draw=False, backend='cpp')


def test_auto_falls_back_for_other_dtypes_and_devices(monkeypatch):
    # float32: the torch engine runs it
    kw = data_2d(niter=12)
    auto = STalign.LDDMM(**kw, draw=False, dtype=torch.float32)
    ref = STalign.LDDMM(**kw, draw=False, dtype=torch.float32, backend='torch')
    assert all(auto[k].dtype == torch.float32 and torch.equal(auto[k], ref[k]) for k in KEYS_2D)
    if HAVE_CPP:
        with pytest.raises(RuntimeError, match="backend='cpp'"):
            STalign.LDDMM(**kw, draw=False, dtype=torch.float32, backend='cpp')

    # another device: the compiled engine declines it, and 'auto' constructs the torch engine
    # (recorded here instead, so that no GPU is needed)
    class Recorder:
        def __init__(self, problem, num_threads):
            self.device = problem.device

    monkeypatch.setattr(_lddmm_torch, 'TorchEngine', Recorder)
    problem = C.prepare_problem(2, **dict(kw, device='meta'))
    engine = C.select_engine(problem, 'auto')
    assert isinstance(engine, Recorder) and engine.device.type == 'meta'


def test_num_threads_setting(monkeypatch):
    monkeypatch.setenv('STALIGN_NUM_THREADS', '3')
    assert STalign.backend_info()['num_threads'] == 3
    monkeypatch.delenv('STALIGN_NUM_THREADS')
    assert STalign.backend_info()['num_threads'] == torch.get_num_threads()
    kw = data_2d(niter=12)
    one = STalign.LDDMM(**kw, draw=False, num_threads=1)
    two = STalign.LDDMM(**kw, draw=False, num_threads=2)
    assert all(torch.equal(one[k], two[k]) for k in KEYS_2D)  # results do not depend on threads
    with pytest.raises(ValueError):
        STalign.LDDMM(**kw, draw=False, num_threads=0)


# --------------------------------------------------------------------------- figures
@pytest.mark.filterwarnings('error')  # a drawing error only warns
@pytest.mark.parametrize('backend', [pytest.param('cpp', marks=needs_cpp), 'torch'])
@pytest.mark.parametrize('dim,draw', [(2, 'end'), (2, True), (2, 5), (3, 'end'), (3, 4)])
def test_draw_modes_make_the_figures(backend, dim, draw):
    import matplotlib.pyplot as plt
    if dim == 2:  # an RGB target (imshow shows 1, 3 or 4 channels)
        out = STalign.LDDMM(**data_2d(niter=12, CJ=3), draw=draw, backend=backend)
    else:
        out = STalign.LDDMM_3D_to_slice(**data_3d(niter=12), draw=draw, backend=backend)
    assert torch.isfinite(out['v']).all()
    # the smoothing kernel, the 2x3 panel of images and the 1x3 panel of energies
    kernel, panel, energy = (plt.figure(n) for n in plt.get_fignums())
    assert [len(ax.images) for ax in kernel.axes] == [1]
    assert [len(ax.images) for ax in panel.axes] == [1] * 6
    assert sorted(ax.get_title() for ax in panel.axes) == sorted(
        ['space tformed source', 'contrast tformed source', 'Target', 'velocity', 'Weights', 'Error'])
    # the energies up to the last drawn iteration
    drawn = C.draw_iterations(draw, 12)[-1] + 1
    assert [len(ax.lines[0].get_xdata()) for ax in energy.axes] == [drawn] * 3
    if dim == 2:  # the 7 landmarks, transformed (3 panels) and target (2 panels)
        scatters = [c for ax in panel.axes for c in ax.collections]
        assert len(scatters) == 5 and all(len(c.get_offsets()) == 7 for c in scatters)


def test_draw_false_makes_no_figures():
    import matplotlib.pyplot as plt
    for draw in (False, None, 0):
        STalign.LDDMM(**data_2d(niter=3), draw=draw)
        assert plt.get_fignums() == []


# --------------------------------------------------------------------------- workflow
def test_tutorial_workflow():
    '''rasterize -> LDDMM -> transforms, as in the tutorials, with numpy inputs.'''
    rng = np.random.default_rng(1)
    centers = rng.uniform(200, 1800, size=(12, 2))
    cells = np.concatenate([c + rng.normal(scale=120, size=(400, 2)) for c in centers])
    xI, yI = cells[:, 0], cells[:, 1]
    theta = np.deg2rad(10)
    xJ = np.cos(theta) * xI - np.sin(theta) * yI + 80
    yJ = np.sin(theta) * xI + np.cos(theta) * yI - 40
    XI, YI, I = STalign.rasterize(xI, yI, dx=50, draw=0)
    XJ, YJ, J = STalign.rasterize(xJ, yJ, dx=50, draw=0)

    out = STalign.LDDMM([YI, XI], I, [YJ, XJ], J, niter=40, diffeo_start=20, a=400.0, epV=1e4)
    A, v, xv = out['A'], out['v'], out['xv']
    assert A.shape == (3, 3) and v.shape == (3,) + tuple(len(x) for x in xv) + (2,)
    assert v.abs().max() > 0

    pointsI = np.stack([yI, xI], 1)
    pointsIt = STalign.transform_points_source_to_target(xv, v, A, pointsI)
    assert isinstance(pointsIt, torch.Tensor) and pointsIt.shape == pointsI.shape
    # the forward map is the exact inverse of the backward map the fit uses
    back = STalign.transform_points_target_to_source(xv, v, A, pointsIt)
    assert np.abs(back.numpy() - pointsI).max() < 1e-3
    mean, std = STalign.calculate_tre(pointsIt, np.stack([yJ, xJ], 1))
    assert np.isfinite(mean) and np.isfinite(std)

    AI = STalign.transform_image_source_to_target(xv, v, A, [YI, XI], I, [YJ, XJ])
    assert AI.shape == J.shape
    phii = STalign.v_to_phii(xv, v)  # the velocity layout LDDMM returns
    assert phii.shape == (2,) + tuple(len(x) for x in xv)
    extent = STalign.extent_from_x([YJ, XJ])
    assert extent[0] < extent[1] and extent[2] > extent[3]
