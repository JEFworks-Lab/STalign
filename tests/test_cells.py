'''Tests of align_cells (STalign/_cells.py).

align_cells rasterizes two sets of cells and chains LDDMM calls on mean-pooled copies of the
rasters, each level starting from the previous level's affine and velocity field.  These tests
check the pooling, the arguments of each level (with a stand-in LDDMM), the chaining of real
LDDMM calls, a fit of cells moved by a known map, and the transform functions on the result.
'''
import inspect
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from STalign import STalign  # noqa: E402
from STalign import _cells  # noqa: E402
from STalign._lddmm_common import velocity_grid  # noqa: E402

# explicit parameters (so that the fit does not depend on LDDMM's data-driven defaults) that
# converge in a few dozen iterations on the cells below
FIT = dict(epL=1e-6, epT=10.0, epV=1e5, sigmaR=1.5e6)
LEVELS = ((4, 60), (2, 60))


def cells_and_map(seed=3):
    '''Clustered source cells (x, y), and their targets under a rotation, a shift and a warp.'''
    rng = np.random.default_rng(seed)
    centers = rng.uniform(300, 2700, size=(14, 2))
    src = np.concatenate([c + rng.normal(scale=150, size=(300, 2)) for c in centers])
    c0 = src.mean(0)
    th = np.deg2rad(8)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    tgt = (src - c0) @ R.T + c0 + [60, -40]
    tgt = tgt + 40*np.sin(tgt[:, ::-1]/700.0)
    return src, tgt


def rowcol(xy):
    '''(x, y) cell locations as the (row, col) points of the transform functions.'''
    return np.stack([xy[:, 1], xy[:, 0]], 1)


@pytest.fixture(scope='module')
def fit():
    src, tgt = cells_and_map()
    out = STalign.align_cells(src[:, 0], src[:, 1], tgt[:, 0], tgt[:, 1], levels=LEVELS, draw=False, **FIT)
    return src, tgt, out


def recording_lddmm(monkeypatch, fake=None):
    '''Replace the LDDMM that align_cells calls by fake (default: the real one), recording the
    arguments and results of each call.'''
    calls = []
    inner = fake or _cells.LDDMM

    def spy(xI, I, xJ, J, **kw):
        out = inner(xI, I, xJ, J, **kw)
        calls.append(dict(xI=xI, I=I, xJ=xJ, J=J, kw=kw, out=out))
        return out
    spy.__signature__ = inspect.signature(inner)
    monkeypatch.setattr(_cells, 'LDDMM', spy)
    return calls


# --------------------------------------------------------------------------- pooling
def test_pool():
    rng = np.random.default_rng(0)
    I = rng.random((2, 7, 10))
    x = [100.0 + 15.0*np.arange(7), -30.0 + 15.0*np.arange(10)]
    xp, Ip = _cells._pool(x, I, 3)
    # padded with edge values to 9 x 12, then 3 x 3 block means
    P = I[:, [0, 1, 2, 3, 4, 5, 6, 6, 6]][:, :, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9]]
    ref = np.array([[[P[c, 3*i:3*i + 3, 3*j:3*j + 3].mean() for j in range(4)] for i in range(3)] for c in range(2)])
    assert Ip.shape == (2, 3, 4)
    np.testing.assert_allclose(Ip, ref, rtol=1e-14)
    np.testing.assert_allclose(xp[0], 100.0 + 15.0*np.array([1.0, 4.0, 7.0]), rtol=1e-14)
    np.testing.assert_allclose(xp[1], -30.0 + 15.0*np.array([1.0, 4.0, 7.0, 10.0]), rtol=1e-14)
    # factor 1: copies
    xp, Ip = _cells._pool(x, I, 1)
    assert np.array_equal(Ip, I) and Ip is not I and all(np.array_equal(a, b) for a, b in zip(xp, x))
    with pytest.raises(ValueError, match='fewer than 2 pixels'):
        _cells._pool(x, I, 7)


# --------------------------------------------------------------------------- the levels
def fake_lddmm(xI, I, xJ, J, L=None, T=None, A=None, v=None, xv=None, niter=5000, diffeo_start=0, sigmaP=2e1,
               *, draw='end', prefit='auto', **kw):
    '''Stand-in for LDDMM: the affine is scaled by 0.5 at every level, the rest is constant.'''
    if A is None:
        A = np.eye(3)
        if L is not None:
            A[:2, :2] = L
    A = np.array(_cells._numpy(A), dtype=np.float64)
    A[:2, :2] *= 0.5
    xv = xv if xv is not None else [np.linspace(-1.0, 1.0, 4)]*2
    shape = (len(xJ[0]), len(xJ[1]))
    return dict(A=torch.as_tensor(A), v=torch.zeros(3, 4, 4, 2), xv=xv, WM=torch.ones(shape),
                WB=torch.zeros(shape), WA=torch.zeros(shape), prefit=dict(ran=v is None and prefit is not False))


def test_level_arguments(monkeypatch):
    calls = recording_lddmm(monkeypatch, fake_lddmm)
    src, tgt = cells_and_map()
    pI, pJ = rowcol(src[:3]), rowcol(tgt[:3])
    out = STalign.align_cells(src[:, 0], src[:, 1], tgt[:, 0], tgt[:, 1], levels=[(8, 3), (4, 2), (2, 1)],
                              pointsI=pI, pointsJ=pJ, sigmaP=10.0, diffeo_start=5, prefit=True, sigmaM=0.5)
    first, *later = [c['kw'] for c in calls]
    assert [kw['niter'] for kw in (first, *later)] == [3, 2, 1]
    assert [kw['sigmaP'] for kw in (first, *later)] == [80.0, 40.0, 20.0]
    assert [kw['draw'] for kw in (first, *later)] == [False, False, 'end']
    assert all(kw['sigmaM'] == 0.5 and kw['pointsI'] is pI and kw['pointsJ'] is pJ for kw in (first, *later))
    assert first['diffeo_start'] == 5 and first['prefit'] is True
    assert not {'A', 'L', 'T', 'v', 'xv'} & set(first)
    for kw, prev in zip(later, calls):
        # no pre-fit even though the previous level's v is zero (as after affine-only iterations)
        assert kw['diffeo_start'] == 0 and kw['prefit'] is False and not {'L', 'T'} & set(kw)
        assert kw['A'] is prev['out']['A'] and kw['v'] is prev['out']['v'] and kw['xv'] is prev['out']['xv']
    # the source is pooled to the target's pixel size under the current affine (|det L| = 1/4^level)
    assert [(lv['factor'], lv['source_factor']) for lv in out['levels']] == [(8, 8), (4, 8), (2, 8)]
    # the last level's result, with the first level's pre-fit record
    assert all(out[k] is calls[-1]['out'][k] for k in ('A', 'v', 'xv', 'WM', 'WB', 'WA'))
    assert out['prefit'] == dict(ran=True)
    assert out['WM'].shape == tuple(len(x) for x in out['levels'][-1]['xJ'])

    # defaults: sigmaP from LDDMM and scaled with landmarks only, draw from LDDMM, initial L and T
    calls.clear()
    L = np.array([[1.0, 0.1], [0.0, 1.0]])
    STalign.align_cells(src[:, 0], src[:, 1], tgt[:, 0], tgt[:, 1], levels=[(4, 1), (2, 1)], pointsI=pI,
                        pointsJ=pJ, L=L, T=[1.0, 2.0])
    assert [c['kw']['sigmaP'] for c in calls] == [80.0, 40.0]
    assert [c['kw']['draw'] for c in calls] == [False, 'end']
    assert calls[0]['kw']['L'] is L and calls[0]['kw']['T'] == [1.0, 2.0]
    calls.clear()
    STalign.align_cells(src[:, 0], src[:, 1], tgt[:, 0], tgt[:, 1], levels=[(4, 1), (2, 1)], sigmaP=7.0,
                        draw=True)
    assert [c['kw']['sigmaP'] for c in calls] == [7.0, 7.0]
    assert [c['kw']['draw'] for c in calls] == [False, True]


def test_mixture_is_em_by_default(monkeypatch):
    '''align_cells passes mixture='em' to every level unless told otherwise (LDDMM's default is
    None); mixture=None turns the mixture EM off, and any other value is passed as given.'''
    assert inspect.signature(STalign.align_cells).parameters['mixture'].default == 'em'
    assert inspect.signature(STalign.LDDMM).parameters['mixture'].default is None
    calls = recording_lddmm(monkeypatch, fake_lddmm)
    src, tgt = cells_and_map()
    cells = (src[:, 0], src[:, 1], tgt[:, 0], tgt[:, 1])
    opts = dict(start=10, every=2)
    for given, expected in (({}, 'em'), (dict(mixture=None), None), (dict(mixture='em'), 'em'),
                            (dict(mixture=opts), opts)):
        calls.clear()
        STalign.align_cells(*cells, levels=[(4, 1), (2, 1)], **given)
        assert len(calls) == 2 and all(c['kw']['mixture'] == expected for c in calls), given

    # the real fit: the default is LDDMM with mixture='em', mixture=None is LDDMM without it
    monkeypatch.setattr(_cells, 'LDDMM', STalign.LDDMM)
    kw = dict(levels=[(4, 60)], prefit=False, draw=False, **FIT)
    on, off = STalign.align_cells(*cells, **kw), STalign.align_cells(*cells, mixture=None, **kw)
    r = on['rasters']
    xJ, J = _cells._pool([r['YJ'], r['XJ']], r['J'], 4)
    xI, I = _cells._pool([r['YI'], r['XI']], r['I'], on['levels'][0]['source_factor'])
    lddmm = dict(niter=60, prefit=False, draw=False, **FIT)
    ref_on, ref_off = STalign.LDDMM(xI, I, xJ, J, mixture='em', **lddmm), STalign.LDDMM(xI, I, xJ, J, **lddmm)
    for out, ref in ((on, ref_on), (off, ref_off)):
        for k in ('A', 'v', 'WM', 'WB'):
            assert torch.equal(out[k], ref[k]), k
    assert not torch.equal(on['WM'], off['WM'])


def test_bad_arguments():
    src, tgt = cells_and_map()
    x, y = src[:, 0], src[:, 1]
    with pytest.raises(TypeError, match='niter'):
        STalign.align_cells(x, y, x, y, niter=10)
    for levels in ((), [(4, 1, 2)], [(0, 10)], [(2, -1)], [(2.5, 10)], [(2, 'a')], 5):
        with pytest.raises(ValueError, match='levels'):
            STalign.align_cells(x, y, x, y, levels=levels)
    with pytest.raises(ValueError, match='same length'):
        STalign.align_cells(x, y[:-1], x, y)
    with pytest.raises(ValueError, match='blur'):
        STalign.align_cells(x, y, x, y, blur_um=-1.0)
    with pytest.raises(ValueError, match='fewer than 2 pixels'):
        STalign.align_cells(x, y, x, y, levels=[(1000, 1)])


def test_levels_chain_lddmm_calls(monkeypatch):
    calls = recording_lddmm(monkeypatch)
    src, tgt = cells_and_map()
    L, T = np.array([[0.99, 0.02], [-0.02, 1.01]]), np.array([5.0, -3.0])
    out = STalign.align_cells(src[:, 0], src[:, 1], tgt[:, 0], tgt[:, 1], levels=((4, 7), (2, 5), (1, 3)),
                              L=L, T=T, diffeo_start=2, draw=False, **FIT)
    r = out['rasters']
    for a, b in zip((r['XI'], r['YI'], r['I']), STalign.rasterize(src[:, 0], src[:, 1], dx=15.0, blur_um=30.0,
                                                                     draw=0)):
        np.testing.assert_array_equal(a, b)
    for a, b in zip((r['XJ'], r['YJ'], r['J']), STalign.rasterize(tgt[:, 0], tgt[:, 1], dx=15.0, blur=1.0,
                                                                     draw=0)):
        np.testing.assert_array_equal(a, b)
    assert len(calls) == 3
    for c, lv, f in zip(calls, out['levels'], (4, 2, 1)):
        xI, I = _cells._pool([r['YI'], r['XI']], r['I'], lv['source_factor'])
        xJ, J = _cells._pool([r['YJ'], r['XJ']], r['J'], f)
        assert lv['factor'] == f and lv['source_factor'] == f  # |det L| is about 1
        assert np.array_equal(c['I'], I) and np.array_equal(c['J'], J)
        assert all(np.array_equal(a, b) for a, b in zip(c['xI'] + c['xJ'], xI + xJ))
    # each level continues from the previous one on the velocity grid of the first level's source
    assert calls[0]['kw']['L'] is L and calls[0]['kw']['diffeo_start'] == 2
    defaults = inspect.signature(STalign.LDDMM).parameters
    xv0 = velocity_grid(calls[0]['xI'], defaults['expand'].default, defaults['a'].default)
    assert all(np.array_equal(a.numpy(), b) for a, b in zip(calls[0]['out']['xv'], xv0))
    for c, prev in zip(calls[1:], calls):
        assert torch.equal(c['kw']['A'], prev['out']['A']) and torch.equal(c['kw']['v'], prev['out']['v'])
        assert c['kw']['xv'] is prev['out']['xv'] and c['kw']['diffeo_start'] == 0
    # one more step of the chain by hand gives the same result
    last = calls[-1]
    again = STalign.LDDMM(last['xI'], last['I'], last['xJ'], last['J'], **last['kw'])
    for k in ('A', 'v', 'WM', 'WB', 'WA'):
        assert torch.equal(again[k], out[k]), k


# --------------------------------------------------------------------------- the fit
def test_aligns_cells(fit):
    src, tgt, out = fit
    P, Q = rowcol(src), rowcol(tgt)
    err0 = np.linalg.norm(P - Q, axis=1).mean()
    tP = STalign.transform_points_source_to_target(out['xv'], out['v'], out['A'], P)
    err = np.linalg.norm(tP.numpy() - Q, axis=1).mean()
    assert err0 > 140 and err < 0.15*err0, (err0, err)
    assert out['A'].shape == (3, 3) and out['v'].shape == (3,) + tuple(len(x) for x in out['xv']) + (2,)
    assert out['WM'].shape == tuple(len(x) for x in out['levels'][-1]['xJ'])
    assert {'A', 'v', 'xv', 'WM', 'WB', 'WA', 'rasters', 'levels'} <= set(out)


def test_transforms_of_the_result(fit):
    src, tgt, out = fit
    xv, v, A, r = out['xv'], out['v'], out['A'], out['rasters']
    P = rowcol(src)
    tP = STalign.transform_points_source_to_target(xv, v, A, P)
    back = STalign.transform_points_target_to_source(xv, v, A, tP)
    assert np.abs(back.numpy() - P).max() < 1e-3
    xI, xJ = [r['YI'], r['XI']], [r['YJ'], r['XJ']]
    AI = STalign.transform_image_source_to_target(xv, v, A, xI, r['I'], xJ)
    assert AI.shape == r['J'].shape

    def corr(a, b):
        return np.corrcoef(np.ravel(a), np.ravel(b))[0, 1]
    A0 = torch.eye(3, dtype=torch.float64)
    AI0 = STalign.transform_image_source_to_target(xv, torch.zeros_like(v), A0, xI, r['I'], xJ)
    assert corr(AI, r['J']) > 0.9 > corr(AI0, r['J'])
    AJ = STalign.transform_image_target_to_source(xv, v, A, xJ, r['J'], xI)
    assert AJ.shape == r['I'].shape and corr(AJ, r['I']) > 0.9
    phii = STalign.build_transform(xv, v, A, XJ=xJ)
    assert phii.shape == (len(xJ[0]), len(xJ[1]), 2)
    mean, std = STalign.calculate_tre(tP, rowcol(tgt))
    assert mean < 25 and np.isfinite(std)


# --------------------------------------------------------------------------- the public name
def test_public():
    assert STalign.align_cells is _cells.align_cells
    assert STalign.align_cells.__module__ == STalign.__name__
    params = inspect.signature(STalign.align_cells).parameters
    assert [p for p in params][:4] == ['xI', 'yI', 'xJ', 'yJ']
    assert (params['dx'].default, params['blur_um'].default, params['levels'].default) == (15.0, 30.0,
                                                                                            ((4, 500), (2, 500)))


def test_option_b_import(tmp_path):
    '''The tutorials' "option B" (the STalign folder on sys.path, import STalign) has it too.'''
    code = (f'import sys\nsys.path.append({os.path.join(REPO, "STalign")!r})\nimport STalign\n'
            'assert STalign.__file__.endswith("STalign.py")\n'
            'assert callable(STalign.align_cells) and STalign.align_cells.__module__ == "STalign"\n')
    env = {k: v for k, v in os.environ.items() if k != 'PYTHONPATH'}
    result = subprocess.run([sys.executable, '-c', code], cwd=tmp_path, env=env, capture_output=True, text=True,
                            timeout=300)
    assert result.returncode == 0, result.stderr
