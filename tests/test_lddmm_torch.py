"""LDDMM front-end and pure torch engine, tested against the plain-autograd oracle (reference.py).

* gradients at non-trivial states, with sample points outside the image (border clamping)
  and outside the velocity grid (zero padding): relative error <= 1e-9
* 300-iteration trajectories: relative difference of A and v <= 1e-7
* the approved fixes (F1-F6), input conventions, draw modes, threads and backend selection
* the affine step control (on by default in 3D): trajectories with oscillating affine steps,
  segmented runs, defaults and validation
* the landmark maps ('exact', the default, and 'euler') and the mixture EM (mixture='em' or a
  dict of options): gradients, trajectories, options and validation, and the mixture EM's
  effect on a target with missing tissue
* the oracle itself against the frozen original STalign (skipped when it is not available)
"""
import dataclasses
import functools
import importlib.util
import inspect
import multiprocessing as mp
import os
import sys
import warnings

import numpy as np
import pytest
import torch

import reference as R
from STalign import _lddmm, _lddmm_common as C, _lddmm_torch as TE

GRAD_TOL = 1e-9
TRAJ_TOL = 1e-7


def relerr(a, b):
    a = torch.as_tensor(a).detach().cpu().double()
    b = torch.as_tensor(b).detach().cpu().double()
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-300))


def assert_close(ours, ref, keys, tol):
    errs = {k: relerr(ours[k], ref[k]) for k in keys}
    assert all(e <= tol for e in errs.values()), errs
    return errs


# The oracle has no affine pre-fit: the comparisons run without it (tested in test_defaults.py).
def torch_lddmm(**kw):
    return _lddmm.LDDMM(**kw, draw=False, backend='torch', prefit=False)


def torch_lddmm3d(**kw):
    return _lddmm.LDDMM_3D_to_slice(**kw, draw=False, backend='torch', prefit=False)


# --------------------------------------------------------------------------- problems
SYN2D = dict(L=R.rotation_2d(5.0), T=np.array([1.0, -0.5]), a=8.0, expand=2.0, nt=3, niter=300, diffeo_start=30,
             epL=1e-4, epT=1e-2, epV=5.0, sigmaM=0.5, sigmaB=1.0, sigmaA=2.0, sigmaR=50.0, sigmaP=2.0)
SYN3D = dict(L=np.eye(3), T=np.array([-2.0, 0.5, -0.5]), a=6.0, expand=1.25, nt=4, niter=300, epL=3e-5, epT=3e-3,
             epV=10.0, sigmaM=0.5, sigmaB=1.0, sigmaA=2.0, sigmaR=1e3)


def data_2d(CI=3, CJ=1, seed=0, points=True, **extra):
    d = R.synthetic_2d(CI=CI, CJ=CJ, seed=seed)
    if not points:
        d.pop('pointsI'), d.pop('pointsJ')
    return dict(d, **dict(SYN2D, **extra))


def data_3d(seed=0, **extra):
    return dict(R.synthetic_3d(seed=seed), **dict(SYN3D, **extra))


def state_2d(seed=1):
    """A non-trivial 2D state.  The velocity grid covers only the source image, so the target
    samples fall outside both the image and the velocity grid."""
    d = R.synthetic_2d(CI=3, CJ=2, seed=0)
    xv = [x.numpy() for x in R.velocity_grid([torch.tensor(x) for x in d['xI']], 1.0, 8.0)]
    v = R.smooth_random_v(xv, 3, 4.0, seed).numpy()
    WM = np.random.default_rng(seed).uniform(0.2, 1.0, size=d['J'].shape[1:])
    return d, dict(L=1.15 * R.rotation_2d(15.0), T=np.array([5.0, -4.0]), v=v, xv=xv), WM


def state_3d(seed=2):
    d = R.synthetic_3d(seed=0)
    xv = [x.numpy() for x in R.velocity_grid([torch.tensor(x) for x in d['xI']], 1.0, 6.0)]
    v = R.smooth_random_v(xv, 4, 3.0, seed).numpy()
    WM = np.random.default_rng(seed).uniform(0.2, 1.0, size=(1,) + d['J'].shape[1:])
    return d, dict(L=1.1 * R.rotation_3d(10.0, 6.0), T=np.array([2.0, 1.0, -1.0]), v=v, xv=xv), WM


def engine_at(dim, d, state, WM, params, crop=None, fft=None, num_threads=4):
    """TorchEngine positioned at a given state (A, v and WM)."""
    problem = C.prepare_problem(dim, d['xI'], d['I'], d['xJ'], d['J'], pointsI=d.get('pointsI'),
                                pointsJ=d.get('pointsJ'), niter=1, **state, **params)
    problem = dataclasses.replace(problem, WM=np.ascontiguousarray(WM.reshape(problem.J.shape[1:])))
    return TE.TorchEngine(problem, num_threads, crop=crop, fft=fft)


def fraction_outside(X, grid):
    out = torch.zeros(X.shape[:-1], dtype=torch.bool)
    for i, x in enumerate(grid):
        out |= (X[..., i] < float(x[0])) | (X[..., i] > float(x[-1]))
    return float(out.double().mean())


def check_gradient(ours, ref):
    errs = assert_close(ours, ref, ('Lgrad', 'Tgrad', 'vgrad', 'direction'), GRAD_TOL)
    for k in ('EM', 'ER', 'EP'):
        if ref[k] is not None:
            assert abs(float(ours[k]) - float(ref[k])) <= GRAD_TOL * abs(float(ref[k])), k
    return errs


# --------------------------------------------------------------------------- gradients
@pytest.mark.parametrize('points', [True, False])
@pytest.mark.parametrize('crop', [True, False])
@pytest.mark.parametrize('fft', ['matmul', 'pool', 'torch'])
def test_gradient_2d_multichannel(points, crop, fft):
    d, state, WM = state_2d()
    if not points:
        d = {k: val for k, val in d.items() if not k.startswith('points')}
    params = dict(a=8.0, sigmaM=0.5, sigmaR=50.0, sigmaP=2.0)
    ref = R.gradients(d['xI'], d['I'], d['xJ'], d['J'], state['L'], state['T'], state['v'], state['xv'], WM,
                      pointsI=d.get('pointsI'), pointsJ=d.get('pointsJ'), **params)
    assert 0.02 < fraction_outside(ref['Xs'], d['xI']) < 0.9
    assert 0.02 < fraction_outside(ref['Xs'], state['xv']) < 0.9
    ours = engine_at(2, d, state, WM, params, crop=crop, fft=fft).gradient()
    check_gradient(ours, ref)


@pytest.mark.parametrize('crop', [True, False])
@pytest.mark.parametrize('fft', ['pool', 'torch'])
def test_gradient_3d(crop, fft):
    d, state, WM = state_3d()
    params = dict(a=6.0, sigmaM=0.5, sigmaR=20.0)
    ref = R.gradients(d['xI'], d['I'], d['xJ'], d['J'], state['L'], state['T'], state['v'], state['xv'], WM, **params)
    assert 0.02 < fraction_outside(ref['Xs'], d['xI']) < 0.95
    assert 0.02 < fraction_outside(ref['Xs'], state['xv']) < 0.95
    ours = engine_at(3, d, state, WM, params, crop=crop, fft=fft).gradient()
    check_gradient(ours, ref)


def test_gradient_after_iterations():
    """Gradient at the state the engine reaches after 60 iterations (E-step weights included)."""
    kw = data_2d(CI=3, CJ=2, niter=60, diffeo_start=5)
    problem = C.prepare_problem(2, **kw)
    engine = TE.TorchEngine(problem, 4)
    engine.run(60)
    s = engine.snapshot()
    ref = R.gradients(kw['xI'], kw['I'], kw['xJ'], kw['J'], s['L'], s['T'], s['v'], problem.xv, s['WM'],
                      a=kw['a'], sigmaM=kw['sigmaM'], sigmaR=kw['sigmaR'], sigmaP=kw['sigmaP'],
                      pointsI=kw['pointsI'], pointsJ=kw['pointsJ'])
    check_gradient(engine.gradient(), ref)


def test_energy_history():
    """Recorded [E, EM, ER, EP] (with the tracked spectrum of v and its resync) vs the oracle."""
    for dim, kw in ((2, data_2d(niter=70, diffeo_start=5)), (3, data_3d(niter=70, affine_step_control=True))):
        ref = (R.LDDMM if dim == 2 else R.LDDMM_3D_to_slice)(**kw)
        engine = TE.TorchEngine(C.prepare_problem(dim, **kw), 4)
        engine.run(70)
        E = engine.snapshot()['Esave'].numpy()
        assert E.shape == ref['Esave'].shape
        assert np.max(np.abs(E - ref['Esave']) / np.abs(ref['Esave']).max(0)) < 1e-10


# --------------------------------------------------------------------------- trajectories
def test_trajectory_2d_points_int_mu_diffeo_start():
    kw = data_2d(CI=3, CJ=1, muA=torch.tensor([1]), muB=torch.tensor([0]))
    ref = R.LDDMM(**kw)
    assert_close(torch_lddmm(**kw), ref, ('A', 'v', 'WM', 'WA', 'WB'), TRAJ_TOL)


def test_trajectory_2d_three_target_channels_float32_and_list_mu_initial_A_v():
    d = R.synthetic_2d(CI=1, CJ=3, seed=4)
    xv = R.velocity_grid([torch.tensor(x) for x in d['xI']], 2.0, 8.0)
    A0 = np.eye(3)
    A0[:2, :2], A0[:2, 2] = R.rotation_2d(3.0), [0.5, -0.5]
    kw = dict(xI=d['xI'], I=d['I'], xJ=d['xJ'], J=d['J'], A=torch.tensor(A0), v=R.smooth_random_v(xv, 3, 1.5, 7),
              xv=xv, muA=torch.tensor([0.7, 0.6, 0.7], dtype=torch.float32), muB=[0, 0, 0.1],
              **{k: val for k, val in SYN2D.items() if k not in ('L', 'T', 'diffeo_start')})
    kw['niter'] = 150
    assert_close(torch_lddmm(**kw), R.LDDMM(**kw), ('A', 'v', 'WM', 'WA', 'WB'), TRAJ_TOL)


def test_trajectory_3d_two_channel_source():
    kw = data_3d(muA=torch.tensor([3]), muB=[0.0])
    ref = R.LDDMM_3D_to_slice(**kw)
    assert_close(torch_lddmm3d(**kw), ref, ('A', 'v', 'WM', 'WA', 'WB', 'Xs'), TRAJ_TOL)


@pytest.mark.parametrize('control', [True, False])
def test_trajectory_3d_diffeo_start_estimated_mu(control):
    kw = data_3d(seed=1, niter=80, diffeo_start=20, affine_step_control=control)
    ref = R.LDDMM_3D_to_slice(**kw)
    assert_close(torch_lddmm3d(**kw), ref, ('A', 'v', 'WM', 'WA', 'WB', 'Xs'), TRAJ_TOL)


def test_lattice_aligned_start():
    """Exact affine, v = 0 and target pixels on source grid lines: interpolation derivatives are
    one-sided there, so the first image sample grid must be rounded like the original's."""
    d = R.synthetic_2d(CI=3, CJ=1, seed=2)
    xJ = [np.arange(40) * 1.0 - 20.0, np.arange(36) * 1.25 - 24.0]  # whole source pixels apart
    J = R._blobs(np.meshgrid(*xJ, indexing='ij'), np.random.default_rng(3).uniform(-18, 18, size=(10, 2)),
                 np.random.default_rng(4).uniform(3, 7, size=10), np.random.default_rng(5).uniform(0.2, 1, size=(10, 1)))
    kw = dict(xI=d['xI'], I=d['I'], xJ=xJ, J=J, a=8.0, niter=6, epL=1e-4, epT=1e-2, epV=5.0, sigmaM=0.5, sigmaR=50.0)
    assert_close(torch_lddmm(**kw), R.LDDMM(**kw), ('A', 'v'), 1e-12)
    d3 = R.synthetic_3d(seed=3)
    xJ3 = [np.arange(30) * 1.5 - 15.75, np.arange(32) * 1.25 - 15.625]  # on the source y/x lattice
    kw3 = dict(xI=d3['xI'], I=d3['I'], xJ=xJ3, J=np.random.default_rng(4).uniform(0.5, 1.5, size=(1, 30, 32)),
               L=np.eye(3), T=np.array([-d3['xI'][0][8], 0.0, 0.0]), a=6.0, nt=4, niter=6, epL=3e-5, epT=3e-3,
               epV=10.0, sigmaM=0.5, sigmaR=1e5)
    assert_close(torch_lddmm3d(**kw3), R.LDDMM_3D_to_slice(**kw3), ('A', 'v', 'Xs'), 1e-12)


# --------------------------------------------------------------------------- affine step control
FITS = {2: (functools.partial(_lddmm.LDDMM, prefit=False), R.LDDMM),
        3: (functools.partial(_lddmm.LDDMM_3D_to_slice, prefit=False), R.LDDMM_3D_to_slice)}


def result_keys(dim):
    return ('A', 'v', 'WM', 'WA', 'WB') + (('Xs',) if dim == 3 else ())


@pytest.mark.parametrize('dim', [2, 3])
def test_trajectory_affine_step_control(dim):
    """The affine step control vs the oracle, with steps large enough that the affine
    parameters oscillate (their multipliers shrink, some to the floor, and grow back),
    diffeo_start > 0, landmarks in 2D, and runs split into segments (draw=N, engine.run(n))."""
    kw = R.step_control_problem(dim)
    fit, oracle = FITS[dim]
    ref = oracle(**kw, affine_step_control=True)
    m = ref['step_multipliers']
    assert m.min() < 0.05 and float(ref['v'].abs().max()) > 0.5  # the control and the deformation are active
    out = fit(**kw, draw=False, backend='torch', affine_step_control=True)
    assert_close(out, ref, result_keys(dim), TRAJ_TOL)
    # segments between the draws (iterations 0, 70, 140, ...): the control's state persists
    segmented = fit(**kw, draw=70, backend='torch', affine_step_control=True)
    assert all(torch.equal(segmented[k], out[k]) for k in result_keys(dim))
    engine = TE.TorchEngine(C.prepare_problem(dim, **kw, affine_step_control=True), 4)
    for n in (1, 12, 37, 50, 200):
        engine.run(n)
    snap = engine.snapshot()
    assert torch.equal(snap['A_eval'], out['A']) and torch.equal(snap['v_eval'], out['v'])
    assert_close(snap, {'step_multipliers': m}, ('step_multipliers',), TRAJ_TOL)


def test_affine_step_control_defaults_and_validation(monkeypatch):
    """On by default in LDDMM_3D_to_slice and off in LDDMM; the oracle's 3D default sigmaR is the
    package's legacy default 2e6 (the package's default is data-driven); the oracle uses the
    package's constants; the multipliers are in the snapshot only when on."""
    p2 = inspect.signature(_lddmm.LDDMM).parameters
    p3 = inspect.signature(_lddmm.LDDMM_3D_to_slice).parameters
    assert p2['affine_step_control'].default is False and p3['affine_step_control'].default is True
    assert p2['affine_step_control'].kind == p3['affine_step_control'].kind == inspect.Parameter.KEYWORD_ONLY
    assert p3['sigmaR'].default is None and p2['sigmaR'].default is None
    assert C.LEGACY_DEFAULTS[3]['sigmaR'] == R.SIGMAR_3D == 2e6 and C.LEGACY_DEFAULTS[2]['sigmaR'] == 5e5
    assert ((C.AFFINE_STEP_SHRINK, C.AFFINE_STEP_GROW, C.AFFINE_STEP_FLOOR, C.AFFINE_STEP_CAP)
            == (R.SHRINK, R.GROW, R.FLOOR, R.CAP) == (0.5, 1.05, 1e-3, 1.0))
    for dim, default in ((2, False), (3, True)):
        kw = dict(R.step_control_problem(dim), niter=10)
        fit = FITS[dim][0]
        base = fit(**kw, draw=False, backend='torch')
        same = fit(**kw, draw=False, backend='torch', affine_step_control=default)
        other = fit(**kw, draw=False, backend='torch', affine_step_control=not default)
        assert all(torch.equal(base[k], same[k]) for k in result_keys(dim))
        assert not torch.equal(base['A'], other['A'])
        for control in (False, True):
            engine = TE.TorchEngine(C.prepare_problem(dim, **kw, affine_step_control=control), 4)
            engine.run(2)
            assert ('step_multipliers' in engine.snapshot()) == control
    kw = dict(R.step_control_problem(2), niter=2)
    for bad in ('yes', None, 1.0):
        with pytest.raises(ValueError, match='affine_step_control'):
            _lddmm.LDDMM(**kw, draw=False, affine_step_control=bad)
    # the constants are read from _lddmm_common when the problem is prepared
    monkeypatch.setattr(C, 'AFFINE_STEP_GROW', 1.0)
    assert C.prepare_problem(2, **kw).step_grow == 1.0
    monkeypatch.setattr(C, 'AFFINE_STEP_GROW', 0.9)
    with pytest.raises(ValueError, match='AFFINE_STEP_GROW'):
        C.prepare_problem(2, **kw)


# --------------------------------------------------------------------------- approved fixes
def test_returns_last_evaluated_pair_and_niter0():
    """F2: A and v are the pair of the last forward pass; niter=0 returns the initial state."""
    kw = data_2d(niter=12, diffeo_start=0)
    ref = R.LDDMM(**kw)
    out = torch_lddmm(**kw)
    assert relerr(out['v'], ref['v']) < 1e-12 and relerr(out['v'], ref['v_next']) > 1e-3
    out0 = torch_lddmm(**dict(kw, niter=0))
    A0 = np.eye(3)
    A0[:2, :2], A0[:2, 2] = kw['L'], kw['T']
    assert torch.equal(out0['A'], torch.tensor(A0)) and not out0['v'].any()
    assert torch.equal(out0['WM'], torch.full(kw['J'].shape[1:], 0.5, dtype=torch.float64))
    kw3 = data_3d(niter=0)
    out3, ref3 = torch_lddmm3d(**kw3), R.LDDMM_3D_to_slice(**kw3)
    assert out3['Xs'].shape == (1,) + kw3['J'].shape[1:] + (3,)
    assert relerr(out3['Xs'], ref3['Xs']) < 1e-14


def test_mu_coercion():
    """F3: one value per channel; size 1 broadcasts; equal entries warn; other sizes raise."""
    for mu in (2, 2.0, [2], np.array([2.0]), torch.tensor([2]), torch.tensor(2.0, dtype=torch.float32)):
        np.testing.assert_array_equal(C.coerce_mu(mu, 3, 'muA'), [2.0, 2.0, 2.0])
    np.testing.assert_array_equal(C.coerce_mu(torch.tensor([1, 2, 3]), 3, 'muA'), [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(C.coerce_mu(torch.tensor([0.7], dtype=torch.float32), 1, 'muA'),
                                  [np.float64(np.float32(0.7))])
    assert C.coerce_mu(None, 3, 'muA') is None
    with pytest.warns(UserWarning, match='muA has 3 entries'):
        np.testing.assert_array_equal(C.coerce_mu([3, 3, 3], 1, 'muA'), [3.0])
    for bad in ([1, 2], [1, 2, 3, 4], []):
        with pytest.raises(ValueError):
            C.coerce_mu(bad, 3, 'muB')


def test_mu_broadcast_quirk_in_3d():
    """F3 through the API: a 3-entry mean against a 1-channel target warns and equals a 1-entry mean."""
    kw = data_3d(niter=60)
    with pytest.warns(UserWarning, match='muA has 3 entries'):
        a = torch_lddmm3d(**kw, muA=torch.tensor([3, 3, 3]), muB=torch.tensor([0]))
    b = torch_lddmm3d(**kw, muA=3.0, muB=0)
    assert all(torch.equal(a[k], b[k]) for k in ('A', 'v', 'WM', 'WA', 'WB'))


def test_3d_regularizer_gradient_is_proportional_to_v():
    """F1: with the 3-axis regularizer, K * dER/dv == DV / sigmaR^2 * v exactly."""
    d, state, WM = state_3d()
    ref = R.gradients(d['xI'], d['I'], d['xJ'], d['J'], state['L'], state['T'], state['v'], state['xv'], WM,
                      a=6.0, sigmaR=20.0)
    LL, K, DV = R.kernels([torch.tensor(x) for x in state['xv']], 6.0, 2.0)
    smoothed_ER = R.smooth(ref['vgrad_total'] - ref['vgrad'], K)
    assert relerr(smoothed_ER, DV / 20.0 ** 2 * torch.tensor(state['v'])) < 1e-12


def test_3d_points_warn():
    """F4: there is no landmark term in 3D."""
    kw = data_3d(niter=1)
    with pytest.warns(UserWarning, match='no landmark matching term'):
        torch_lddmm3d(**kw, pointsI=np.zeros((2, 3)), pointsJ=np.zeros((2, 2)))


@pytest.mark.parametrize('draw', ['end', True, 4, 0, False, None])
def test_draw_modes(draw):
    """F5: drawing never changes the results; figures only when asked."""
    import matplotlib.pyplot as plt
    kw = data_2d(niter=12, diffeo_start=0)
    base = torch_lddmm(**kw)
    plt.close('all')
    out = _lddmm.LDDMM(**kw, draw=draw, backend='torch')
    assert all(torch.equal(out[k], base[k]) for k in ('A', 'v', 'WM', 'WA', 'WB'))
    assert len(plt.get_fignums()) == (3 if draw else 0)


def test_draw_3d_and_draw_errors():
    import matplotlib.pyplot as plt
    kw = data_3d(niter=6)
    out = _lddmm.LDDMM_3D_to_slice(**kw, draw=5, backend='torch', prefit=False)
    assert torch.equal(out['v'], torch_lddmm3d(**kw)['v'])
    assert len(plt.get_fignums()) == 3
    # the figures accept 1, 3 or 4 channels only: a 2-channel target warns and the fit completes
    kw2 = data_2d(CI=1, CJ=2, niter=3, points=False)
    with pytest.warns(UserWarning, match='drawing failed'):
        out = _lddmm.LDDMM(**kw2, draw='end', backend='torch')
    assert torch.isfinite(out['v']).all()
    with pytest.raises(ValueError):
        _lddmm.LDDMM(**kw2, draw='sometimes')


# --------------------------------------------------------------------------- landmark maps
def fit_problem(dim=2, backend='torch', **kw):
    """Fit through the front-end with Problem options that LDDMM does not take as arguments."""
    return C.run_lddmm(C.prepare_problem(dim, **kw), draw=False, num_threads=4, backend=backend)


def test_gradient_euler_landmark_map():
    """The Euler map of earlier versions (the exact map is the default in the other tests)."""
    d, state, WM = state_2d()
    params = dict(a=8.0, sigmaM=0.5, sigmaR=50.0, sigmaP=2.0)
    ref = R.gradients(d['xI'], d['I'], d['xJ'], d['J'], state['L'], state['T'], state['v'], state['xv'], WM,
                      pointsI=d['pointsI'], pointsJ=d['pointsJ'], landmark_map='euler', **params)
    assert relerr(ref['pointsIt'], R.gradients(d['xI'], d['I'], d['xJ'], d['J'], state['L'], state['T'], state['v'],
                                               state['xv'], WM, pointsI=d['pointsI'], pointsJ=d['pointsJ'],
                                               **params)['pointsIt']) > 1e-4  # the maps differ
    engine = engine_at(2, d, state, WM, dict(params, landmark_map='euler'))
    check_gradient(engine.gradient(), ref)


def test_exact_landmark_map_is_the_returned_forward_map():
    """The landmark term moves pointsI with the map transform_points_source_to_target applies to
    the fitted A and v (to its 1e-6 cell tolerance), unlike the Euler map."""
    from STalign import STalign as S
    kw = data_2d(niter=80, diffeo_start=0)
    cell = min(float(x[1] - x[0]) for x in C.velocity_grid(kw['xI'], kw['expand'], kw['a']))
    for landmark_map, close in (('exact', True), ('euler', False)):
        engine = TE.TorchEngine(C.prepare_problem(2, **kw, landmark_map=landmark_map), 4)
        engine.run(80)
        snap = engine.snapshot()
        mapped = S.transform_points_source_to_target(engine.problem.xv, snap['v_eval'], snap['A_eval'], kw['pointsI'])
        err = float((mapped - snap['pointsIt']).abs().max())
        assert (err < 1e-5 * cell) == close, (landmark_map, err)
    assert snap.get('landmark_unconverged') is None  # counted for the exact map only


def test_trajectory_euler_landmark_map():
    kw = data_2d(niter=200, diffeo_start=20)
    assert_close(fit_problem(**kw, landmark_map='euler'), R.LDDMM(**kw, landmark_map='euler'),
                 ('A', 'v', 'WM', 'WA', 'WB'), TRAJ_TOL)


def unconverged_problem(nt=1):
    """Landmarks in a velocity field too rough for its time steps to be invertible."""
    d = R.synthetic_2d(CI=1, CJ=1, seed=0)
    xv = [x.numpy() for x in R.velocity_grid([torch.tensor(x) for x in d['xI']], 1.0, 8.0)]
    v = np.random.default_rng(0).uniform(-20.0, 20.0, size=(nt,) + tuple(len(x) for x in xv) + (2,))
    return dict(d, v=v, xv=xv, a=8.0, niter=1, sigmaP=2.0)


def test_unconverged_landmark_map_warns():
    kw = unconverged_problem()
    engine = TE.TorchEngine(C.prepare_problem(2, **kw), 4)
    engine.run(1)
    assert engine.snapshot()['landmark_unconverged'] > 0
    with pytest.warns(RuntimeWarning, match='exact landmark map did not converge'):
        out = fit_problem(**kw)
    assert torch.isfinite(out['A']).all()
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        fit_problem(**kw, landmark_map='euler')


def test_landmark_map_validation():
    kw = data_2d(niter=1)
    assert C.prepare_problem(2, **kw).landmark_map == 'exact' == C.Problem.landmark_map
    for bad in ('Exact', 'implicit', None, 1):
        with pytest.raises(ValueError, match='landmark_map'):
            C.prepare_problem(2, **kw, landmark_map=bad)


# --------------------------------------------------------------------------- mixture EM
MIX_MAB = dict(classes='MAB', start=20, every=3, fix_muB=False)


@pytest.mark.parametrize('mixture,extra', [('em', {}), (MIX_MAB, dict(niter=80)),
                                           (dict(start=10, pi_min=0.8, pi_init=(2, 1, 1)), dict(muB=[0.1, 0.0]))],
                         ids=['em', 'MAB_estimated_means', 'user_muB'])
def test_trajectory_mixture(mixture, extra):
    """The mixture EM vs the oracle: 'em' over 200 iterations (E-steps from 50); the artifact
    class with both means estimated; a user muB (fixed) and other priors.  (Longer runs of the
    second one are chaotic: differences of 1e-12 grow to 1e-2 by iteration 200.)"""
    kw = data_2d(CI=3, CJ=2, **dict(dict(niter=200), **extra))
    ref = R.LDDMM(**kw, mixture=mixture)
    out = _lddmm.LDDMM(**kw, mixture=mixture, draw=False, backend='torch')
    assert_close(out, ref, ('A', 'v', 'WM', 'WA', 'WB'), TRAJ_TOL)
    classes = mixture.get('classes', 'MB') if isinstance(mixture, dict) else 'MB'
    assert float(ref['WM'].min()) < 0.999 and float(ref['WB'].max()) > 1e-3  # the E-steps ran
    assert (float(ref['WA'].max()) > 0.0) == ('A' in classes)


def test_mixture_options_and_initial_state():
    J = R.synthetic_2d(CJ=2)['J']
    assert C.mixture_options(None) is None
    assert C.mixture_options('em') == dict(C.MIXTURE_EM, pi_init=(0.9, 0.05, 0.05)) and C.MIXTURE_EM == R.MIXTURE_EM
    o = C.mixture_options(dict(start=np.int64(7), every=2, classes='MAB', pi_min=0, pi_init=[2, 1, 1], fix_muA=True))
    assert (o['start'], o['every'], o['classes'], o['pi_min'], o['fix_muA'], o['fix_muB']) == (7, 2, 'MAB', 0.0, True, True)
    assert o['pi_init'] == (0.5, 0.25, 0.25)
    bad = [dict(begin=3), dict(start=-1), dict(start=2.0), dict(every=0), dict(every=True), dict(classes='AB'),
           dict(pi_min=1.0), dict(pi_min=-0.1), dict(pi_init=(1, 1)), dict(pi_init=(1, 0, 1)), dict(fix_muB=1)]
    for b in bad + ['EM', 1, True]:
        with pytest.raises(ValueError, match='mixture'):
            C.mixture_options(b)
    # initial weights and class means (the oracle computes them independently)
    kw = data_2d(CJ=2, niter=0)
    for muA, muB in ((None, None), (2.0, [0.1, 0.2])):
        p = C.prepare_problem(2, **kw, muA=muA, muB=muB, mixture='em')
        assert (p.WM == 1).all() and not p.WA.any() and not p.WB.any()
        s = R.mixture_state('em', torch.tensor(p.J), muA, muB, 0.5, 2.0, 1.0)
        assert relerr(p.mixture['muA'], s['muA']) < 1e-15 and np.array_equal(p.mixture['muB'], s['muB'].numpy())
    assert np.array_equal(p.mixture['muB'], [0.1, 0.2])
    out = _lddmm.LDDMM(**kw, mixture='em', draw=False, backend='torch')  # niter=0: the initial state
    assert (out['WM'] == 1).all() and not out['WB'].any()
    # 2D only
    with pytest.raises(NotImplementedError, match='mixture'):
        torch_lddmm3d(**data_3d(niter=1), mixture='em')
    with pytest.raises(NotImplementedError, match='mixture'):
        C.prepare_problem(3, **data_3d(niter=1), mixture='em')
    assert inspect.signature(_lddmm.LDDMM).parameters['mixture'].kind == inspect.Parameter.KEYWORD_ONLY


def test_mixture_off_keeps_the_original_weights():
    """mixture=None gives the original E-step, bitwise the same as without the argument."""
    kw = data_2d(niter=60, points=False)
    a, b = torch_lddmm(**kw), torch_lddmm(**kw, mixture=None)
    assert all(torch.equal(a[k], b[k]) for k in ('A', 'v', 'WM', 'WA', 'WB'))
    assert not torch.equal(a['WM'], torch_lddmm(**kw, mixture='em')['WM'])


def test_mixture_drops_the_weight_of_missing_target_tissue():
    """A target that lacks the left part of the source's tissue (the rest is the source plus
    noise, so the true map is the identity): the mixture EM gives that part a matching weight
    near 0 and keeps the rest near 1, so the source tissue is not pulled into the gap, while
    the original weights stay near 1 everywhere and the remaining tissue moves by ~30 um."""
    from STalign import STalign as S
    rng = np.random.default_rng(4)
    x = [np.arange(40) * 30.0 - 585.0, np.arange(40) * 30.0 - 585.0]
    X = np.meshgrid(*x, indexing='ij')
    centers, widths = rng.uniform(-350, 350, size=(14, 2)), rng.uniform(60, 120, size=14)
    I = R._blobs(X, centers, widths, rng.uniform(0.5, 1.0, size=(14, 1)))
    missing = X[1] < -150
    J = np.where(missing, 0.0, I) + 0.02 * rng.standard_normal(I.shape)
    tissue = I[0] > 0.3
    kept = np.stack([X[0][tissue & ~missing], X[1][tissue & ~missing]], 1)
    fit = dict(niter=200, epL=1e-7, epT=1.0, epV=2e4, sigmaR=5e5)  # stable steps for these small images
    WM, err = {}, {}
    for mixture in (None, 'em'):
        out = torch_lddmm(xI=x, I=I, xJ=x, J=J, mixture=mixture, **fit)
        WM[mixture] = out['WM'].numpy()
        moved = S.transform_points_source_to_target(out['xv'], out['v'], out['A'], kept).numpy()
        err[mixture] = np.linalg.norm(moved - kept, axis=1).mean()
    assert WM['em'][tissue & missing].mean() < 0.05 and WM['em'][tissue & ~missing].mean() > 0.95
    assert WM[None][tissue & missing].mean() > 0.95
    assert err['em'] < 5.0 and err[None] > 20.0, err


# --------------------------------------------------------------------------- threads
def test_resolve_num_threads(monkeypatch):
    monkeypatch.delenv('STALIGN_NUM_THREADS', raising=False)
    assert C.resolve_num_threads() == torch.get_num_threads()
    monkeypatch.setenv('STALIGN_NUM_THREADS', '3')
    assert C.resolve_num_threads() == 3
    assert C.resolve_num_threads(2) == 2
    for bad in (0, -1, 'x', True):
        with pytest.raises(ValueError):
            C.resolve_num_threads(bad)
    monkeypatch.setenv('STALIGN_NUM_THREADS', 'many')
    with pytest.raises(ValueError, match='STALIGN_NUM_THREADS'):
        C.resolve_num_threads()
    assert TE.choose_threads(10 ** 7, 3, 5) == 5 and TE.choose_threads(100, 2, 8) == 1


def test_threads_capped_restored_and_results_invariant(monkeypatch):
    """F6: the engine never uses more threads than the cap (num_threads, else torch's setting),
    restores torch's setting, and its result does not depend on the thread count."""
    kw = dict(R.synthetic_2d(CI=2, CJ=1, seed=0, shape_J=(150, 132)), **dict(SYN2D, niter=12, diffeo_start=0))
    seen = []
    iterate = TE.TorchEngine._iterate
    monkeypatch.setattr(TE.TorchEngine, '_iterate', lambda self: (seen.append(torch.get_num_threads()), iterate(self)))
    monkeypatch.delenv('STALIGN_NUM_THREADS', raising=False)
    before = torch.get_num_threads()
    outs = {}
    for n in (1, 2, 16):
        seen.clear()
        outs[n] = _lddmm.LDDMM(**kw, draw=False, backend='torch', num_threads=n)
        assert set(seen) == {min(n, 4)}  # 19800 target pixels: the size heuristic picks 4
        assert torch.get_num_threads() == before
    torch.set_num_threads(2)
    try:
        seen.clear()
        _lddmm.LDDMM(**dict(kw, niter=2), draw=False, backend='torch')
        assert set(seen) == {2} and torch.get_num_threads() == 2
    finally:
        torch.set_num_threads(before)
    for k in ('A', 'v', 'WM'):
        assert torch.equal(outs[1][k], outs[2][k]) and torch.equal(outs[1][k], outs[16][k]), k


def _fft_in_child(queue):
    # Linux torch wheels use GNU OpenMP (libgomp), which is not fork-safe: once the parent has run a
    # multi-threaded torch op, the child's first multi-threaded op deadlocks, inside torch and
    # whatever STalign does (torch's DataLoader workers set 1 thread for the same reason).  This
    # test is about STalign's FFT pool, which is called with 2 workers explicitly below, so the
    # child's own torch ops run on 1 thread.
    torch.set_num_threads(1)
    x = torch.arange(96.0).reshape(6, 4, 4)
    queue.put(bool(torch.equal(TE._pool_map(torch.fft.rfft2, x, 2), torch.fft.rfft2(x))))


@pytest.mark.skipif('fork' not in mp.get_all_start_methods(), reason='needs the fork start method')
@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_fft_pool_after_fork():
    """F6: a forked child must not reuse the parent's (dead) pool workers."""
    x = torch.randn(6, 8, 8, dtype=torch.float64)
    assert torch.equal(TE._pool_map(torch.fft.rfft2, x, 2), torch.fft.rfft2(x))  # parent creates the pool
    ctx = mp.get_context('fork')
    q = ctx.Queue()
    p = ctx.Process(target=_fft_in_child, args=(q,))
    p.start()
    p.join(60)
    if p.is_alive():
        p.kill()
        pytest.fail('child process hung on the FFT pool')
    assert p.exitcode == 0 and q.get(timeout=5)


# --------------------------------------------------------------------------- backends
def test_backend_selection(monkeypatch):
    kw = data_2d(niter=3, points=False)
    problem = C.prepare_problem(2, **kw)
    with pytest.raises(ValueError):
        C.select_engine(problem, 'fast')
    assert isinstance(C.select_engine(problem, 'torch'), TE.TorchEngine)
    info = C.backend_info()
    assert info['auto'] in ('cpp', 'torch') and info['cpp'] == (info['cpp_error'] is None)

    class Declines:
        def __init__(self, problem, num_threads):
            raise C.Unsupported('not this one')

    real = C._sibling

    def fake(name, missing):
        if name != '_lddmm_cpp':
            return real(name)
        if missing:
            raise ImportError('no compiled extension')
        return type('FakeModule', (), {'CppEngine': Declines})

    for missing in (True, False):
        monkeypatch.setattr(C, '_sibling', lambda name: fake(name, missing))
        assert isinstance(C.select_engine(problem, 'auto'), TE.TorchEngine)
        with pytest.raises(RuntimeError, match="backend='cpp'"):
            C.select_engine(problem, 'cpp')


# --------------------------------------------------------------------------- inputs and outputs
def test_input_conventions():
    kw = data_2d(niter=8, diffeo_start=2)
    base = torch_lddmm(**kw)
    kwt = dict(kw)
    for k in ('I', 'J', 'pointsI', 'pointsJ', 'L', 'T'):
        kwt[k] = torch.tensor(kw[k])
    kwt['xI'] = [torch.tensor(x) for x in kw['xI']]
    kwt['xJ'] = [list(x) for x in kw['xJ']]
    out = torch_lddmm(**kwt)
    assert all(torch.equal(out[k], base[k]) for k in ('A', 'v', 'WM'))
    A0 = np.eye(3)
    A0[:2, :2], A0[:2, 2] = kw['L'], kw['T']
    kwa = {k: val for k, val in kw.items() if k not in ('L', 'T')}
    assert torch.equal(torch_lddmm(**kwa, A=A0)['v'], base['v'])
    with pytest.raises(Exception, match='must not specify L or T'):
        torch_lddmm(**kw, A=A0)
    with pytest.raises(Exception, match='must input both xv and v'):
        torch_lddmm(**kw, v=np.zeros((3, 4, 4, 2)))
    with pytest.raises(Exception, match='corresponding sets of points'):
        torch_lddmm(**dict(kw, pointsJ=None))
    with pytest.raises(ValueError, match='does not match'):
        torch_lddmm(**dict(kw, J=kw['J'][:, 1:]))
    # 3D: J with or without the slice axis, xJ with 2 or 3 arrays
    kw3 = data_3d(niter=5)
    a = torch_lddmm3d(**kw3)
    b = torch_lddmm3d(**dict(kw3, J=kw3['J'][:, None], xJ=[np.zeros(1)] + kw3['xJ']))
    assert all(torch.equal(a[k], b[k]) for k in ('A', 'v', 'Xs'))


def test_output_contract():
    kw = data_2d(niter=3)
    out = torch_lddmm(**kw)
    nv = tuple(len(x) for x in out['xv'])
    assert list(out) == ['A', 'v', 'xv', 'WM', 'WB', 'WA', 'prefit', 'settings']
    assert out['A'].shape == (3, 3) and out['v'].shape == (3,) + nv + (2,)
    assert all(out[k].shape == kw['J'].shape[1:] for k in ('WM', 'WB', 'WA'))
    assert all(isinstance(x, torch.Tensor) and x.dtype == torch.float64 for x in out['xv'])
    kw3 = data_3d(niter=3)
    out = torch_lddmm3d(**kw3)
    assert list(out) == ['A', 'v', 'xv', 'WM', 'WB', 'WA', 'Xs', 'prefit', 'settings']
    H, W = kw3['J'].shape[1:]
    assert out['A'].shape == (4, 4) and out['Xs'].shape == (1, H, W, 3) and out['WM'].shape == (1, H, W)
    assert all(out[k].dtype == torch.float64 and out[k].device.type == 'cpu' for k in ('A', 'v', 'WM', 'WB', 'WA', 'Xs'))


def test_float32_and_padding():
    kw = data_2d(niter=40, diffeo_start=5)
    out32 = _lddmm.LDDMM(**kw, draw=False, backend='torch', dtype=torch.float32)
    out64 = torch_lddmm(**kw)
    assert out32['v'].dtype == torch.float32
    assert relerr(out32['A'], out64['A']) < 1e-4 and relerr(out32['v'], out64['v']) < 1e-3
    # the target (50 x 44) is not a multiple of the point chunks: the padding must stay inert even
    # when the allocator recycles memory full of NaNs
    for _ in range(3):
        junk = torch.full((4_000_000,), float('nan'), dtype=torch.float64)
        del junk
    assert torch.isfinite(torch_lddmm(**kw)['v']).all()


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='no MPS device')
def test_mps_float32_smoke():
    kw = data_2d(niter=20, diffeo_start=5)
    try:
        out = _lddmm.LDDMM(**kw, draw=False, device='mps', dtype=torch.float32)
    except RuntimeError as e:
        # The virtual GPUs of hosted macOS CI runners report MPS but their Metal compiler rejects
        # some of torch's own kernels; that is the machine, not STalign.
        if 'pipeline state' in str(e) or 'CompilerError' in str(e):
            pytest.skip(f'the Metal compiler of this machine rejects a torch kernel: {str(e)[:80]}')
        raise
    assert out['v'].device.type == 'mps'
    assert relerr(out['A'].cpu(), torch_lddmm(**kw)['A']) < 1e-4


# --------------------------------------------------------------------------- the oracle itself
def _original():
    """Version 1.0's STalign.py, if available (STALIGN_ORIGINAL or dev/ref/stalign_orig.py)."""
    path = os.environ.get('STALIGN_ORIGINAL') or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dev', 'ref', 'stalign_orig.py')
    if not os.path.exists(path):
        pytest.skip('the frozen original STalign.py is not available (set STALIGN_ORIGINAL)')
    spec = importlib.util.spec_from_file_location('stalign_original', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.filterwarnings('ignore')
def test_oracle_reproduces_original_bitwise():
    """Where no fix applies the oracle is the original, bit for bit (60 iterations, so the E-step
    runs).  F2: the original's 'v' is the oracle's 'v_next'.  F1: compared with legacy_er3d, and
    3D without the affine step control (the original had none).  The landmarks move by the
    original's Euler steps (landmark_map='euler')."""
    S = _original()
    kw = data_2d(CI=3, CJ=3, niter=62, diffeo_start=7, muA=torch.tensor([1, 1, 1]))
    o, r = S.LDDMM(**kw), R.LDDMM(**kw, landmark_map='euler')
    assert torch.equal(r['A'], o['A']) and torch.equal(r['v_next'], o['v'])
    assert all(torch.equal(r[k], o[k]) for k in ('WM', 'WA', 'WB'))
    kw3 = data_3d(niter=57, diffeo_start=3, sigmaR=1e5, muB=torch.tensor([0]))
    o, r = S.LDDMM_3D_to_slice(**kw3), R.LDDMM_3D_to_slice(**kw3, legacy_er3d=True, affine_step_control=False)
    assert torch.equal(r['A'], o['A']) and torch.equal(r['v_next'], o['v'])
    assert all(torch.equal(r[k], o[k]) for k in ('WM', 'WA', 'WB', 'Xs'))


def test_oracle_f1_changes_only_the_regularizer():
    """F1 in the oracle: the L, T and matching gradients are unchanged; the v gradient changes
    by exactly the difference of the two regularizer gradients."""
    d, state, WM = state_3d()
    args = (d['xI'], d['I'], d['xJ'], d['J'], state['L'], state['T'], state['v'], state['xv'], WM)
    new = R.gradients(*args, a=6.0, sigmaR=20.0)
    old = R.gradients(*args, a=6.0, sigmaR=20.0, legacy_er3d=True)
    for k in ('Lgrad', 'Tgrad', 'vgrad'):
        assert torch.equal(new[k], old[k]), k
    LL, K, DV = R.kernels([torch.tensor(x) for x in state['xv']], 6.0, 2.0)
    v = torch.tensor(state['v'], requires_grad=True)
    dER = [torch.autograd.grad(R.regularizer(v, LL, DV, 20.0, legacy), v)[0] for legacy in (False, True)]
    assert relerr(new['vgrad_total'] - old['vgrad_total'], dER[0] - dER[1]) < 1e-14
