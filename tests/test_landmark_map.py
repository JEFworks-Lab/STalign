'''The exact landmark map of LDDMM (Problem.landmark_map == 'exact', the default), on both
engines.

* its gradient with respect to L, T and v matches central finite differences of the landmark
  energy EP (the Euler map's gradient does not)
* an exactly singular Newton matrix M = I - Dv/nt falls back to the forward Euler step, with
  its gradient, in both engines and in the oracle, and is counted in the warning

The matching weights WM are set to 0 so that the image term contributes nothing: the
energies and gradients are those of the landmark term alone.
'''
import numpy as np
import pytest
import torch

import reference as R
from STalign import STalign
from STalign import _lddmm_common as C
from STalign import _lddmm_torch as TE

HAVE_CPP = STalign.backend_info()['cpp']
BACKENDS = [pytest.param('cpp', marks=pytest.mark.skipif(not HAVE_CPP, reason='the C++ extension is not built')),
            'torch']
PARAMS = dict(a=8.0, sigmaM=0.5, sigmaR=50.0, sigmaP=2.0)


def engine(backend, problem):
    if backend == 'cpp':
        from STalign import _lddmm_cpp
        return _lddmm_cpp.CppEngine(problem, 2)
    return TE.TorchEngine(problem, 2)


def numpy(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def landmark_problem(d, state, params=PARAMS, **kw):
    '''A prepared 2D problem at the state (L, T, v, xv) with WM = 0 (landmark term only).'''
    problem = C.prepare_problem(2, d['xI'], d['I'], d['xJ'], d['J'], pointsI=d['pointsI'], pointsJ=d['pointsJ'],
                                niter=1, **state, **params, **kw)
    problem.WM = np.zeros_like(problem.WM)
    return problem


# --------------------------------------------------------------------------- finite differences
@pytest.fixture(scope='module')
def fd_state():
    '''synthetic_2d with landmarks; a velocity field large enough that the exact and the Euler
    maps differ, on a grid that covers the source image only.'''
    d = R.synthetic_2d(CI=1, CJ=1)
    xv = [x.numpy() for x in R.velocity_grid([torch.tensor(x) for x in d['xI']], 1.0, 8.0)]
    v = R.smooth_random_v(xv, 3, 4.0, seed=1).numpy()
    dv = R.smooth_random_v(xv, 3, 1.0, seed=7).numpy()  # direction of the v derivative
    state = dict(L=1.05 * R.rotation_2d(10.0), T=np.array([2.0, -1.5]), v=v, xv=xv)
    return d, state, dv


@pytest.mark.parametrize('backend', BACKENDS)
def test_exact_landmark_gradient_matches_finite_differences(fd_state, backend):
    d, state, dv = fd_state

    def EP(**change):
        return float(engine(backend, landmark_problem(d, dict(state, **change))).gradient()['EP'])

    g = engine(backend, landmark_problem(d, state)).gradient()
    assert float(g['EM']) == 0.0
    fd = {}
    hL, hT, hv = 1e-6, 1e-5, 1e-5
    fd['Lgrad'] = np.array([[(EP(L=state['L'] + hL * E) - EP(L=state['L'] - hL * E)) / (2 * hL)
                             for E in np.eye(4).reshape(2, 2, 2, 2)[i]] for i in range(2)])
    fd['Tgrad'] = np.array([(EP(T=state['T'] + hT * e) - EP(T=state['T'] - hT * e)) / (2 * hT) for e in np.eye(2)])
    fd['v'] = (EP(v=state['v'] + hv * dv) - EP(v=state['v'] - hv * dv)) / (2 * hv)
    ours = {k: numpy(g[k]) for k in ('Lgrad', 'Tgrad')}
    ours['v'] = float(np.sum(numpy(g['vgrad']) * dv))
    for k in ('Lgrad', 'Tgrad', 'v'):
        scale = np.abs(ours[k]).max()
        assert scale > 0.1 and np.abs(ours[k] - fd[k]).max() < 1e-6 * scale, (k, ours[k], fd[k])
    # the gradient of the Euler map (a different map) does not pass this test
    euler = engine(backend, landmark_problem(d, state, landmark_map='euler')).gradient()
    assert abs(float(np.sum(numpy(euler['vgrad']) * dv)) - fd['v']) > 1e-3 * abs(fd['v'])


# --------------------------------------------------------------------------- singular Newton matrix
def singular_state(nt=2):
    '''A velocity field whose row component is nt * row on rows -2..2 (grid spacing 2) and 0
    on the other rows near it: at the landmark (0, 4), on a grid node, Dv/nt = diag(1, 0)
    exactly, so M = I - Dv/nt = diag(0, 1) is singular and its fixed point is the landmark
    itself (in both time steps).  The other landmarks see a smooth random velocity.'''
    d = R.synthetic_2d(CI=1, CJ=1)
    xv = [np.arange(17) * 2.0 - 16.0, np.arange(17) * 2.0 - 16.0]
    v = 0.4 * R.smooth_random_v([torch.tensor(x) for x in xv], nt, 1.0, seed=3).numpy()
    v[:, 5:12] = 0.0
    v[:, 7:10, :, 0] = nt * xv[0][7:10, None]
    d['pointsI'] = np.array([[0.0, 4.0], [-12.0, 6.0], [10.0, -10.0]])
    d['pointsJ'] = d['pointsI'] + np.array([1.0, -1.5])
    return d, dict(L=1.02 * R.rotation_2d(3.0), T=np.array([0.5, -0.3]), v=v, xv=xv, nt=nt)


def test_singular_newton_matrix_falls_back_to_the_euler_step():
    d, state = singular_state()
    params = dict(PARAMS, a=4.0)
    ref = R.gradients(d['xI'], d['I'], d['xJ'], d['J'], state['L'], state['T'], state['v'],
                      [torch.tensor(x) for x in state['xv']], np.zeros(d['J'].shape[1:]), pointsI=d['pointsI'],
                      pointsJ=d['pointsJ'], **params)
    A = np.eye(3)
    A[:2, :2], A[:2, 2] = state['L'], state['T']
    np.testing.assert_allclose(numpy(ref['pointsIt'])[0], A[:2, :2] @ d["pointsI"][0] + A[:2, 2], rtol=1e-14)
    backends = ['cpp', 'torch'] if HAVE_CPP else ['torch']
    for backend in backends:
        eng = engine(backend, landmark_problem(d, state, params))
        g = eng.gradient()
        for k in ('Lgrad', 'Tgrad', 'vgrad', 'direction'):
            assert np.abs(numpy(g[k]) - numpy(ref[k])).max() < 1e-9 * np.abs(numpy(ref[k])).max(), (backend, k)
        assert float(g['EP']) == pytest.approx(float(ref['EP']), rel=1e-13)
        eng.run(1)  # one more evaluation: the landmark (0, 4) at both time steps
        snap = eng.snapshot()
        assert snap['landmark_unconverged'] == 4, backend
        np.testing.assert_allclose(numpy(snap['pointsIt']), numpy(ref['pointsIt']), rtol=1e-13)
        problem = landmark_problem(d, state, params)
        problem.niter = 3
        with pytest.warns(RuntimeWarning, match='singular'):
            out = C.run_lddmm(problem, draw=False, num_threads=2, backend=backend)
        assert torch.isfinite(out['A']).all() and torch.isfinite(out['v']).all()
