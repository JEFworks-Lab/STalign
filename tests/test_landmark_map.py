'''The exact landmark map of LDDMM (Problem.landmark_map == 'exact', the default), on both
engines.

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
