'''The data-driven front-end of LDDMM and LDDMM_3D_to_slice (_lddmm_common.resolve_defaults
and _prefit.py).

* the affine pre-fit recovers a large rotation (2D) and a tilted slice plane (3D), keeps the
  initial orientation on ties, is deterministic, follows the 'auto' rules and never stops a fit
* the physical sigmaR and the curvature-scaled steps: formulas and invariances; given values
  are used as is
* the reported settings reproduce a fit, and both backends agree on the default path
* legacy_defaults=True gives the fit of the package before the new defaults bit for bit: against
  the explicit old values, and against the package of commit bec34e4 ("Make the build and test
  suite clean on Linux", the last commit before the new defaults) when STALIGN_LEGACY_SRC
  points to a copy of its source tree (with the C++ extension built for the 'cpp' backend),
  for example

      git archive bec34e4 STalign setup.py pyproject.toml requirements.txt | tar -x -C DIR
      (cd DIR && python setup.py build_ext --inplace)
      STALIGN_LEGACY_SRC=DIR pytest tests/test_defaults.py
'''
import inspect
import math
import os
import subprocess
import sys
import warnings

import numpy as np
import pytest
import torch

import reference as R
from STalign import STalign
from STalign import _lddmm_common as C
from STalign import _prefit as P

HAVE_CPP = STalign.backend_info()['cpp']
needs_cpp = pytest.mark.skipif(not HAVE_CPP, reason='the C++ extension is not built')
BACKENDS = [pytest.param('cpp', marks=needs_cpp), 'torch']
KEYS_2D = ('A', 'v', 'WM', 'WB', 'WA')
KEYS_3D = KEYS_2D + ('Xs',)
LEGACY_SRC = os.environ.get('STALIGN_LEGACY_SRC')


def blob_image(x, centers, widths, amps):
    return R._blobs(np.meshgrid(*x, indexing='ij'), centers, widths, amps)


def rotated_pair(deg=105.0, shift=(40.0, -60.0), seed=5):
    '''Blob source and target in um (30 um pixels) with target(y) = source(R^-1 (y - shift)):
    the source-to-target affine is L = R(deg), T = shift.  Returns (data, L, T, centers).'''
    rng = np.random.default_rng(seed)
    x = [np.arange(48) * 30.0 - 705.0, np.arange(44) * 30.0 - 645.0]
    cen = rng.uniform(-420, 420, size=(10, 2))
    wid = rng.uniform(70, 140, size=10)
    amp = rng.uniform(0.3, 1.0, size=(10, 1))
    L = R.rotation_2d(deg)
    Y = np.stack(np.meshgrid(*x, indexing='ij'), -1) - np.asarray(shift)
    X = Y @ L  # rows: R^-1 (y - shift)
    J = R._blobs([X[..., 0], X[..., 1]], cen, wid, amp)
    return dict(xI=x, I=blob_image(x, cen, wid, amp), xJ=x, J=J), L, np.asarray(shift), cen


def um_3d():
    '''reference.synthetic_3d in um: 50 um atlas voxels and 50 um slice pixels.'''
    d = R.synthetic_3d()
    return dict(xI=[x * 50.0 for x in d['xI']], I=d['I'], xJ=[x * 50.0 for x in d['xJ']], J=d['J'])


def fit2d(data, **kw):
    return STalign.LDDMM(**data, **dict(dict(draw=False, num_threads=2), **kw))


def fit3d(data, **kw):
    return STalign.LDDMM_3D_to_slice(**data, **dict(dict(draw=False, num_threads=2, nt=4), **kw))


def affine(L, T):
    A = np.eye(len(T) + 1)
    A[:-1, :-1], A[:-1, -1] = L, T
    return A


# --------------------------------------------------------------------------- signatures
def test_signatures():
    p2 = inspect.signature(STalign.LDDMM).parameters
    p3 = inspect.signature(STalign.LDDMM_3D_to_slice).parameters
    for k in ('niter', 'epL', 'epT', 'epV', 'sigmaR'):
        assert p2[k].default is None, k
    assert p3['sigmaR'].default is None
    assert (p3['niter'].default, p3['epL'].default, p3['epT'].default, p3['epV'].default) == (5000, 1e-6, 1e1, None)
    for p in (p2, p3):
        assert (p['prefit'].default, p['sigma_disp'].default, p['legacy_defaults'].default) == ('auto', 94.03, False)
        assert all(p[k].kind == inspect.Parameter.KEYWORD_ONLY for k in ('prefit', 'sigma_disp', 'legacy_defaults'))
    assert C.LEGACY_DEFAULTS == {2: dict(niter=5000, epL=2e-8, epT=2e-1, epV=2e3, sigmaR=5e5),
                                 3: dict(niter=5000, epL=1e-6, epT=1e1, epV=1e3, sigmaR=2e6)}


def test_niter_defaults():
    data, _, _, _ = rotated_pair()
    for dim, d, expected in ((2, data, 2000), (3, um_3d(), 5000)):
        for legacy, n in ((False, expected), (True, 5000)):
            problem = C.prepare_problem(dim, **d, niter=None, sigmaR=1.0)
            C.resolve_defaults(problem, prefit=False, legacy_defaults=legacy)
            assert problem.niter == n
    with pytest.raises(ValueError, match='niter'):  # an engine never sees an unset value
        C.run_lddmm(C.prepare_problem(2, **data, niter=None))


# --------------------------------------------------------------------------- sigmaR
def test_intensity_scale():
    J = np.zeros((1, 40, 40))
    J[0, 10:30, 10:30] = 5.0
    assert P.intensity_scale(J) == pytest.approx(5.0, rel=1e-12)  # the tissue intensity
    J[0, 10:20, 10:30] = 1.0  # intensity-weighted mean: (200 * 1 + 200 * 25) / (200 * 1 + 200 * 5)
    assert P.intensity_scale(J) == pytest.approx(5200 / 1200, rel=1e-12)
    two = np.concatenate((J, 3.0 * J))  # rms over channels
    assert P.intensity_scale(two) == pytest.approx(np.sqrt(5.0) * 5200 / 1200, rel=1e-12)
    assert P.intensity_scale(J + 7.0) == pytest.approx(P.intensity_scale(J), rel=1e-12)  # background removed
    assert P.intensity_scale(np.full((2, 5, 5), 3.0)) == 1.0


def test_source_pixel_size():
    xJ = [np.arange(5) * 30.0, np.arange(6) * 20.0]
    assert P.source_pixel_size(xJ, np.eye(2), 2) == pytest.approx(math.sqrt(600.0))
    assert P.source_pixel_size(xJ, 2.0 * R.rotation_2d(33.0), 2) == pytest.approx(math.sqrt(600.0) / 2)
    assert P.source_pixel_size([np.zeros(1)] + xJ, np.eye(3), 3) == pytest.approx(math.sqrt(600.0))
    L3 = np.diag([5.0, 2.0, 2.0]) @ R.rotation_3d(0.0, 0.0)  # the normal's scale does not matter
    assert P.source_pixel_size([np.zeros(1)] + xJ, L3, 3) == pytest.approx(math.sqrt(600.0) / 2)


def test_physical_sigmaR():
    data, _, _, _ = rotated_pair()

    def sigmaR(dim=2, **kw):
        d = data if dim == 2 else um_3d()
        problem = C.prepare_problem(dim, **dict(d, **kw), sigmaR=None)
        return P.physical_sigmaR(problem, 94.03)

    s_J = P.intensity_scale(data['J'])
    base = sigmaR(sigmaM=s_J)
    # 30 um pixels, a = 500, nt = 3, sigmaM = s_J: the earlier default 5e5 (sigma_disp 94.03 is rounded)
    assert base == pytest.approx(94.03 * math.sqrt(12 * math.pi) * 500 * math.sqrt(3), rel=1e-12)
    assert base == pytest.approx(5e5, rel=2e-5)
    assert sigmaR(sigmaM=2 * s_J) == pytest.approx(2 * base, rel=1e-12)
    assert sigmaR(sigmaM=s_J, J=3 * data['J']) == pytest.approx(base / 3, rel=1e-12)
    assert sigmaR(sigmaM=s_J, a=250.0) == pytest.approx(base / 2, rel=1e-12)
    assert sigmaR(sigmaM=s_J, nt=12) == pytest.approx(2 * base, rel=1e-12)
    # the target pixel size in source units: coarser target pixels or a source in smaller units
    assert sigmaR(sigmaM=s_J, xJ=[x * 2 for x in data['xJ']]) == pytest.approx(2 * base, rel=1e-12)
    assert sigmaR(sigmaM=s_J, L=2 * R.rotation_2d(20.0)) == pytest.approx(base / 2, rel=1e-12)
    d3 = um_3d()
    s3 = sigmaR(3, sigmaM=1.0)
    dx3 = 50.0
    assert s3 == pytest.approx(94.03 * math.sqrt(64 * math.pi) * 500 ** 1.5 * math.sqrt(3) * dx3 / 30
                               / P.intensity_scale(d3['J']), rel=1e-12)


def test_given_sigmaR_and_steps_are_used_as_is():
    data, _, _, _ = rotated_pair()
    out = fit2d(data, niter=0, sigmaR=1234.5, epL=1e-9, epT=0.3, epV=7.0, prefit=False)
    assert out['settings'] == dict(niter=0, sigmaR=1234.5, epL=1e-9, epT=0.3, epV=7.0)
    out3 = fit3d(um_3d(), niter=0, sigmaR=4.5e6, prefit=False)
    assert out3['settings'] == dict(niter=0, sigmaR=4.5e6, epL=1e-6, epT=1e1, epV=1e3)


# --------------------------------------------------------------------------- steps
def test_curvature_proxy_matches_its_definition():
    ndimage = pytest.importorskip('scipy.ndimage')
    data, L, T, _ = rotated_pair(deg=17.0)
    I = np.concatenate((data['I'], data['I'] ** 2))
    L, T = 1.1 * L, T + 25.0  # target pixels between source pixels and outside the source
    G = P.curvature_proxy(data['xI'], I, data['xJ'], data['J'], L, T)
    # G = (sum_c var J_c) * mean_k sum_j |grad AI_kj|^2 / var AI_k, AI by linear interpolation
    xI, xJ = data['xI'], data['xJ']
    Y = np.stack(np.meshgrid(*xJ, indexing='ij'), 0).reshape(2, -1)
    X = np.linalg.solve(L, Y - T[:, None])
    idx = [(X[k] - xI[k][0]) / (xI[k][1] - xI[k][0]) for k in range(2)]
    g = []
    for Ik in I:
        AI = ndimage.map_coordinates(Ik, idx, order=1, mode='nearest').reshape(len(xJ[0]), len(xJ[1]))
        gy, gx = np.gradient(AI, xJ[0][1] - xJ[0][0], xJ[1][1] - xJ[1][0])
        g.append(np.sum(gy ** 2 + gx ** 2) / np.var(AI))
    assert G == pytest.approx(np.var(data['J']) * np.mean(g), rel=1e-10)


def test_curvature_steps():
    data, _, _, _ = rotated_pair()

    def steps(resolve=False, **kw):
        problem = C.prepare_problem(2, **dict(data, **kw), epL=None, epT=None, epV=None, sigmaR=5e5)
        if resolve:  # the steps the fit uses (capped by stable_steps)
            C.resolve_defaults(problem, prefit=False)
            return np.array([problem.epL, problem.epT, problem.epV])
        return np.array([P.curvature_steps(problem, C.STEPS_2D)[k] for k in ('epL', 'epT', 'epV')])

    base = steps()
    G = P.curvature_proxy(data['xI'], data['I'], data['xJ'], data['J'], np.eye(2), np.zeros(2))
    np.testing.assert_allclose(base, np.array([2e-8, 2e-1, 2e3]) * 1.1 / G, rtol=1e-14)
    # the image term scales with J^2 / sigmaM^2; its curvature, and so the step, with the inverse
    np.testing.assert_allclose(steps(J=10 * data['J']), base / 100, rtol=1e-12)
    np.testing.assert_allclose(steps(sigmaM=3.0), base * 9, rtol=1e-12)
    np.testing.assert_allclose(steps(I=7 * data['I'] + 2), base, rtol=1e-10)
    # a given step is kept, the others are scaled (and capped)
    problem = C.prepare_problem(2, **data, epL=None, epT=None, epV=55.0, sigmaR=5e5)
    C.resolve_defaults(problem, prefit=False)
    capped = P.stable_steps(problem, dict(epL=base[0], epT=base[1]))
    assert (problem.epL, problem.epT, problem.epV) == (capped['epL'], capped['epT'], 55.0)
    # a constant target has no curvature: the unscaled steps (np.var of a constant 0.3 is not 0)
    for c in (1.0, 0.3):
        for resolve in (False, True):
            np.testing.assert_array_equal(steps(resolve, J=np.full_like(data['J'], c)), [2e-8, 2e-1, 2e3])
    # a constant source channel is skipped
    np.testing.assert_array_equal(steps(I=np.concatenate((data['I'], np.full_like(data['I'], 0.3)))), base)


# --------------------------------------------------------------------------- stable steps
def blob_identity(n=40, dx=30.0, seed=5):
    '''A blob image of n x n pixels of dx um (40 x 40: 1.2 mm across), as source and target.'''
    rng = np.random.default_rng(seed)
    x = [np.arange(n) * dx - (n - 1) * dx / 2] * 2
    cen = rng.uniform(-0.35, 0.35, size=(10, 2)) * n * dx
    I = blob_image(x, cen, rng.uniform(70, 140, size=10), rng.uniform(0.3, 1.0, size=(10, 1)))
    return dict(xI=x, I=I, xJ=x, J=I)


def test_curvature_matches_its_definition():
    '''The quadratic forms of _Curvature (cell moments, velocity grid; affine Hessian) against
    their definition: target pixel j's sample point X_j = A^-1 y_j moves by u, the image term by
    u^T S_j u / 2, S_j = sum_c var J_c / sigmaM^2 * mean_k g_kj g_kj^T / var AI_k with g_kj the
    gradient of the source (bilinear, border padding) at X_j, and landmark i by L u.'''
    d = blob_identity(n=16, dx=60.0)
    I = np.concatenate((d['I'], np.sqrt(d['I'])))
    L, T = 1.1 * R.rotation_2d(17.0), np.array([25.0, -40.0])  # between source pixels, some outside it
    pI = np.array([[-300.0, 200.0], [100.0, 350.0], [250.0, -150.0]])
    problem = C.prepare_problem(2, d['xI'], I, d['xJ'], d['J'], pointsI=pI, pointsJ=pI + 7.0, L=L, T=T, niter=1,
                                epL=1.0, epT=1.0, epV=1.0, sigmaR=1.0, sigmaM=0.7, sigmaP=15.0)
    curv = P._Curvature(problem)
    Y = np.stack(np.meshgrid(*d['xJ'], indexing='ij'), 0).reshape(2, -1)
    X = torch.tensor(np.linalg.solve(L, Y - T[:, None]), requires_grad=True)
    AI = R.interp(d['xI'], torch.tensor(I), X[:, None], padding_mode='border')[:, 0]
    g = [torch.autograd.grad(AI[k].sum(), X, retain_graph=True)[0].numpy().T for k in range(2)]
    var = AI.detach().numpy().var(1)
    S = np.var(d['J']) / 0.7 ** 2 * sum(np.einsum('ni,nj->nij', gk, gk) / vk for gk, vk in zip(g, var)) / 2
    rng = np.random.default_rng(3)
    u = rng.standard_normal(problem.K.shape + (2,))
    xv = problem.xv
    su = R.interp(xv, torch.tensor(u).permute(2, 0, 1), X.detach()[:, None])[:, 0].numpy().T
    sp = R.interp(xv, torch.tensor(u).permute(2, 0, 1), torch.tensor(pI.T)[:, None])[:, 0].numpy().T
    form = np.einsum('ni,nij,nj->', su, S, su) + np.sum((sp @ L.T) ** 2) / 15.0 ** 2
    assert np.sum(u * curv.apply(u)) == pytest.approx(form, rel=1e-10)
    # affine: dTheta = [dL | dT] moves X_j by -L^-1 dTheta (X_j, 1) and landmark i by dTheta (p_i, 1)
    dth = rng.standard_normal((2, 3))
    Z = np.concatenate((X.detach().numpy().T, np.ones((Y.shape[1], 1))), 1)
    du = (Z @ dth.T) @ np.linalg.inv(L).T
    Zp = np.concatenate((pI, np.ones((3, 1))), 1)
    form = np.einsum('ni,nij,nj->', du, S, du) + np.sum((Zp @ dth.T) ** 2) / 15.0 ** 2
    vec = np.concatenate((dth[:, :2].ravel(), dth[:, 2]))  # (L00, L01, L10, L11, T0, T1)
    assert vec @ P.affine_hessian(problem) @ vec == pytest.approx(form, rel=1e-10)


def test_velocity_curvature_is_the_largest_eigenvalue():
    '''velocity_curvature = lambda_max(K H) / nt, against a dense eigenvalue computation; also
    where the two largest eigenvalues are close (one slightly elongated blob: shifts along its
    two axes), on which a single power iteration can settle on the second one.'''
    x = [np.arange(16) * 60.0 - 450.0] * 2
    one = blob_image(x, np.array([[30.0, -20.0]]), np.array([150.0]), np.array([[1.0]]))
    one = np.exp(-((np.meshgrid(*x, indexing='ij')[0] - 30.0) ** 2 * (1 / 1.1 ** 2 - 1)) / (2 * 150.0 ** 2)) * one
    pI = np.array([[-300.0, 200.0], [100.0, 350.0]])
    for d, pts in ((blob_identity(n=16, dx=60.0), dict(pointsI=pI, pointsJ=pI)), (dict(xI=x, I=one, xJ=x, J=one), {})):
        problem = C.prepare_problem(2, **d, **pts, niter=1, epL=1.0, epT=1.0, epV=1.0, sigmaR=1.0)
        curv = P._Curvature(problem)
        n = problem.K.size * 2
        H = np.stack([curv.apply(e.reshape(problem.K.shape + (2,))).ravel() for e in np.eye(n)], 1)
        K = np.stack([np.fft.ifft2(np.fft.fft2(e.reshape(problem.K.shape + (2,)), axes=(0, 1))
                                   * problem.K[..., None], axes=(0, 1)).real.ravel() for e in np.eye(n)], 1)
        lam = np.sort(np.linalg.eigvals(K @ H).real)
        assert P.velocity_curvature(problem) == pytest.approx(lam[-1] / problem.nt, rel=1e-8)
    assert lam[-2] > 0.8 * lam[-1]  # the second case is nearly degenerate


@pytest.mark.parametrize('backend', BACKENDS)
def test_velocity_curvature_predicts_the_stable_step(backend):
    '''Gradient descent is stable below epV = 2 / (lambda_v + DV / sigmaR^2) and diverges above.
    The target is the source shifted by a few um, so the fit has something to do.'''
    d = blob_identity()
    start = dict(T=[3.0, -2.0], prefit=False, niter=300, backend=backend)
    problem = C.prepare_problem(2, **d, T=start['T'], niter=None, epL=None, epT=None, epV=None, sigmaR=None)
    C.resolve_defaults(problem, prefit=False)
    limit = 2.0 / (P.velocity_curvature(problem) + problem.DV / problem.sigmaR ** 2)
    assert problem.epV == pytest.approx(limit / 2.0, rel=1e-12)  # the cap applies here
    assert float(fit2d(d, epV=0.9 * limit, **start)['v'].abs().max()) < 5.0
    assert float(fit2d(d, epV=1.1 * limit, **start)['v'].abs().max()) > 20.0


@pytest.mark.parametrize('backend', BACKENDS)
def test_default_steps_are_stable_on_a_small_image(backend):
    '''A 1.2 mm image fitted to itself, with landmarks and a small sigma_disp: the default steps
    keep the map at the identity, while the uncapped curvature_steps diverge.'''
    d = blob_identity()
    p = np.random.default_rng(1).uniform(-400, 400, size=(6, 2))
    for kw in (dict(), dict(pointsI=p, pointsJ=p), dict(sigma_disp=10.0), dict(pointsI=p, pointsJ=p, sigma_disp=10.0)):
        out = fit2d(d, niter=300, backend=backend, **kw)
        assert float(out['v'].abs().max()) < 1.0, kw
        assert np.abs(out['A'].numpy() - np.eye(3)).max() < 1.0, kw
    problem = C.prepare_problem(2, **d, niter=None, epL=None, epT=None, epV=None, sigmaR=None)
    C.resolve_defaults(problem, prefit=False)
    rule = P.curvature_steps(problem, C.STEPS_2D)
    assert rule['epV'] > 4 * problem.epV
    assert float(fit2d(d, niter=300, backend=backend, epV=rule['epV'])['v'].abs().max()) > 20.0


def test_default_affine_steps_are_stable_far_from_the_origin():
    '''Landmarks 20 mm from the origin: the landmark term's curvature in L grows with |x|^2.'''
    d = blob_identity()
    sh = 20000.0
    d = dict(xI=[x + sh for x in d['xI']], I=d['I'], xJ=[x + sh for x in d['xJ']], J=d['J'])
    p = np.random.default_rng(1).uniform(-400, 400, size=(6, 2)) + sh
    out = fit2d(d, pointsI=p, pointsJ=p, niter=300)
    assert np.abs(out['A'].numpy() - np.eye(3)).max() < 1e-3
    assert float(out['v'].abs().max()) < 1.0
    problem = C.prepare_problem(2, **d, pointsI=p, pointsJ=p, niter=None, epL=None, epT=None, epV=None, sigmaR=None)
    C.resolve_defaults(problem)
    rule = P.curvature_steps(problem, C.STEPS_2D)
    assert problem.epL < rule['epL'] / 4 and problem.epL / rule['epL'] == pytest.approx(problem.epT / rule['epT'])
    with warnings.catch_warnings():  # diverges: the landmark map does not converge
        warnings.simplefilter('ignore', RuntimeWarning)
        out = fit2d(d, pointsI=p, pointsJ=p, niter=300, epL=rule['epL'], epT=rule['epT'])
    assert np.abs(out['A'].numpy() - np.eye(3)).max() > 1.0


def test_stable_steps_keeps_small_steps():
    '''Steps below the caps are returned unchanged (the fit is then the one of curvature_steps);
    the affine cap uses the engines' factor 10 only when the diffeomorphism is on from the start.'''
    d = blob_identity()
    problem = C.prepare_problem(2, **d, niter=1, epL=None, epT=None, epV=None, sigmaR=5e5)
    lam = P.velocity_curvature(problem) + problem.DV / problem.sigmaR ** 2
    H = P.affine_hessian(problem)
    small = dict(epV=0.5 / lam, epL=1e-12, epT=1e-6)
    assert P.stable_steps(problem, small) == small
    big = P.stable_steps(problem, dict(epV=2.0 / lam, epL=1e-3, epT=1e3))
    assert big['epV'] == pytest.approx(1.0 / lam, rel=1e-12)
    D = np.sqrt(np.array([big['epL']] * 4 + [big['epT']] * 2) / 10.0)
    assert np.linalg.eigvalsh(D[:, None] * H * D[None, :])[-1] == pytest.approx(0.5, rel=1e-10)
    problem.diffeo_start = 50
    late = P.stable_steps(problem, dict(epL=1e-3, epT=1e3))
    assert late['epL'] == pytest.approx(big['epL'] / 10, rel=1e-10)


def test_3d_epV_is_capped_for_the_regularizer():
    '''3D: the default epV (1e3) is lowered only where epV DV / sigmaR^2 would exceed 1.'''
    d = um_3d()
    out = fit3d(d, niter=0, prefit=False)
    assert out['settings']['epV'] == 1e3
    for sd, sM in ((1.0, 0.1), (3.0, 0.1)):
        problem = C.prepare_problem(3, **d, nt=4, niter=None, epV=None, sigmaR=None, sigmaM=sM)
        C.resolve_defaults(problem, prefit=False, sigma_disp=sd)
        assert problem.epV == pytest.approx(problem.sigmaR ** 2 / problem.DV, rel=1e-12) and problem.epV < 1e3
        out = fit3d(d, niter=100, prefit=False, sigma_disp=sd, sigmaM=sM)
        assert torch.isfinite(out['v']).all() and float(out['v'].abs().max()) < 10.0
    out = fit3d(d, niter=100, prefit=False, sigma_disp=1.0, sigmaM=0.1, epV=1e3)  # given: used as is
    assert out['settings']['epV'] == 1e3
    assert not (float(out['v'].abs().max()) < 1e3)  # and it diverges (or overflows to nan)


# --------------------------------------------------------------------------- pre-fit
@pytest.fixture(scope='module')
def rotated_fits():
    '''Default fits (niter=0: the pre-fit only) of a target rotated by 105 degrees, with 2 and 1
    threads (torch's own setting is restored).'''
    data, L, T, cen = rotated_pair()
    before = torch.get_num_threads()
    one = fit2d(data, niter=0, num_threads=1)
    assert torch.get_num_threads() == before
    return data, affine(L, T), cen, fit2d(data, niter=0), one


def test_prefit_recovers_a_large_rotation(rotated_fits):
    data, A_true, cen, out, _ = rotated_fits
    info = out['prefit']
    # the starts rotate the target-to-source map, which turns by -105 = 255 degrees
    assert info['ran'] and len(info['starts']) == P.ANGLES_2D and info['starts'][info['best']] in (240.0, 270.0)
    A = out['A'].numpy()
    np.testing.assert_array_equal(A, info['A'])
    np.testing.assert_array_equal(info['A_start'], np.eye(3))
    P_true = cen @ A_true[:2, :2].T + A_true[:2, 2]
    P_fit = cen @ A[:2, :2].T + A[:2, 2]
    assert np.abs(P_fit - P_true).max() < 3.0  # um (30 um pixels); up to 950 um at the start
    # starting from another affine: the search composes with it
    start = affine(R.rotation_2d(-150.0), [100.0, 0.0])
    other = fit2d(data, niter=0, A=start)
    np.testing.assert_array_equal(other['prefit']['A_start'], start)
    assert np.abs(cen @ other['A'][:2, :2].numpy().T + other['A'][:2, 2].numpy() - P_true).max() < 3.0


def test_prefit_is_deterministic(rotated_fits):
    data, _, _, out, out1 = rotated_fits
    again = fit2d(data, niter=0)
    assert torch.equal(again['A'], out['A'])
    np.testing.assert_array_equal(again['prefit']['energy'], out['prefit']['energy'])
    # other thread counts round differently, but give the same result
    np.testing.assert_allclose(out1['A'].numpy(), out['A'].numpy(), rtol=0, atol=1e-6)


def test_prefit_keeps_the_orientation_on_ties():
    '''A rotationally symmetric image: the unrotated start ties with the best one (within 1%)
    without being the best, and wins.'''
    x = [np.arange(40) * 30.0 - 585.0, np.arange(40) * 30.0 - 585.0]
    I = blob_image(x, np.zeros((1, 2)), [200.0], np.ones((1, 1)))
    J = I + 0.02 * np.random.default_rng(0).standard_normal(I.shape)
    out = fit2d(dict(xI=x, I=I, xJ=x, J=J), niter=0)
    E = out['prefit']['energy']
    assert np.argmin(E) != 0 and E[0] <= E.min() * 1.01 and out['prefit']['best'] == 0
    L = out['A'][:2, :2].numpy()
    assert abs(math.degrees(math.atan2(L[1, 0], L[0, 0]))) < 1.0


def test_prefit_tilted_slice_3d():
    d = um_3d()
    out = fit3d(d, niter=0)
    info = out['prefit']
    assert info['ran'] and len(info['starts']) == 27 and info['starts'][13] == (0.0, 0.0, 0.0)
    # the slice shows the atlas plane z = 150 + 0.2 y - 0.1 x (um)
    Y, X = [g.ravel() for g in np.meshgrid(*d['xJ'], indexing='ij')]
    Ai = np.linalg.inv(out['A'].numpy())
    z = Ai[0, 0] * 0 + Ai[0, 1] * Y + Ai[0, 2] * X + Ai[0, 3]
    tissue = d['J'][0].ravel() > np.percentile(d['J'], 50)
    err = np.abs(z - (150.0 + 0.2 * Y - 0.1 * X))[tissue]
    assert err.mean() < 30.0  # um; 153 um at the start (the plane z = 0)


def test_prefit_3d_keeps_the_initial_plane_on_ties():
    '''An atlas that is constant along its first axis: every start sees the same slice up to
    a stretch, so all 27 starts tie (within 1%), and the untilted, unshifted start wins.'''
    d = um_3d()
    I = np.repeat(d['I'][:, 10:11], d['I'].shape[1], axis=1)
    J = I[:1, 0] + 0.02 * np.random.default_rng(0).standard_normal(I.shape[2:])
    out = fit3d(dict(xI=d['xI'], I=I, xJ=d['xI'][1:], J=J), niter=0)
    E = out['prefit']['energy']
    assert E.max() <= 1.01 * E.min() and out['prefit']['best'] == 13
    assert out['prefit']['starts'][13] == (0.0, 0.0, 0.0)


def test_prefit_3d_tie_rule(monkeypatch):
    '''The rule itself, on given energies (no optimization: each start keeps its plane):
    among the starts within 1% of the lowest energy, the one with the smallest
    |tilt 1| + |tilt 2| + |offset| / 100 (degrees, um) wins.'''
    d = um_3d()
    problem = C.prepare_problem(3, **d, niter=0, sigmaR=1.0)
    starts = [(a, b, o) for a in P.TILTS_3D for b in P.TILTS_3D for o in P.OFFSETS_3D]
    index = {s: k for k, s in enumerate(starts)}
    energies = {}
    monkeypatch.setattr(P, '_adam', lambda params, energy, iters, lr: torch.as_tensor(energies['E']))
    # the source point of the target's tissue centroid (as the search computes it): the plane
    # of the start with offset o passes through the initial plane's point plus o along its normal
    xJc, Jc = P.pool_image(d['xJ'], d['J'][:, 0] if d['J'].ndim == 4 else d['J'], P.search_pool(d['xJ']))
    cJ, _ = P.weighted_centroid(xJc, P.tissue_weights(Jc))
    cases = [({(-6.0, -6.0, -300.0): 1.0, (0.0, 0.0, 0.0): 1.009}, (0.0, 0.0, 0.0)),
             ({(-6.0, -6.0, -300.0): 1.0, (0.0, 0.0, 0.0): 1.02, (0.0, -6.0, 0.0): 1.005, (0.0, 0.0, 300.0): 1.008},
              (0.0, 0.0, 300.0)),
             ({(6.0, -6.0, 0.0): 1.0, (0.0, 0.0, 0.0): 1.02, (0.0, 6.0, 0.0): 1.011}, (6.0, -6.0, 0.0))]
    for given, winner in cases:
        E = np.full(len(starts), 2.0)
        for s, e in given.items():
            E[index[s]] = e
        energies['E'] = E
        L, T, info = P.affine_prefit(problem)
        assert info['best'] == index[winner], (given, starts[info['best']])
        x0 = np.linalg.solve(info['A'], np.r_[0.0, cJ, 1.0])[:3]  # source point of the centroid
        np.testing.assert_allclose(x0, [winner[2], cJ[0], cJ[1]], rtol=0, atol=1e-9)


def test_prefit_rules():
    data, _, _, _ = rotated_pair()
    points = dict(pointsI=np.array([[0.0, 0.0], [300.0, 100.0], [-200.0, 250.0]]))
    points['pointsJ'] = points['pointsI'] + 10.0

    def info(**kw):
        problem = C.prepare_problem(2, **{k: v for k, v in dict(data, **kw).items() if k not in ('prefit', 'legacy')},
                                    sigmaR=1.0)
        return C.resolve_defaults(problem, prefit=kw.get('prefit', 'auto'),
                                  legacy_defaults=kw.get('legacy', False))['prefit']

    def skipped(reason, A=np.eye(3)):  # the 'prefit' entry when it did not run: A is the start
        return {'ran': False, 'reason': reason, 'A': A}

    np.testing.assert_equal(info(**points), skipped('landmarks'))
    np.testing.assert_equal(info(prefit=False, L=2 * np.eye(2), T=[3.0, 4.0]),
                            skipped('disabled', affine(2 * np.eye(2), [3.0, 4.0])))
    np.testing.assert_equal(info(legacy=True), skipped('legacy defaults'))
    with_points = info(prefit=True, **points)  # one start (the initial orientation), landmarks in the energy
    assert with_points['ran'] and with_points['starts'] == [0.0]
    for bad in ('yes', None, 1):
        with pytest.raises(ValueError, match='prefit'):
            info(prefit=bad)
    with pytest.raises(ValueError, match='legacy_defaults'):
        info(legacy='no')
    # a nonzero initial velocity (a warm start) turns 'auto' off, through the public functions
    # too; a zero one (e.g. to choose the velocity grid) does not
    xv = [np.linspace(-900, 900, 4)] * 2
    v = np.zeros((3, 4, 4, 2))
    v[1, 2, 2, 0] = 1.0
    np.testing.assert_equal(fit2d(data, niter=0, v=v, xv=xv)['prefit'], skipped('initial velocity'))
    assert info(v=0 * v, xv=xv, prefit='auto')['ran']


def test_prefit_skips_a_target_without_tissue():
    '''A constant target has no tissue to align to: the pre-fit is skipped with one warning
    (no numpy RuntimeWarning), the given steps are kept and the fit runs.'''
    data, _, _, _ = rotated_pair()
    d3 = um_3d()
    fits = ((fit2d, dict(data, J=np.full_like(data['J'], 0.3))), (fit3d, dict(d3, J=np.full_like(d3['J'], 0.3))))
    for fit, d in fits:
        for prefit in ('auto', True):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                out = fit(d, niter=3, prefit=prefit)
            assert [(w.category, str(w.message)) for w in caught] == [
                (UserWarning, 'the affine pre-fit was skipped: the target image has no tissue (it is constant at the '
                              'resolution of the pre-fit); starting from the given affine')]
            np.testing.assert_equal(out['prefit'], {'ran': False, 'reason': 'no target tissue',
                                                    'A': np.eye(len(d['xI']) + 1)})
            assert all(torch.isfinite(out[k]).all() for k in ('A', 'v', 'WM'))
        if fit is fit2d:
            assert {k: out['settings'][k] for k in C.STEPS_2D} == C.STEPS_2D


def test_prefit_failure_warns_and_the_fit_goes_on(monkeypatch):
    data, _, _, _ = rotated_pair()
    ref = fit2d(data, niter=5, prefit=False)

    def broken(problem):
        raise RuntimeError('no luck')

    monkeypatch.setattr(P, 'affine_prefit', broken)
    with pytest.warns(UserWarning, match='pre-fit failed .*no luck'):
        out = fit2d(data, niter=5)
    np.testing.assert_equal(out['prefit'], {'ran': False, 'reason': 'failed: RuntimeError: no luck', 'A': np.eye(3)})
    assert all(torch.equal(out[k], ref[k]) for k in KEYS_2D)


# --------------------------------------------------------------------------- default path
@pytest.fixture(scope='module')
def default_fits():
    '''Default fits (pre-fit, physical sigmaR, curvature steps) with both backends.'''
    data, _, _, _ = rotated_pair(deg=40.0)
    backends = ['cpp', 'torch'] if HAVE_CPP else ['torch']
    return data, {b: fit2d(data, niter=60, backend=b) for b in backends}


def test_settings_reproduce_the_fit(default_fits):
    ''''settings' and the start prefit['A'] repeat a fit, whether the pre-fit ran or not.'''
    data, fits = default_fits
    out = next(iter(fits.values()))
    backend = next(iter(fits))
    settings = out['settings']
    assert out['prefit']['ran']
    assert settings['niter'] == 60 and settings['sigmaR'] > 0 and min(settings[k] for k in ('epL', 'epT', 'epV')) > 0
    again = fit2d(data, A=out['prefit']['A'], prefit=False, backend=backend, **settings)
    assert all(torch.equal(again[k], out[k]) for k in KEYS_2D)
    assert float(out['v'].abs().max()) > 0.1  # the deformation is on
    # without the pre-fit, from a given L and T
    start = dict(L=R.rotation_2d(35.0), T=[30.0, -50.0])
    out = fit2d(data, niter=60, prefit=False, backend=backend, **start)
    assert not out['prefit']['ran']
    np.testing.assert_array_equal(out['prefit']['A'], affine(start['L'], start['T']))
    again = fit2d(data, A=out['prefit']['A'], prefit=False, backend=backend, **out['settings'])
    assert all(torch.equal(again[k], out[k]) for k in KEYS_2D)


def test_backends_agree_on_the_default_path(default_fits):
    _, fits = default_fits
    if len(fits) < 2:
        pytest.skip('the C++ extension is not built')
    cpp, ref = fits['cpp'], fits['torch']
    np.testing.assert_array_equal(cpp['prefit']['A'], ref['prefit']['A'])
    assert cpp['settings'] == ref['settings']
    for k in KEYS_2D:
        assert float((cpp[k] - ref[k]).abs().max() / ref[k].abs().max()) < 1e-9, k


# --------------------------------------------------------------------------- legacy defaults
OLD_2D = dict(epL=2e-8, epT=2e-1, epV=2e3, sigmaR=5e5)


def landmarks(deg=8.0):
    '''Landmarks of rotated_pair(deg): pointsJ = R(deg) pointsI + shift.'''
    pI = np.array([[0.0, 0.0], [300.0, 100.0], [-200.0, 250.0], [150.0, -300.0]])
    return pI, pI @ R.rotation_2d(deg).T + np.array([40.0, -60.0])


@pytest.mark.parametrize('backend', BACKENDS)
def test_legacy_defaults_are_the_old_values(backend):
    data, _, _, _ = rotated_pair(deg=8.0)
    out = fit2d(data, niter=60, backend=backend, legacy_defaults=True)
    ref = fit2d(data, niter=60, backend=backend, prefit=False, **OLD_2D)
    assert all(torch.equal(out[k], ref[k]) for k in KEYS_2D)
    assert out['settings'] == dict(OLD_2D, niter=60)
    # with landmarks also the forward Euler landmark map (not a public argument)
    pI, pJ = landmarks()
    out = fit2d(data, pointsI=pI, pointsJ=pJ, niter=60, backend=backend, legacy_defaults=True)
    problem = C.prepare_problem(2, **data, pointsI=pI, pointsJ=pJ, niter=60, landmark_map='euler', **OLD_2D)
    ref = C.run_lddmm(problem, draw=False, num_threads=2, backend=backend)
    assert all(torch.equal(out[k], ref[k]) for k in KEYS_2D)
    exact = fit2d(data, pointsI=pI, pointsJ=pJ, niter=60, backend=backend, **OLD_2D)
    assert not torch.equal(exact['A'], out['A'])  # the default is the exact map
    d3 = um_3d()
    out = fit3d(d3, niter=30, backend=backend, legacy_defaults=True)
    ref = fit3d(d3, niter=30, backend=backend, prefit=False, sigmaR=2e6)
    assert all(torch.equal(out[k], ref[k]) for k in KEYS_3D)


LEGACY_SCRIPT = '''
import sys
sys.path.insert(0, sys.argv[1])
import numpy as np
import torch
torch.set_num_threads(2)
from STalign import STalign
assert STalign.__file__.startswith(sys.argv[1]), STalign.__file__
d = dict(np.load(sys.argv[2]))
backend = sys.argv[4]
x2 = dict(xI=[d['xI0'], d['xI1']], I=d['I'], xJ=[d['xJ0'], d['xJ1']], J=d['J'])
x3 = dict(xI=[d['zI0'], d['zI1'], d['zI2']], I=d['I3'], xJ=[d['zJ0'], d['zJ1']], J=d['J3'])
fits = {'2d': STalign.LDDMM(**x2, niter=60, draw=False, backend=backend),
        '2d_points': STalign.LDDMM(**x2, pointsI=d['pI'], pointsJ=d['pJ'], niter=60, draw=False, backend=backend),
        '3d': STalign.LDDMM_3D_to_slice(**x3, nt=4, niter=30, draw=False, backend=backend)}
np.savez(sys.argv[3], **{f'{n}_{k}': out[k].numpy() for n, out in fits.items() for k in out if k != 'xv'})
'''


@pytest.mark.skipif(not LEGACY_SRC, reason='set STALIGN_LEGACY_SRC to a source tree of commit bec34e4')
@pytest.mark.parametrize('backend', BACKENDS)
def test_legacy_defaults_match_the_previous_version(tmp_path, backend):
    data, _, _, _ = rotated_pair(deg=8.0)
    d3 = um_3d()
    pI, pJ = landmarks()
    inputs = dict(xI0=data['xI'][0], xI1=data['xI'][1], I=data['I'], xJ0=data['xJ'][0], xJ1=data['xJ'][1],
                  J=data['J'], pI=pI, pJ=pJ, zI0=d3['xI'][0], zI1=d3['xI'][1], zI2=d3['xI'][2], I3=d3['I'],
                  zJ0=d3['xJ'][0], zJ1=d3['xJ'][1], J3=d3['J'])
    np.savez(tmp_path / 'inputs.npz', **inputs)
    env = {k: v for k, v in os.environ.items() if k != 'PYTHONPATH'}
    result = subprocess.run([sys.executable, '-c', LEGACY_SCRIPT, os.path.abspath(LEGACY_SRC),
                             str(tmp_path / 'inputs.npz'), str(tmp_path / 'old.npz'), backend],
                            cwd=tmp_path, env=env, capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stderr
    old = np.load(tmp_path / 'old.npz')
    new = {'2d': fit2d(data, niter=60, backend=backend, legacy_defaults=True),
           '2d_points': fit2d(data, pointsI=pI, pointsJ=pJ, niter=60, backend=backend, legacy_defaults=True),
           '3d': fit3d(d3, niter=30, backend=backend, legacy_defaults=True)}
    for name, out in new.items():
        for k in (KEYS_3D if name == '3d' else KEYS_2D):
            np.testing.assert_array_equal(out[k].numpy(), old[f'{name}_{k}'], err_msg=f'{name} {k}')
