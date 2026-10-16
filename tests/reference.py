"""Reference ("oracle") implementation of the STalign LDDMM iteration for the test suite.

This is a deliberately plain transcription of the original ``STalign.LDDMM`` and
``STalign.LDDMM_3D_to_slice`` loops (JEFworks/STalign), using torch autograd for every
gradient.  It is slow and it is not used by the package.  Both backends (pure torch and
C++) are tested against it.

The original formulas are kept operation by operation, so where no fix applies the oracle
reproduces the original bitwise.  The approved fixes are applied:

F1  The 3D regularizer transforms over all three spatial axes::

        ER = sum_{t,c} |fftn(v, dims (1,2,3))|^2 * LL * DV / 2 / (n0 n1 n2) / sigmaR^2

    The original transformed only dims (1,2) but multiplied by the 3D symbol LL.  With the
    fix the K-smoothed gradient of ER is exactly DV / sigmaR^2 * v, as it is in 2D.
    ``legacy_er3d=True`` restores the original formula; it exists only so the oracle can be
    validated against the original code.
F2  The returned A and v are the last *evaluated* pair: ``A`` is the affine and ``v`` the
    velocity used in the last forward pass.  The original returned that A but the v after
    the final update.  WM/WB/WA are the weights after the last iteration (as before; the
    E-step uses the last evaluation), and the 3D ``Xs`` comes from the last forward pass
    (as before).  ``niter=0`` returns the initial state instead of crashing.  The extra keys
    ``A_next`` / ``v_next`` hold the parameters after the final update (``v_next`` is what
    the original returned as ``v``), and ``Esave`` the energy history.
F3  User-supplied muA / muB are coerced to one value per target channel (``coerce_mu``).

and the optional affine step control (``affine_step_control``; on by default in 3D, as in the
package, and off in 2D).  Each affine parameter (L row-major, then T) has a step multiplier m,
initially 1, with the previous gradient g_prev initially 0.  Every iteration, after the
backward pass and before the affine update::

    g = concat(L.grad.ravel(), T.grad);  s = g * g_prev
    m = where(s < 0, max(SHRINK * m, FLOOR), where(s > 0, min(GROW * m, CAP), m));  g_prev = g
    L -= (epL / fac) * m[:d*d].reshape(d, d) * L.grad;  T -= (epT / fac) * m[d*d:] * T.grad

where fac is the original's 1 + 9 * (it >= diffeo_start).  The v update is unchanged.  Off,
the original update lines run unchanged.  The result's ``step_multipliers`` holds the final m.
The default 3D sigmaR (2e6) is the package's with legacy_defaults=True: the oracle has no
affine pre-fit and no data-driven settings, so comparisons pass prefit=False and explicit
settings to the package.

Version 2.0 adds two options of the 2D fit:

landmark_map='exact' (default)  The landmarks move with the exact inverse of the backward map
    instead of the original's forward Euler steps ('euler').  Each time step solves
    p_{t+1} = p_t + v_t(p_{t+1}) / nt by fixed-point iteration (no gradient); then one Newton
    step of that implicit equation, p_{t+1} = z + M^-1 (p_t + v_t(z)/nt - z) with z the fixed
    point and M = I - Dv_t(z)/nt held constant (Dv_t by autograd), gives the implicit-function
    gradient through autograd.  Where M is exactly singular (det M = 0 or NaN) the step is the
    forward Euler step p_t + v_t(p_t) / nt.
mixture='em' (or a dict of options, default None)  The mixture EM: WM starts at 1, WA = WB = 0,
    and the original's weight updates are replaced by the E/M-steps documented in
    ``STalign._lddmm_common.Problem`` (log-space E-step, matching variance estimated from the
    residual, class variances tied to it by sigmaA/sigmaM and sigmaB/sigmaM).
"""
import warnings

import numpy as np
import torch
from torch.nn.functional import grid_sample

DTYPE = torch.float64

# affine step control constants (the package's defaults, _lddmm_common.AFFINE_STEP_*)
SHRINK = 0.5
GROW = 1.05
FLOOR = 1e-3
CAP = 1.0
SIGMAR_3D = 2e6  # sigmaR of LDDMM_3D_to_slice with legacy_defaults=True

# exact landmark map: a fixed-point solve stops when no point moves by more than this fraction
# of a velocity grid cell (or a few ulps of the coordinates), after at most LANDMARK_MAX_ITER sweeps
LANDMARK_TOL = 1e-10
LANDMARK_MAX_ITER = 100

# mixture EM (mixture='em'): default options, and the floor of the matching variance
MIXTURE_EM = dict(start=50, every=5, classes='MB', pi_min=0.5, pi_init=(0.9, 0.05, 0.05),
                  fix_muA=False, fix_muB=True)
VARIANCE_FLOOR = 1e-12


# --------------------------------------------------------------------------- helpers
def _t(x, dtype=DTYPE):
    """Copy of x as a CPU tensor of the given dtype (numpy, torch or nested lists)."""
    if isinstance(x, torch.Tensor):
        return x.detach().to('cpu', dtype).clone()
    return torch.tensor(np.asarray(x), dtype=dtype)


def interp(x, I, phii, **kwargs):
    """Copy of the original ``interp``: sample the 2D image I (channels first) at phii (2, ...)."""
    I = torch.as_tensor(I)
    phii = torch.clone(torch.as_tensor(phii))
    for i in range(2):
        phii[i] -= x[i][0]
        phii[i] /= x[i][-1] - x[i][0]
    phii *= 2.0
    phii -= 1.0
    out = grid_sample(I[None], phii.flip(0).permute((1, 2, 0))[None], align_corners=True, **kwargs)
    return out[0]


def interp3D(x, I, phii, **kwargs):
    """Copy of the original ``interp3D``: sample the 3D image I (channels first) at phii (3, ...)."""
    I = torch.as_tensor(I)
    phii = torch.clone(torch.as_tensor(phii))
    for i in range(3):
        phii[i] -= x[i][0]
        phii[i] /= x[i][-1] - x[i][0]
    phii *= 2.0
    phii -= 1.0
    out = grid_sample(I[None], phii.flip(0).permute((1, 2, 3, 0))[None], align_corners=True, **kwargs)
    return out[0]


def to_A(L, T):
    """Affine matrix [[L, T], [0, 1]] (2D or 3D), as the original ``to_A`` / ``to_A_3D``."""
    d = L.shape[0]
    O = torch.zeros(d + 1, device=L.device, dtype=L.dtype)
    O[-1] = 1.0
    return torch.cat((torch.cat((L, T[:, None]), 1), O[None]))


def velocity_grid(xI, expand, a, dtype=DTYPE):
    """Velocity grid of the original: the source box grown by ``expand``, spacing a/2."""
    minv = torch.as_tensor([x[0] for x in xI], dtype=dtype)
    maxv = torch.as_tensor([x[-1] for x in xI], dtype=dtype)
    minv, maxv = (minv + maxv) * 0.5 + 0.5 * torch.tensor([-1.0, 1.0], dtype=dtype)[..., None] * (maxv - minv) * expand
    return [torch.arange(m, M, a * 0.5, dtype=dtype) for m, M in zip(minv, maxv)]


def kernels(xv, a, p):
    """Fourier symbol LL of (1 - a^2 Laplacian)^(2p) on the velocity grid, K = 1/LL, voxel volume DV."""
    dv = torch.as_tensor([x[1] - x[0] for x in xv], dtype=xv[0].dtype)
    fv = [torch.arange(len(x), dtype=xv[0].dtype) / len(x) / d for x, d in zip(xv, dv)]
    FV = torch.stack(torch.meshgrid(fv, indexing='ij'), -1)
    LL = (1.0 + 2.0 * a ** 2 * torch.sum((1.0 - torch.cos(2.0 * np.pi * FV * dv)) / dv ** 2, -1)) ** (p * 2.0)
    return LL, 1.0 / LL, torch.prod(dv)


def coerce_mu(mu, C, name='mu'):
    """F3: a user-supplied class mean as a float64 tensor with one value per target channel.

    Size C is used as is and size 1 is broadcast.  Another size with all-equal values warns
    and uses that value (the original broadcast it, which multiplied the class's squared
    distance by the size).  Anything else is a ValueError.
    """
    if mu is None:
        return None
    mu = _t(mu).reshape(-1)
    if mu.numel() == C:
        return mu
    if mu.numel() == 1:
        return mu.expand(C).clone()
    if mu.numel() > 1 and bool(torch.all(mu == mu[0])):
        warnings.warn(f'{name} has {mu.numel()} entries but the target has {C} channel(s); '
                      f'using {name}[0] for every channel', stacklevel=3)
        return mu[:1].expand(C).clone()
    raise ValueError(f'{name} must have 1 or {C} entries (one per target channel), got {mu.numel()}')


def regularizer(v, LL, DV, sigmaR, legacy_er3d=False):
    """ER = sum over time steps and components of |L v|^2 * DV / 2 / sigmaR^2 (no dt; F1 in 3D)."""
    dim = v.ndim - 2
    dims = (1, 2) if (dim == 2 or legacy_er3d) else (1, 2, 3)
    ER = torch.sum(torch.sum(torch.abs(torch.fft.fftn(v, dim=dims)) ** 2, dim=(0, -1)) * LL) * DV / 2.0
    for d in dims:
        ER = ER / v.shape[d]
    return ER / sigmaR ** 2


def smooth(g, K):
    """K * g over the spatial dims of g (nt, *nv, d): the update direction applied to v."""
    dims = tuple(range(1, g.ndim - 1))
    return torch.fft.ifftn(torch.fft.fftn(g, dim=dims) * K[..., None], dim=dims).real


# --------------------------------------------------------------------------- landmarks
def points_velocity(xv, vt, P):
    """The velocity vt (nv0, nv1, 2) at the points P (N, 2) (bilinear, zero outside): (N, 2)."""
    return interp(xv, vt.permute(2, 0, 1), P.T[..., None])[..., 0].T


def landmark_tol(xv):
    """Stopping tolerance of the exact landmark map: LANDMARK_TOL of the smallest velocity grid
    cell, but at least a few ulps of the grid's coordinates."""
    h = min(float(x[-1] - x[0]) / (len(x) - 1) for x in xv)
    R = max(max(abs(float(x[0])), abs(float(x[-1]))) for x in xv)
    return max(LANDMARK_TOL * h, 64.0 * np.finfo(np.float64).eps * R)


def implicit_step(xv, vt, p, nt):
    """One step of the exact landmark map: the solution of z = p + vt(z) / nt, differentiable
    in p and vt by the implicit function theorem (see the module docstring)."""
    tol = landmark_tol(xv)
    f = vt.detach()
    with torch.no_grad():
        x = p.detach()
        z = x + points_velocity(xv, f, x) / nt
        for _ in range(LANDMARK_MAX_ITER):
            zn = x + points_velocity(xv, f, z) / nt
            moved = float((zn - z).abs().max())
            z = zn
            if moved <= tol:
                break
    with torch.enable_grad():
        zz = z.clone().requires_grad_(True)
        Dv = torch.stack([torch.autograd.grad(points_velocity(xv, f, zz)[:, k].sum(), zz)[0] for k in range(2)], 1)
    M = torch.eye(2, dtype=p.dtype) - Dv / nt  # (N, 2, 2), constant
    # an exactly singular M (det 0 or NaN): the Euler step instead, as the engines do
    singular = ~((M[:, 0, 0] * M[:, 1, 1] - M[:, 0, 1] * M[:, 1, 0]).abs() > 0)
    M = torch.where(singular[:, None, None], torch.eye(2, dtype=p.dtype), M)
    r = p + points_velocity(xv, vt, z) / nt - z
    newton = z + torch.linalg.solve(M, r[..., None])[..., 0]
    if not bool(singular.any()):
        return newton
    return torch.where(singular[:, None], p + points_velocity(xv, vt, p) / nt, newton)


# --------------------------------------------------------------------------- forward model
def forward(xI, I, xJ, J, L, T, v, xv, WM, pointsI=None, landmark_map='exact'):
    """One forward pass of the original: sample points, deformed source, contrast fit.

    Returns dict(A, Xs, AI, fAI, pointsIt).  pointsIt is None without landmarks (always in 3D).
    The landmarks move with the exact inverse of the backward map, or with the original's
    forward Euler steps if landmark_map == 'euler'.
    """
    dim = len(xI)
    nt = v.shape[0]
    XJ = torch.stack(torch.meshgrid(*xJ, indexing='ij'), -1)
    A = to_A(L, T)
    Ai = torch.linalg.inv(A)
    pointsIt = None
    if dim == 2:
        Xs = (Ai[:2, :2] @ XJ[..., None])[..., 0] + Ai[:2, -1]
        for t in range(nt - 1, -1, -1):
            Xs = Xs + interp(xv, -v[t].permute(2, 0, 1), Xs.permute(2, 0, 1)).permute(1, 2, 0) / nt
        if pointsI is not None and pointsI.shape[0] > 0:
            pointsIt = torch.clone(pointsI)
            for t in range(nt):
                if landmark_map == 'euler':
                    pointsIt = pointsIt + interp(xv, v[t].permute(2, 0, 1), pointsIt.T[..., None])[..., 0].T / nt
                else:
                    pointsIt = implicit_step(xv, v[t], pointsIt, nt)
            pointsIt = (A[:2, :2] @ pointsIt.T + A[:2, -1][..., None]).T
        AI = interp(xI, I, Xs.permute(2, 0, 1), padding_mode="border")
    else:
        Xs = (Ai[:-1, :-1] @ XJ[..., None])[..., 0] + Ai[:-1, -1]
        for t in range(nt - 1, -1, -1):
            Xs = Xs + interp3D(xv, -v[t].permute(3, 0, 1, 2), Xs.permute(3, 0, 1, 2)).permute(1, 2, 3, 0) / nt
        AI = interp3D(xI, I, Xs.permute(3, 0, 1, 2), padding_mode="border")
    # contrast: least squares fit of J by an affine function of the channels of AI (no gradient)
    npix = int(np.prod(AI.shape[1:]))
    B = torch.ones(1 + AI.shape[0], npix, device=AI.device, dtype=AI.dtype)
    B[1:AI.shape[0] + 1] = AI.reshape(AI.shape[0], -1)
    with torch.no_grad():
        BB = B @ (B * WM.ravel()).T
        BJ = B @ ((J * WM).reshape(J.shape[0], npix)).T
        small = 0.1
        coeffs = torch.linalg.solve(BB + small * torch.eye(BB.shape[0], device=BB.device, dtype=BB.dtype), BJ)
    fAI = ((B.T @ coeffs).T).reshape(J.shape)
    return dict(A=A, Xs=Xs, AI=AI, fAI=fAI, pointsIt=pointsIt)


def _inputs(xI, I, xJ, J, pointsI=None, pointsJ=None):
    """Original input conventions; 3D targets get a slice axis and a z coordinate."""
    dim = len(xI)
    xI = [_t(x) for x in xI]
    I = _t(I)
    J = _t(J)
    if dim == 3:
        if J.ndim == 3:
            J = J[:, None]
        if len(xJ) == 2:
            xJ = [[0.0], xJ[0], xJ[1]]
    xJ = [_t(x) for x in xJ]
    if pointsI is not None:
        pointsI, pointsJ = _t(pointsI), _t(pointsJ)
    return xI, I, xJ, J, pointsI, pointsJ


# --------------------------------------------------------------------------- mixture EM
def mixture_state(mixture, J, muA, muB, sigmaM, sigmaA, sigmaB):
    """Initial state of the mixture EM for mixture='em' or a dict of options (2D target J (C, H, W)).

    The class means are the user's (coerced) muA / muB, else the mean of J and the lower median
    of the outer 1/20 of J's rows and columns."""
    opts = dict(MIXTURE_EM, **({} if mixture == 'em' else mixture))
    C, H, W = J.shape
    estA, estB = muA is None and not opts['fix_muA'], muB is None and not opts['fix_muB']
    muA = J.mean(dim=(1, 2)) if muA is None else coerce_mu(muA, C, 'muA')
    if muB is None:
        bh, bw = max(1, H // 20), max(1, W // 20)
        mask = torch.zeros((H, W), dtype=torch.bool)
        mask[:bh] = mask[-bh:] = True
        mask[:, :bw] = mask[:, -bw:] = True
        muB = J[:, mask].median(dim=1).values
    else:
        muB = coerce_mu(muB, C, 'muB')
    pi = torch.tensor(opts['pi_init'], dtype=DTYPE)
    return dict(opts, pi=pi / pi.sum(), muA=muA, muB=muB, estA=estA, estB=estB, vM=None, vA=None, vB=None,
                rA=sigmaA / sigmaM, rB=sigmaB / sigmaM)


def mixture_due(s, it):
    return it >= s['start'] and (it - s['start']) % s['every'] == 0


def mixture_step(s, r, J):
    """E-step and M-step of the mixture EM from the residual r = fAI - J.  Updates the state s
    and returns the new weights (WM, WA, WB)."""
    C = J.shape[0]
    grid = (Ellipsis, None, None)
    d2M = torch.sum(r ** 2, 0)
    if s['vM'] is None:
        s['vM'] = max(float(d2M.mean()) / C, VARIANCE_FLOOR)
        s['vA'], s['vB'] = s['rA'] ** 2 * s['vM'], s['rB'] ** 2 * s['vM']
    lp = torch.log(s['pi'])

    def loglik(k, d2, var):
        if 'MAB'[k] not in s['classes']:
            return torch.full_like(d2, -np.inf)
        return lp[k] - d2 / (2 * var) - 0.5 * C * np.log(var)

    W = torch.softmax(torch.stack((loglik(0, d2M, s['vM']),
                                   loglik(1, torch.sum((J - s['muA'][grid]) ** 2, 0), s['vA']),
                                   loglik(2, torch.sum((J - s['muB'][grid]) ** 2, 0), s['vB']))), 0)
    WM, WA, WB = W
    sums = W.sum(dim=(1, 2))
    pi = torch.clamp(sums / sums.sum(), min=1e-4)
    if pi[0] < s['pi_min']:
        pi = torch.cat((pi.new_tensor([s['pi_min']]), pi[1:] / pi[1:].sum() * (1 - s['pi_min'])))
    s['pi'] = pi / pi.sum()
    sM, sA, sB = (float(x) for x in sums)
    if s['estA'] and sA > 1e-3:
        s['muA'] = torch.sum(WA * J, dim=(1, 2)) / sA
    if s['estB'] and sB > 1e-3:
        s['muB'] = torch.sum(WB * J, dim=(1, 2)) / sB
    if sM > 0:
        s['vM'] = max(float(torch.sum(WM * d2M)) / (C * sM), VARIANCE_FLOOR)
    s['vA'], s['vB'] = s['rA'] ** 2 * s['vM'], s['rB'] ** 2 * s['vM']
    return WM.clone(), WA.clone(), WB.clone()


# --------------------------------------------------------------------------- gradients at one state
def gradients(xI, I, xJ, J, L, T, v, xv, WM, a=500.0, p=2.0, sigmaM=1.0, sigmaR=5e5, sigmaP=2e1,
              pointsI=None, pointsJ=None, legacy_er3d=False, landmark_map='exact'):
    """Gradients of the objective at one state, as one iteration of the loop computes them.

    2D if len(xI) == 2, else 3D-to-slice.  Inputs use the original layouts (v is
    (nt, *nv, d)).  Returns a dict of tensors:
      Lgrad, Tgrad : dE/dL, dE/dT (what the original reads from L.grad, T.grad)
      vgrad_total  : dE/dv (the original's v.grad, including the regularizer)
      vgrad        : d(EM + EP)/dv (matching and landmark terms only)
      direction    : K * dE/dv, the direction the update subtracts (times epV)
      EM, ER, EP   : energies (EP is None without landmarks)
      Xs, AI, fAI  : sample points, deformed source and its contrast fit
      pointsIt     : the transformed landmarks (None without landmarks)
    landmark_map is 'exact' or 'euler' (see the module docstring).
    """
    xI, I, xJ, J, pointsI, pointsJ = _inputs(xI, I, xJ, J, pointsI, pointsJ)
    xv = [_t(x) for x in xv]
    L, T, v = (_t(x).requires_grad_(True) for x in (L, T, v))
    WM = _t(WM).reshape(J.shape[1:])
    LL, K, DV = kernels(xv, a, p)
    f = forward(xI, I, xJ, J, L, T, v, xv, WM, pointsI, landmark_map)
    EM = torch.sum((f['fAI'] - J) ** 2 * WM) / 2.0 / sigmaM ** 2
    ER = regularizer(v, LL, DV, sigmaR, legacy_er3d)
    EP = None
    E_data = EM
    if f['pointsIt'] is not None:
        EP = torch.sum((f['pointsIt'] - pointsJ) ** 2) / 2.0 / sigmaP ** 2
        E_data = EM + EP
    gL, gT, gv = torch.autograd.grad(E_data, (L, T, v), retain_graph=True)
    gvR, = torch.autograd.grad(ER, (v,))
    total = gv + gvR
    return dict(Lgrad=gL, Tgrad=gT, vgrad=gv, vgrad_total=total, direction=smooth(total, K),
                EM=EM.detach(), ER=ER.detach(), EP=None if EP is None else EP.detach(),
                Xs=f['Xs'].detach(), AI=f['AI'].detach(), fAI=f['fAI'].detach(),
                pointsIt=None if f['pointsIt'] is None else f['pointsIt'].detach())


# --------------------------------------------------------------------------- the loop
def _run(xI, I, xJ, J, pointsI, pointsJ, L, T, A, v, xv, a, p, expand, nt, niter, diffeo_start,
         epL, epT, epV, sigmaM, sigmaB, sigmaA, sigmaR, sigmaP, muA, muB, legacy_er3d, affine_step_control,
         landmark_map='exact', mixture=None):
    xI, I, xJ, J, pointsI, pointsJ = _inputs(xI, I, xJ, J, pointsI, pointsJ)
    dim = len(xI)
    if A is not None:
        if L is not None or T is not None:
            raise Exception('If specifying A, you must not specify L or T')
        A = _t(A)
        L, T = A[:dim, :dim], A[:dim, -1]
    L = torch.eye(dim, dtype=DTYPE) if L is None else _t(L)
    T = torch.zeros(dim, dtype=DTYPE) if T is None else _t(T)
    L.requires_grad_(True)
    T.requires_grad_(True)
    if v is not None and xv is not None:
        v = _t(v)
        xv = [_t(x) for x in xv]
        nt = v.shape[0]
    elif v is None and xv is None:
        xv = velocity_grid(xI, expand, a)
        v = torch.zeros((nt,) + tuple(len(x) for x in xv) + (dim,), dtype=DTYPE)
    else:
        raise Exception('If inputting an initial v, must input both xv and v')
    v.requires_grad_(True)
    LL, K, DV = kernels(xv, a, p)
    if dim == 3:
        # the original keeps these step sizes as tensors in 3D
        epL = torch.tensor(epL, dtype=DTYPE)
        epT = torch.tensor(epT, dtype=DTYPE)
    if pointsI is None or dim == 3:
        pointsI = pointsJ = None
    WM = torch.ones(J[0].shape, dtype=J.dtype) * 0.5
    WB = torch.ones(J[0].shape, dtype=J.dtype) * 0.4
    WA = torch.ones(J[0].shape, dtype=J.dtype) * 0.1
    em = None
    if mixture is not None:
        em = mixture_state(mixture, J, muA, muB, sigmaM, sigmaA, sigmaB)
        WM, WB, WA = torch.ones_like(WM), torch.zeros_like(WB), torch.zeros_like(WA)
    estimate_muA = muA is None
    estimate_muB = muB is None
    muA = coerce_mu(muA, J.shape[0], 'muA')
    muB = coerce_mu(muB, J.shape[0], 'muB')
    sdims = tuple(range(-dim, 0))  # spatial dims of J
    pdims = tuple(range(1, dim + 1))  # spatial dims of a stack of weights
    grid = (Ellipsis,) + (None,) * dim  # mu[..., None, None(, None)]

    if affine_step_control:
        m = torch.ones(dim * dim + dim, dtype=DTYPE)
        g_prev = torch.zeros(dim * dim + dim, dtype=DTYPE)

    Esave = []
    A_eval, v_eval = to_A(L, T).detach(), v.detach().clone()
    Xs = None
    for it in range(niter):
        f = forward(xI, I, xJ, J, L, T, v, xv, WM, pointsI, landmark_map)
        A_eval, v_eval, Xs, fAI = f['A'].detach().clone(), v.detach().clone(), f['Xs'], f['fAI']
        EM = torch.sum((fAI - J) ** 2 * WM) / 2.0 / sigmaM ** 2
        ER = regularizer(v, LL, DV, sigmaR, legacy_er3d)
        E = EM + ER
        tosave = [E.item(), EM.item(), ER.item()]
        if f['pointsIt'] is not None:
            EP = torch.sum((f['pointsIt'] - pointsJ) ** 2) / 2.0 / sigmaP ** 2
            E += EP
            tosave.append(EP.item())
        Esave.append(tosave)
        E.backward()
        with torch.no_grad():
            if not affine_step_control:
                L -= (epL / (1.0 + (it >= diffeo_start) * 9)) * L.grad
                T -= (epT / (1.0 + (it >= diffeo_start) * 9)) * T.grad
            else:
                g = torch.cat((L.grad.reshape(-1), T.grad.reshape(-1)))
                s = g * g_prev
                m = torch.where(s < 0, torch.maximum(SHRINK * m, torch.tensor(FLOOR, dtype=DTYPE)),
                                torch.where(s > 0, torch.minimum(GROW * m, torch.tensor(CAP, dtype=DTYPE)), m))
                g_prev = g
                fac = 1.0 + (it >= diffeo_start) * 9
                L -= (epL / fac) * m[:dim * dim].reshape(dim, dim) * L.grad
                T -= (epT / fac) * m[dim * dim:] * T.grad
            L.grad.zero_()
            T.grad.zero_()
            if it >= diffeo_start:
                v -= smooth(v.grad, K) * epV
            v.grad.zero_()
        if em is not None:  # the mixture EM replaces the original weight updates
            if mixture_due(em, it):
                with torch.no_grad():
                    WM, WA, WB = mixture_step(em, fAI.detach() - J, J)
        # update weights: M step for the class means, then (from iteration 50) the E step
        elif not it % 5:
            with torch.no_grad():
                if estimate_muA:
                    muA = torch.sum(WA * J, dim=sdims) / torch.sum(WA)
                if estimate_muB:
                    muB = torch.sum(WB * J, dim=sdims) / torch.sum(WB)
                if it >= 50:
                    W = torch.stack((WM, WA, WB))
                    pi = torch.sum(W, dim=pdims)
                    pi += torch.max(pi) * 1e-6
                    pi /= torch.sum(pi)
                    WM = pi[0] * torch.exp(-torch.sum((fAI - J) ** 2, 0) / 2.0 / sigmaM ** 2) / np.sqrt(2.0 * np.pi * sigmaM ** 2) ** J.shape[0]
                    WA = pi[1] * torch.exp(-torch.sum((muA[grid] - J) ** 2, 0) / 2.0 / sigmaA ** 2) / np.sqrt(2.0 * np.pi * sigmaA ** 2) ** J.shape[0]
                    WB = pi[2] * torch.exp(-torch.sum((muB[grid] - J) ** 2, 0) / 2.0 / sigmaB ** 2) / np.sqrt(2.0 * np.pi * sigmaB ** 2) ** J.shape[0]
                    WS = WM + WB + WA
                    WS += torch.max(WS) * 1e-6
                    WM /= WS
                    WB /= WS
                    WA /= WS
    out = {'A': A_eval, 'v': v_eval, 'xv': xv,
           'WM': WM.detach().clone(), 'WB': WB.detach().clone(), 'WA': WA.detach().clone(),
           'A_next': to_A(L, T).detach(), 'v_next': v.detach().clone(),
           'Esave': np.array(Esave) if Esave else np.zeros((0, 3))}
    if affine_step_control:
        out['step_multipliers'] = m.clone()
    if dim == 3:
        if Xs is None:  # niter == 0: sample points of the initial state
            with torch.no_grad():
                Xs = forward(xI, I, xJ, J, L, T, v, xv, WM)['Xs']
        out['Xs'] = Xs.detach().clone()
    return out


def LDDMM(xI, I, xJ, J, pointsI=None, pointsJ=None, L=None, T=None, A=None, v=None, xv=None,
          a=500.0, p=2.0, expand=2.0, nt=3, niter=5000, diffeo_start=0, epL=2e-8, epT=2e-1, epV=2e3,
          sigmaM=1.0, sigmaB=2.0, sigmaA=5.0, sigmaR=5e5, sigmaP=2e1, muB=None, muA=None,
          affine_step_control=False, landmark_map='exact', mixture=None):
    """Oracle of the 2D ``LDDMM`` (float64, CPU), same arguments and defaults as the public function,
    with the landmark map and the mixture EM of the module docstring.  The original is
    landmark_map='euler', mixture=None."""
    if (pointsI is None) != (pointsJ is None):
        raise Exception('Must specify corresponding sets of points or none at all')
    return _run(xI, I, xJ, J, pointsI, pointsJ, L, T, A, v, xv, a, p, expand, nt, niter, diffeo_start,
                epL, epT, epV, sigmaM, sigmaB, sigmaA, sigmaR, sigmaP, muA, muB, False, affine_step_control,
                landmark_map, mixture)


def LDDMM_3D_to_slice(xI, I, xJ, J, pointsI=None, pointsJ=None, L=None, T=None, A=None, v=None, xv=None,
                      a=500.0, p=2.0, expand=1.25, nt=3, niter=5000, diffeo_start=0, epL=1e-6, epT=1e1,
                      epV=1e3, sigmaM=1.0, sigmaB=2.0, sigmaA=5.0, sigmaR=SIGMAR_3D, sigmaP=2e1, muA=None,
                      muB=None, legacy_er3d=False, affine_step_control=True):
    """Oracle of ``LDDMM_3D_to_slice`` (float64, CPU).  Landmarks are ignored, as in the original.

    The original is legacy_er3d=True, affine_step_control=False (and sigmaR=1e8)."""
    return _run(xI, I, xJ, J, None, None, L, T, A, v, xv, a, p, expand, nt, niter, diffeo_start,
                epL, epT, epV, sigmaM, sigmaB, sigmaA, sigmaR, sigmaP, muA, muB, legacy_er3d, affine_step_control)


# --------------------------------------------------------------------------- synthetic test problems
def _blobs(X, centers, widths, amps):
    """Sum of Gaussian blobs evaluated on the coordinate arrays X: (C, *X[0].shape)."""
    out = 0.0
    for c, w, amp in zip(centers, widths, amps):
        r2 = sum((x - ci) ** 2 for x, ci in zip(X, c))
        out = out + amp.reshape((-1,) + (1,) * X[0].ndim) * np.exp(-r2 / (2 * w ** 2))[None]
    return out


def rotation_2d(deg):
    t = np.deg2rad(deg)
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


def rotation_3d(deg_x, deg_z):
    """Rotation by deg_x about the first axis followed by deg_z about the last axis."""
    a, b = np.deg2rad(deg_x), np.deg2rad(deg_z)
    Rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    Rz = np.array([[np.cos(b), 0, -np.sin(b)], [0, 1, 0], [np.sin(b), 0, np.cos(b)]])
    return Rx @ Rz


def synthetic_2d(CI=3, CJ=2, seed=0, shape_J=(50, 44)):
    """Smooth multi-channel source and target related by a rotation, a shift and a channel mix.

    Returns dict(xI, I, xJ, J, pointsI, pointsJ) with numpy arrays (the landmarks include two
    outliers).  Grids have different, non-unit spacings; ``shape_J`` sets the number of target
    pixels over the same field of view.
    """
    rng = np.random.default_rng(seed)
    xI = [np.arange(56) * 1.0 - 27.0, np.arange(48) * 1.25 - 29.0]
    xJ = [np.arange(shape_J[0]) * (55.0 / shape_J[0]) - 26.0, np.arange(shape_J[1]) * (57.2 / shape_J[1]) - 28.0]
    nb = 14
    cen = rng.uniform(-21, 21, size=(nb, 2))
    wid = rng.uniform(3.0, 7.0, size=nb)
    amp = rng.uniform(0.2, 1.0, size=(nb, CI))
    I = _blobs(np.meshgrid(*xI, indexing='ij'), cen, wid, amp)
    R = 0.95 * rotation_2d(8.0)
    c = np.array([2.0, -1.0])
    XJ = np.meshgrid(*xJ, indexing='ij')
    J0 = _blobs([R[0, 0] * XJ[0] + R[0, 1] * XJ[1] + c[0], R[1, 0] * XJ[0] + R[1, 1] * XJ[1] + c[1]], cen, wid, amp)
    mix = rng.uniform(0.3, 1.2, size=(CJ, CI))
    J = np.einsum('dc,chw->dhw', mix, J0) + 0.02 * rng.standard_normal((CJ,) + XJ[0].shape)
    pI = cen[:5].copy()
    pJ = np.linalg.solve(R, (pI - c).T).T
    pI = np.concatenate((pI, [[26.0, -28.0], [-27.0, 30.0]]))
    pJ = np.concatenate((pJ, [[30.0, -31.0], [-29.0, 34.0]]))
    return dict(xI=xI, I=I, xJ=xJ, J=J, pointsI=pI, pointsJ=pJ)


def synthetic_3d(seed=0):
    """Two-channel 3D source volume and a tilted 2D slice target, like the Allen atlas cases.

    Returns dict(xI, I, xJ, J): xJ has the two in-plane axes and J is (1, H, W).
    """
    rng = np.random.default_rng(seed)
    xI = [np.arange(20) * 2.0 - 19.0, np.arange(24) * 1.5 - 17.25, np.arange(28) * 1.25 - 16.875]
    nb = 16
    cen = rng.uniform(-14, 14, size=(nb, 3))
    wid = rng.uniform(2.5, 6.0, size=nb)
    amp = rng.uniform(0.2, 1.0, size=(nb, 1))
    f = _blobs(np.meshgrid(*xI, indexing='ij'), cen, wid, amp)[0]
    f = f / np.mean(np.abs(f))
    I = np.stack((f, (f - f.mean()) ** 2))
    y = np.arange(36) * 1.0 - 17.5
    x = np.arange(40) * 1.0 - 19.5
    Y, X = np.meshgrid(y, x, indexing='ij')
    J = _blobs([3.0 + 0.2 * Y - 0.1 * X, Y, X], cen, wid, amp)[0]
    J = J / np.mean(np.abs(J)) + 0.02 * rng.standard_normal(J.shape)
    return dict(xI=xI, I=I, xJ=[y, x], J=J[None])


def smooth_random_v(xv, nt, amp, seed):
    """Smooth random velocity field in the original layout (nt, *nv, d) with max |v| == amp."""
    dim = len(xv)
    g = torch.Generator().manual_seed(seed)
    v = torch.randn((nt,) + tuple(len(x) for x in xv) + (dim,), generator=g, dtype=DTYPE)
    dims = tuple(range(1, dim + 1))
    F2 = sum(torch.meshgrid(*[torch.fft.fftfreq(len(x), dtype=DTYPE) ** 2 for x in xv], indexing='ij'))
    v = torch.fft.ifftn(torch.fft.fftn(v, dim=dims) * torch.exp(-F2 / (2 * 0.06 ** 2))[..., None], dim=dims).real
    return v * (amp / v.abs().max())


# Fit parameters for the affine step control tests: steps large enough that the affine
# parameters oscillate, so that their multipliers shrink (nearly to the floor) and grow back.
# (With epL=1e-4 and the exact landmark map, L oscillates with growing amplitude for ~20
# iterations before its multiplier shrinks, which amplifies rounding differences ~1e4-fold.)
STEP_2D = dict(a=12.0, sigmaM=0.3, sigmaR=5e3, sigmaP=4.0, epT=1e-1, epL=5e-5, epV=0.2, niter=300,
               diffeo_start=30)
STEP_3D = dict(nt=4, a=10.0, expand=1.25, sigmaM=0.5, sigmaR=5e2, epT=1e-1, epL=1e-4, epV=5.0, niter=300,
               diffeo_start=20, L=0.95 * rotation_3d(4.0, -3.0), T=[0.5, 0.3, -0.4])


def step_control_problem(dim):
    """Inputs and fit parameters (a dict of LDDMM / LDDMM_3D_to_slice keyword arguments) on
    which the affine step control engages.  2D: three source and three target channels and
    landmarks."""
    if dim == 2:
        return dict(synthetic_2d(CI=3, CJ=3), **STEP_2D)
    return dict(synthetic_3d(), **STEP_3D)
