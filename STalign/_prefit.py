'''Settings computed from the images before an LDDMM fit.

* :func:`affine_prefit`: a coarse affine pre-alignment.  Many starts (rotations of the
  initial affine in 2D; tilted and shifted planes in 3D) are fitted at once on mean-pooled
  images, the best one is refined at a finer level, and LDDMM starts from the result.
* :func:`physical_sigmaR`: the regularization weight sigmaR for a prior displacement scale
  given in micrometers, so that the deformation prior does not depend on the pixel size, the
  intensity scale of the target or the smoothness scale a.
* :func:`curvature_steps`: gradient descent steps scaled by a curvature estimate of the image
  matching term (2D).
* :func:`stable_steps`: default steps capped where gradient descent would be unstable.  In 2D
  from the curvature of the energy in the velocity field and in the affine at the initial
  state, from the image term (the same assumptions as curvature_steps), the landmarks and the
  regularizer; in 3D only epV, from the regularizer.

The pre-fit matches the target J with a linear contrast map of the source [1, I_1, ..., I_C]
(least squares with LDDMM's 0.1 ridge), with weights that are 1 on target tissue and lower
on target background, and an intensity bias that varies linearly in space.  Coordinates are
centred on the target's tissue centroid and scaled by its radius, so one step size serves
rotation, scale, shear and translation.  Everything is deterministic: there is no random
start, and ties between starts go to the start closest to the initial affine.
'''
import math
import time

import numpy as np
import torch
from torch.nn.functional import grid_sample

F64 = torch.float64

# Coarse search images have about this many pixels along the longer target axis; the best
# start is refined at twice that resolution.
SEARCH_PIXELS = 80
# 2D starts: the initial affine rotated about the target tissue centroid by k * 360 / N degrees.
ANGLES_2D = 12
ITERS_2D = 100
REFINE_ITERS_2D = 60
# 3D starts: the initial plane tilted about its two in-plane axes by every pair of TILTS_3D
# (degrees) and shifted along its normal by every value of OFFSETS_3D (coordinate units, um).
TILTS_3D = (-6.0, 0.0, 6.0)
OFFSETS_3D = (-300.0, 0.0, 300.0)
ITERS_3D = 200
REFINE_ITERS_3D = 100
# Adam learning rate (halved for the refinement), the matching weight of target background
# relative to target tissue, the order of the spatial intensity bias (0: none, 1: linear),
# and the relative energy difference within which starts tie.
LEARNING_RATE = 0.02
BACKGROUND_WEIGHT = 0.1
BIAS_ORDER = 1
TIE_TOLERANCE = 0.01

# physical_sigmaR: the reference pixel size (um) and the constant C_d of the prior variance
# Var v = sigmaR^2 / (C_d pi a^d) per time step of the regularizer (1 - a^2 Laplacian)^(2p)
# at p = 2 in d dimensions.
PIXEL_SIZE_REF = 30.0
PRIOR_CONSTANT = {2: 12.0, 3: 64.0}

# curvature_steps: the curvature at which the step sizes of earlier versions are kept.
CURVATURE_REF = 1.1

# stable_steps: gradient descent with step size e on a quadratic of curvature c is stable when
# e c < 2 and decreases the energy monotonically when e c <= 1.  The default velocity step
# keeps epV (lambda_v + DV / sigmaR^2) <= STEP_LIMIT_V and the default affine steps keep the
# largest eigenvalue of their scaled affine Hessian <= STEP_LIMIT_AFFINE (see stable_steps), so
# the joint step stays below 1.5 (lambda_max of a block matrix <= the sum over its blocks).
# lambda_v comes from a subspace iteration on a block of EIG_BLOCK vectors (so that nearly
# equal largest eigenvalues cannot stall it), which stops after EIG_ITERATIONS steps or when
# the largest Ritz value changes by less than EIG_TOL (relative).
STEP_LIMIT_V = 1.0
STEP_LIMIT_AFFINE = 0.5
EIG_BLOCK = 8
EIG_ITERATIONS = 100
EIG_TOL = 1e-6


# =========================================================================== images
def pool_image(x, I, f):
    '''Mean-pool an image by an integer factor along every spatial axis.

    Parameters
    ----------
    x : list of 1D arrays
        Pixel locations along each axis.
    I : array, (C, *n)
        Image with channels first.
    f : int or tuple of int
        Pooling factor (per axis). The image is padded at the end with its edge values to a
        multiple of f.

    Returns
    -------
    x_pooled, I_pooled : list of 1D arrays, array
        Pixel locations (the means of the pooled locations) and the pooled image, float64.
    '''
    I = np.asarray(I, dtype=np.float64)
    d = I.ndim - 1
    fs = (f,) * d if np.isscalar(f) else tuple(f)
    if all(int(k) == 1 for k in fs):
        return [np.asarray(xi, np.float64).copy() for xi in x], I.copy()
    pad = [(0, 0)] + [(0, (-n) % int(k)) for n, k in zip(I.shape[1:], fs)]
    Ip = np.pad(I, pad, mode='edge')
    shape = [I.shape[0]]
    for n, k in zip(Ip.shape[1:], fs):
        shape += [n // int(k), int(k)]
    Ip = Ip.reshape(shape).mean(axis=tuple(range(2, 2 * d + 1, 2)))
    xo = []
    for xi, k, (_, pe) in zip(x, fs, pad[1:]):
        xi = np.asarray(xi, np.float64)
        dxi = xi[1] - xi[0] if len(xi) > 1 else 1.0
        xe = np.concatenate([xi, xi[-1] + dxi * np.arange(1, pe + 1)])
        xo.append(xe.reshape(-1, int(k)).mean(1))
    return xo, Ip


def pixel_size(x):
    '''Mean pixel spacing over the axes of a grid.'''
    return float(np.mean([abs(xi[1] - xi[0]) for xi in x if len(xi) > 1]))


def tissue_weights(J):
    '''Per-pixel tissue signal of a 2D target (C, H, W): the mean over channels of
    |J - background|, the background being the median of the image border.'''
    J = np.asarray(J, np.float64)
    Jm = J.reshape(J.shape[0], -1)
    border = np.concatenate([J[:, 0, :], J[:, -1, :], J[:, :, 0], J[:, :, -1]], axis=1)
    bg = np.median(border, axis=1)
    return np.abs(Jm - bg[:, None]).mean(0).reshape(J.shape[1:])


def match_weights(J, w_bg, frac=0.15):
    '''Matching weight per target pixel: 1 on target tissue, w_bg on target background.

    Tissue is the tissue signal above `frac` of its 99th percentile, softened linearly below.
    With w_bg < 1 source tissue can fall onto target background cheaply (a target with missing
    tissue), while every target tissue pixel must still be explained.'''
    w = tissue_weights(J)
    thr = frac * np.percentile(w, 99)
    t = np.clip(w / max(thr, 1e-300), 0.0, 1.0)
    return w_bg + (1.0 - w_bg) * t


def weighted_centroid(x, w):
    '''Centroid and rms radius of the weights w on the grid x.'''
    X = np.meshgrid(*[np.asarray(xi, np.float64) for xi in x], indexing='ij')
    s = w.sum()
    c = np.array([(Xi * w).sum() / s for Xi in X])
    r = math.sqrt(sum(((Xi - ci) ** 2 * w).sum() for Xi, ci in zip(X, c)) / s)
    return c, r


def _grid_norm(x):
    '''(lo, span) tensors of a grid, for grid_sample's normalized coordinates.'''
    lo = torch.tensor([float(xi[0]) for xi in x], dtype=F64)
    span = torch.tensor([float(xi[-1] - xi[0]) for xi in x], dtype=F64)
    return lo, span


def _sample(image, lo, span, X):
    '''Sample image (1, C, *n) at points X (K, *m, d) in axis order (d = 2: bilinear, d = 3:
    trilinear, border padding, as LDDMM samples the source).  Returns (K, C, *m).'''
    g = ((X - lo) / span * 2.0 - 1.0).flip(-1)
    K = X.shape[0]
    if X.shape[-1] == 2:
        return grid_sample(image.expand(K, *image.shape[1:]), g, align_corners=True, padding_mode='border')
    out = grid_sample(image.expand(K, *image.shape[1:]), g[:, None], align_corners=True, padding_mode='border')
    return out[:, :, 0]


def _bias_basis(Y, c, r, order):
    '''Spatial basis (q, N) of a smoothly varying contrast at target points Y (..., 2):
    1, plus the two normalized coordinates for order >= 1, plus their products for order 2.'''
    if not order:
        return None
    Yn = ((Y - torch.as_tensor(c, dtype=F64)) / r).reshape(-1, 2).T
    rows = [torch.ones_like(Yn[0]), Yn[0], Yn[1]]
    if order >= 2:
        rows += [Yn[0] * Yn[0], Yn[0] * Yn[1], Yn[1] * Yn[1]]
    return torch.stack(rows)


def _contrast_residual(AI, Jt, W, P=None, ridge=0.1):
    '''Weighted least squares residual of the target against [1, AI] for every start.

    AI (K, C, N) deformed source, Jt (C_J, N) target, W (N,) weights.  With a spatial basis
    P (q, N) the regressors are [1, AI] times every row of P.  Returns E (K,) = sum W (fAI - J)^2 / 2.
    '''
    K, C, N = AI.shape
    B = torch.cat([torch.ones(K, 1, N, dtype=AI.dtype), AI], 1)
    if P is not None:
        B = (B[:, :, None, :] * P[None, None]).reshape(K, -1, N)
        C = B.shape[1] - 1
    BW = B * W
    BB = BW @ B.transpose(1, 2) + ridge * torch.eye(1 + C, dtype=AI.dtype)
    coef = torch.linalg.solve(BB, BW @ Jt.T)
    fAI = coef.transpose(1, 2) @ B
    return 0.5 * ((fAI - Jt) ** 2 * W).sum((1, 2))


def _adam(params, energy, iters, lr, betas=(0.9, 0.99), decay_at=0.7):
    '''Batched Adam on leaf tensors whose first axis is the start; energy() returns the energy
    of every start (K,).  Each start's energy is divided by its initial value so all starts move
    at a similar speed, and the learning rate drops 10x after decay_at * iters iterations.
    Returns the final energies.'''
    opt = torch.optim.Adam(params, lr=lr, betas=betas)
    E0 = None
    for it in range(iters):
        if it == int(decay_at * iters):
            for group in opt.param_groups:
                group['lr'] = lr * 0.1
        opt.zero_grad()
        E = energy()
        if E0 is None:
            E0 = E.detach().clamp_min(1e-30)
        (E / E0).sum().backward()
        opt.step()
    with torch.no_grad():
        return energy()


# =========================================================================== 2D
def search_2d(xI, I, xJ, J, L, T, *, pool, refine_pool=None, n_angles=ANGLES_2D, iters=ITERS_2D,
              refine_iters=REFINE_ITERS_2D, lr=LEARNING_RATE, w_bg=BACKGROUND_WEIGHT, bias=BIAS_ORDER,
              tie_tol=TIE_TOLERANCE, sigmaM=1.0, pointsI=None, pointsJ=None, sigmaP=20.0):
    '''Multi-start affine pre-alignment of a 2D source to a 2D target.

    The target-to-source map is x = M (y - cJ) + c, with cJ the target tissue centroid.  The
    starts are the initial affine (L, T) composed with rotations about cJ by k * 360 / n_angles
    degrees.  Every start is fitted (Adam on M and c) on the target pooled by `pool` (and the
    source pooled to about the same pixel size); the start with the lowest energy wins, where
    starts within tie_tol (relative) of the lowest tie and the smallest rotation among them
    wins.  It is then refined on the images pooled by `refine_pool` (None: no refinement).

    With landmarks (pointsI, pointsJ) the energy includes STalign's landmark term (weighted
    by sigmaP, the image term scaled to the full-resolution pixel count).

    Returns
    -------
    dict
        'L', 'T': the pre-fitted affine (target = L source + T); 'energy': the final energy of
        every start; 'starts': the rotation of every start (degrees); 'best': index of the
        winner; 'pool': (source, target) pooling factors of the search.
    '''
    I = np.asarray(I, np.float64)
    J = np.asarray(J, np.float64)
    dxJ, dxI = pixel_size(xJ), pixel_size(xI)
    # source pixels in target units (the source may be in other units, e.g. image pixels)
    sL = math.sqrt(abs(np.linalg.det(np.asarray(L, np.float64))))
    fJ = max(1, int(pool))
    fI = max(1, int(round(fJ * dxJ / (dxI * sL))))
    xJc, Jc = pool_image(xJ, J, fJ)
    xIc, Ic = pool_image(xI, I, fI)
    cJ, rJ = weighted_centroid(xJc, tissue_weights(Jc))
    A0 = np.eye(3)
    A0[:2, :2] = L
    A0[:2, 2] = T
    B0 = np.linalg.inv(A0)                      # target -> source
    M0 = B0[:2, :2]
    c0 = B0[:2, :2] @ cJ + B0[:2, 2]            # source point of the target centroid
    sc = float(np.sqrt(abs(np.linalg.det(M0))))
    angles = np.arange(n_angles) * (360.0 / n_angles) if n_angles > 1 else np.zeros(1)
    rotations = [np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
                 for th in (math.radians(a) for a in angles)]
    with_points = pointsI is not None and pointsJ is not None and len(pointsI) > 0
    if with_points:
        pI = torch.as_tensor(np.asarray(pointsI, np.float64))
        pJ = torch.as_tensor(np.asarray(pointsJ, np.float64))

    def fit(xIc, Ic, xJc, Jc, Ms, cs, iters, lr):
        image = torch.as_tensor(Ic, dtype=F64)[None]
        loI, spI = _grid_norm(xIc)
        Y = torch.stack(torch.meshgrid(*[torch.as_tensor(x, dtype=F64) for x in xJc], indexing='ij'), -1)
        Yn = (Y - torch.as_tensor(cJ)) / rJ
        P = _bias_basis(Y, cJ, rJ, bias)
        Jt = torch.as_tensor(Jc, dtype=F64).reshape(Jc.shape[0], -1)
        W = torch.as_tensor(0.5 * match_weights(Jc, w_bg).reshape(-1), dtype=F64)
        Ms = torch.as_tensor(np.stack(Ms), dtype=F64)
        cs = torch.as_tensor(np.stack(cs), dtype=F64)
        # M = M_start (I + G), c = c_start + rJ sc t
        G = torch.zeros(Ms.shape[0], 2, 2, dtype=F64, requires_grad=True)
        t = torch.zeros(Ms.shape[0], 2, dtype=F64, requires_grad=True)
        # the coarse image term sums over fewer pixels than the full-resolution one: scale it so
        # that its balance with the landmark term is that of LDDMM's energy
        pix_ratio = (len(xJ[0]) * len(xJ[1])) / float(Jt.shape[1])

        def energy():
            M = Ms @ (torch.eye(2, dtype=F64) + G)
            c = cs + rJ * sc * t
            X = (M[:, None, None] @ (Yn[None, ..., None] * rJ))[..., 0] + c[:, None, None]
            AI = _sample(image, loI, spI, X).reshape(X.shape[0], Ic.shape[0], -1)
            E = _contrast_residual(AI, Jt, W, P) / sigmaM ** 2
            if with_points:
                # source -> target: y = M^-1 (x - c) + cJ, and the landmark term of LDDMM
                Mi = torch.linalg.inv(M)
                p = (Mi[:, None] @ (pI[None, ..., None] - c[:, None, :, None]))[..., 0] + torch.as_tensor(cJ)
                E = E * pix_ratio + ((p - pJ) ** 2).sum((1, 2)) / 2.0 / sigmaP ** 2
            return E

        E = _adam([G, t], energy, iters, lr)
        with torch.no_grad():
            return E.numpy(), (Ms @ (torch.eye(2, dtype=F64) + G)).numpy(), (cs + rJ * sc * t).numpy()

    E, M, c = fit(xIc, Ic, xJc, Jc, [M0 @ R for R in rotations], [c0] * len(rotations), iters, lr)
    # among the starts that tie with the lowest energy keep the smallest rotation: a nearly
    # symmetric tissue cannot turn the initial orientation around on a tie
    rotation = np.minimum(angles % 360.0, 360.0 - angles % 360.0)
    near = np.flatnonzero(E <= np.nanmin(E) + tie_tol * abs(np.nanmin(E)))
    best = int(near[np.argmin(rotation[near])])
    Mb, cb = M[best], c[best]
    if refine_pool is not None and refine_iters > 0:
        fJ2 = max(1, int(refine_pool))
        fI2 = max(1, int(round(fJ2 * dxJ / (dxI * sL))))
        xJr, Jr = pool_image(xJ, J, fJ2)
        xIr, Ir = pool_image(xI, I, fI2)
        _, Mr, cr = fit(xIr, Ir, xJr, Jr, [Mb], [cb], refine_iters, lr * 0.5)
        Mb, cb = Mr[0], cr[0]
    # back to (L, T): B y = M (y - cJ) + c, A = B^-1
    B = np.eye(3)
    B[:2, :2] = Mb
    B[:2, 2] = cb - Mb @ cJ
    A = np.linalg.inv(B)
    return dict(L=A[:2, :2].copy(), T=A[:2, 2].copy(), energy=E, starts=[float(a) for a in angles], best=best,
                pool=(fI, fJ))


# =========================================================================== 3D
def _rotation_in_plane(u, w, angle):
    '''Rotation by `angle` (degrees) in the plane spanned by the unit vectors u and w (3x3 tensor).'''
    u = torch.as_tensor(u, dtype=F64)
    w = torch.as_tensor(w, dtype=F64)
    P = torch.outer(u, u) + torch.outer(w, w)
    Q = torch.outer(w, u) - torch.outer(u, w)
    th = torch.tensor(math.radians(angle), dtype=F64)
    return torch.eye(3, dtype=F64) + (torch.cos(th) - 1.0) * P + torch.sin(th) * Q


def _rotvec(w):
    '''Rodrigues formula: rotation matrices (K, 3, 3) of the rotation vectors w (K, 3).'''
    th = torch.sqrt((w ** 2).sum(-1) + 1e-30)
    k = w / th[:, None]
    Kx = torch.zeros(w.shape[0], 3, 3, dtype=F64)
    Kx[:, 0, 1], Kx[:, 0, 2] = -k[:, 2], k[:, 1]
    Kx[:, 1, 0], Kx[:, 1, 2] = k[:, 2], -k[:, 0]
    Kx[:, 2, 0], Kx[:, 2, 1] = -k[:, 1], k[:, 0]
    s, c = torch.sin(th)[:, None, None], torch.cos(th)[:, None, None]
    return torch.eye(3, dtype=F64) + s * Kx + (1 - c) * (Kx @ Kx)


def _plane_fit_data(xIc, Ic, xJc, Jc, cJ, rJ, w_bg, bias):
    image = torch.as_tensor(Ic, dtype=F64)[None]
    loI, spI = _grid_norm(xIc)
    Y = torch.stack(torch.meshgrid(*[torch.as_tensor(x, dtype=F64) for x in xJc], indexing='ij'), -1)
    Yc = Y - torch.as_tensor(cJ)
    P = _bias_basis(Y, cJ, rJ, bias)
    Jt = torch.as_tensor(Jc, dtype=F64).reshape(Jc.shape[0], -1)
    W = torch.as_tensor(0.5 * match_weights(Jc, w_bg).reshape(-1), dtype=F64)
    return image, loI, spI, Yc, P, Jt, W


def _refine_3d(xIc, Ic, xJc, Jc, M, c, cJ, rJ, iters, lr, sigmaM, w_bg, bias):
    '''Polish one plane (M (3, 2), c (3,)) with additive updates of M and c.'''
    image, loI, spI, Yc, P, Jt, W = _plane_fit_data(xIc, Ic, xJc, Jc, cJ, rJ, w_bg, bias)
    Mt = torch.as_tensor(np.asarray(M, np.float64).reshape(1, 3, 2))
    ct = torch.as_tensor(np.asarray(c, np.float64).reshape(1, 3))
    D = torch.zeros(1, 3, 2, dtype=F64, requires_grad=True)
    t = torch.zeros(1, 3, dtype=F64, requires_grad=True)
    sc = float(np.linalg.norm(np.asarray(M, np.float64), axis=0).mean())

    def energy():
        Mk = Mt + sc * D
        X = (Mk[:, None, None] @ Yc[None, ..., None])[..., 0] + (ct + rJ * t)[:, None, None]
        AI = _sample(image, loI, spI, X).reshape(1, Ic.shape[0], -1)
        return _contrast_residual(AI, Jt, W, P) / sigmaM ** 2

    _adam([D, t], energy, iters, lr)
    with torch.no_grad():
        return (Mt + sc * D).numpy()[0], (ct + rJ * t).numpy()[0]


def search_3d(xI, I, xJ, J, L, T, *, pool, refine_pool=None, tilts=TILTS_3D, offsets=OFFSETS_3D,
              iters=ITERS_3D, refine_iters=REFINE_ITERS_3D, lr=LEARNING_RATE, w_bg=BACKGROUND_WEIGHT,
              bias=BIAS_ORDER, tie_tol=TIE_TOLERANCE, sigmaM=1.0):
    '''Multi-start pre-alignment of a 3D source (an atlas) to a 2D target slice.

    The slice point y = (row, col) maps to the source point x = M (y - cJ) + c, M (3, 2) and
    c (3,), cJ the target tissue centroid.  The starts are the initial plane (from L, T)
    tilted about its two in-plane axes by every pair of `tilts` (degrees) and shifted along
    its normal by every value of `offsets` (coordinate units).  Each start fits a rotation of
    the plane, an in-plane affine map and a shift (Adam) on the target pooled by `pool`; the
    lowest energy wins, where starts within tie_tol (relative) of the lowest tie and the start
    closest to the initial plane among them wins.  The winner is refined on the images pooled
    by `refine_pool` (None: no refinement).

    Returns
    -------
    dict
        'L', 'T': the pre-fitted 3D affine; 'energy': the final energy of every start;
        'starts': (tilt 1, tilt 2, offset) of every start; 'best': index of the winner;
        'pool': (source, target) pooling factors of the search.
    '''
    I = np.asarray(I, np.float64)
    J = np.asarray(J, np.float64)
    if J.ndim == 4:
        J = J[:, 0]
    xJ = [np.asarray(x, np.float64).reshape(-1) for x in (xJ if len(xJ) == 2 else xJ[1:])]
    dxJ, dxI = pixel_size(xJ), pixel_size(xI)
    A0 = np.eye(4)
    A0[:3, :3] = L
    A0[:3, 3] = T
    B0 = np.linalg.inv(A0)
    M0 = B0[:3, 1:3]                            # in-plane directions in the source
    s_in = float(np.mean(np.linalg.norm(M0, axis=0)))   # source units per target unit
    fJ = max(1, int(pool))
    fI = max(1, int(round(fJ * dxJ * s_in / dxI)))
    xJc, Jc = pool_image(xJ, J, fJ)
    xIc, Ic = pool_image(xI, I, fI)
    cJ, rJ = weighted_centroid(xJc, tissue_weights(Jc))
    c0 = B0[:3, 1:3] @ cJ + B0[:3, 3]           # source point of the target centroid (z = 0)
    e1 = M0[:, 0] / np.linalg.norm(M0[:, 0])
    e2 = M0[:, 1] - e1 * (e1 @ M0[:, 1])
    e2 /= np.linalg.norm(e2)
    normal = np.cross(e1, e2)
    starts = [(ta, tb, off) for ta in tilts for tb in tilts for off in offsets]
    Rs = [(_rotation_in_plane(normal, e1, ta) @ _rotation_in_plane(normal, e2, tb)).numpy() for ta, tb, _ in starts]
    cs = [c0 + off * normal for _, _, off in starts]

    image, loI, spI, Yc, P, Jt, W = _plane_fit_data(xIc, Ic, xJc, Jc, cJ, rJ, w_bg, bias)
    K = len(starts)
    Rt = torch.as_tensor(np.stack(Rs), dtype=F64)
    ct = torch.as_tensor(np.stack(cs), dtype=F64)
    M0t = torch.as_tensor(M0, dtype=F64)
    w = torch.zeros(K, 3, dtype=F64, requires_grad=True)     # rotation vector about the start
    t = torch.zeros(K, 3, dtype=F64, requires_grad=True)
    G = torch.zeros(K, 2, 2, dtype=F64, requires_grad=True)  # in-plane affine

    def plane():
        return (_rotvec(w) @ Rt) @ M0t @ (torch.eye(2, dtype=F64) + G)

    def energy():
        X = (plane()[:, None, None] @ Yc[None, ..., None])[..., 0] + (ct + rJ * t)[:, None, None]
        AI = _sample(image, loI, spI, X).reshape(K, Ic.shape[0], -1)
        return _contrast_residual(AI, Jt, W, P) / sigmaM ** 2

    E = _adam([w, t, G], energy, iters, lr).numpy()
    with torch.no_grad():
        M, c = plane().numpy(), (ct + rJ * t).numpy()
    distance = np.array([abs(ta) + abs(tb) + abs(off) / 100.0 for ta, tb, off in starts])
    near = np.flatnonzero(E <= np.nanmin(E) + tie_tol * abs(np.nanmin(E)))
    best = int(near[np.argmin(distance[near])])
    Mb, cb = M[best], c[best]
    if refine_pool is not None and refine_iters > 0:
        fJ2 = max(1, int(refine_pool))
        fI2 = max(1, int(round(fJ2 * dxJ * s_in / dxI)))
        xJr, Jr = pool_image(xJ, J, fJ2)
        xIr, Ir = pool_image(xI, I, fI2)
        Mb, cb = _refine_3d(xIr, Ir, xJr, Jr, Mb, cb, cJ, rJ, refine_iters, lr * 0.5, sigmaM, w_bg, bias)
    # the target -> source affine B: columns 1, 2 are the plane, column 0 its unit normal times the
    # length of the initial normal column (on the same side)
    n = np.cross(Mb[:, 0], Mb[:, 1])
    n = n / np.linalg.norm(n) * np.linalg.norm(B0[:3, 0])
    if np.dot(B0[:3, 0], n) < 0:
        n = -n
    B = np.eye(4)
    B[:3, 0] = n
    B[:3, 1:3] = Mb
    B[:3, 3] = cb - Mb @ cJ
    A = np.linalg.inv(B)
    return dict(L=A[:3, :3].copy(), T=A[:3, 3].copy(), energy=E, starts=[tuple(map(float, s)) for s in starts],
                best=best, pool=(fI, fJ))


# =========================================================================== entry point
def search_pool(xJ):
    '''Pooling factor of the coarse search: about SEARCH_PIXELS pixels along the longer
    target axis.'''
    n = max(len(x) for x in xJ[-2:])
    return max(1, int(round(n / float(SEARCH_PIXELS))))


def has_tissue(problem):
    '''Whether the target of a prepared LDDMM problem has any tissue at the resolution of the
    coarse search: the tissue signal (tissue_weights) of the target pooled by search_pool is
    nonzero somewhere.  A constant target has none, and the search could not place it (its
    tissue centroid is undefined).'''
    J = problem.J[:, 0] if problem.dim == 3 else problem.J
    xJ = problem.xJ[-2:]
    _, Jc = pool_image(xJ, J, search_pool(xJ))
    return bool(np.any(tissue_weights(Jc) > 0))


def affine_prefit(problem):
    '''Run the pre-alignment for a prepared LDDMM problem from its initial L and T.

    2D landmarks, if present, enter the energy and the search keeps the initial orientation
    (a single start).  The settings are this module's constants, read at call time.  Returns
    (L, T, info); info describes the search for the result dict.
    '''
    t0 = time.perf_counter()
    f = search_pool(problem.xJ)
    common = dict(pool=f, refine_pool=max(1, f // 2), lr=LEARNING_RATE, w_bg=BACKGROUND_WEIGHT, bias=BIAS_ORDER,
                  tie_tol=TIE_TOLERANCE, sigmaM=problem.sigmaM)
    if problem.dim == 2:
        points = len(problem.pointsI) > 0
        r = search_2d(problem.xI, problem.I, problem.xJ, problem.J, problem.L, problem.T,
                      n_angles=1 if points else ANGLES_2D, iters=ITERS_2D, refine_iters=REFINE_ITERS_2D,
                      pointsI=problem.pointsI if points else None, pointsJ=problem.pointsJ if points else None,
                      sigmaP=problem.sigmaP, **common)
    else:
        r = search_3d(problem.xI, problem.I, problem.xJ, problem.J, problem.L, problem.T, tilts=TILTS_3D,
                      offsets=OFFSETS_3D, iters=ITERS_3D, refine_iters=REFINE_ITERS_3D, **common)
    L, T = r['L'], r['T']
    if not (np.all(np.isfinite(L)) and np.all(np.isfinite(T)) and abs(np.linalg.det(L)) > 0):
        raise FloatingPointError('the pre-fit gave a non-finite or singular affine')
    d = problem.dim
    A_start = np.eye(d + 1)
    A_start[:d, :d], A_start[:d, d] = problem.L, problem.T
    A = np.eye(d + 1)
    A[:d, :d], A[:d, d] = L, T
    info = dict(ran=True, A=A, A_start=A_start, energy=r['energy'], starts=r['starts'], best=r['best'],
                pool=r['pool'], seconds=time.perf_counter() - t0)
    return L, T, info


# =========================================================================== scales
def background_level(Jc):
    '''Most common intensity of one channel (the mode of a 256-bin histogram).'''
    lo, hi = float(Jc.min()), float(Jc.max())
    if hi <= lo:
        return lo
    h, e = np.histogram(Jc, bins=256, range=(lo, hi))
    k = int(np.argmax(h))
    return 0.5 * (e[k] + e[k + 1]) if k > 0 else lo


def intensity_scale(J):
    '''Intensity scale of an image: the rms over channels of sum (J_c - b_c)^2 / sum |J_c - b_c|,
    b_c the background level.  For a count raster (background 0) it is the intensity-weighted
    mean intensity, i.e. the typical tissue intensity.  1 for a constant image.'''
    J = np.asarray(J, np.float64)
    s2 = []
    for Jc in J.reshape(J.shape[0], -1):
        d = Jc - background_level(Jc)
        a = np.abs(d).sum()
        s2.append((np.sum(d * d) / a) ** 2 if a > 0 else 0.0)
    s = float(np.sqrt(np.mean(s2)))
    return s if s > 0 else 1.0


def source_pixel_size(xJ, L, dim):
    '''Target pixel size in source units under the linear map L (target = L source + T):
    sqrt(dx_row dx_col / |det L|) in 2D and sqrt(dx_row dx_col |L^-1 e_row x L^-1 e_col|) in 3D.'''
    area = abs(float(xJ[-2][1] - xJ[-2][0])) * abs(float(xJ[-1][1] - xJ[-1][0]))
    L = np.asarray(L, np.float64)
    if dim == 2:
        return math.sqrt(area / abs(np.linalg.det(L)))
    Li = np.linalg.inv(L)
    return math.sqrt(area * float(np.linalg.norm(np.cross(Li[:, 1], Li[:, 2]))))


def physical_sigmaR(problem, sigma_disp):
    '''sigmaR for a prior displacement scale sigma_disp (in source coordinate units, um).

        sigmaR = sigma_disp sqrt(C_d pi) a^(d/2) sqrt(nt) (dx / 30) sigmaM / s_J

    sqrt(C_d pi) a^(d/2) sqrt(nt) makes sigma_disp the pointwise standard deviation of the
    prior's displacement (C_2 = 12, C_3 = 64, the constants of the default p = 2), and
    (dx / 30) sigmaM / s_J keeps the balance between the image term and the prior independent
    of the target pixel size dx (in source units, under the initial affine) and of the target
    intensity scale s_J.  At the default a = 500, nt = 3 and sigma_disp = 94.03, a 30 um target
    with sigmaM = s_J gets the previous default 5e5.
    '''
    d = problem.dim
    dx = source_pixel_size(problem.xJ, problem.L, d)
    s_J = intensity_scale(problem.J)
    return (float(sigma_disp) * math.sqrt(PRIOR_CONSTANT[d] * math.pi) * problem.a ** (d / 2.0)
            * math.sqrt(problem.nt) * (dx / PIXEL_SIZE_REF) * problem.sigmaM / s_J)


def _gradient_energy(J, xJ):
    '''sum over channels and pixels of |grad J|^2 (per coordinate unit), J (C, H, W).'''
    spacing = [float(x[1] - x[0]) for x in xJ[-2:]]
    G = 0.0
    for Jc in J:
        gy, gx = np.gradient(Jc, *spacing)
        G += float(np.sum(gy * gy + gx * gx))
    return G


def curvature_proxy(xI, I, xJ, J, L, T):
    '''Curvature scale of LDDMM's 2D image matching term at the affine (L, T):

        G = (sum_c var J_c) * mean_k sum_j |grad AI_kj|^2 / var AI_k

    AI = the source under the affine, sampled on the target grid (bilinear, border padding).
    Source channels without variance (in AI, or constant in I) are skipped; without any,
    G = sum_j |grad J_j|^2.  Constant channels of J add no variance (np.var can round to a
    tiny positive number there).
    '''
    J = np.asarray(J, np.float64)
    H, W = J.shape[-2:]
    Y = torch.stack(torch.meshgrid(*[torch.as_tensor(np.asarray(x, np.float64)) for x in xJ], indexing='ij'), -1)
    A = np.eye(3)
    A[:2, :2], A[:2, 2] = L, T
    Ai = torch.as_tensor(np.linalg.inv(A))
    X = (Ai[:2, :2] @ Y[..., None])[..., 0] + Ai[:2, 2]
    lo, span = _grid_norm(xI)
    I = np.asarray(I, np.float64)
    AI = _sample(torch.as_tensor(I)[None], lo, span, X[None])[0].numpy()
    vJ = float(sum(np.var(Jc) for Jc in J.reshape(J.shape[0], -1) if np.ptp(Jc) > 0))
    g = [_gradient_energy(AIk[None], xJ) / np.var(AIk) for AIk, Ik in zip(AI, I)
         if np.var(AIk) > 0 and np.ptp(Ik) > 0]
    return vJ * float(np.mean(g)) if g else _gradient_energy(J, xJ)


def curvature_steps(problem, steps):
    '''Step sizes (epL, epT, epV) scaled to the curvature of the image term (2D).

    Every step in `steps` is multiplied by sigmaM^2 * 1.1 / G, G = curvature_proxy at the
    problem's initial affine: steps that are stable for the curvature of the data and the
    weight 1/sigmaM^2 of the image term, equal to the given ones at G = 1.1 sigmaM^2.  Without
    any curvature (a constant target) the steps are returned unchanged.
    '''
    G = curvature_proxy(problem.xI, problem.I, problem.xJ, problem.J, problem.L, problem.T)
    f = problem.sigmaM ** 2 * CURVATURE_REF / G if np.isfinite(G) and G > 0 else 1.0
    return {k: float(val) * f for k, val in steps.items()}


# =========================================================================== stability (2D)
_CHUNK = 1 << 18  # target pixels per chunk of the curvature passes (bounds their memory)


def _target_points(problem, start, stop):
    '''Sample points X = A^-1 y (source coordinates, (n, 2)) of the target pixels start:stop
    (row-major) under the problem's affine.'''
    xJ = problem.xJ
    k = np.arange(start, stop)
    Y = np.stack((xJ[0][k // len(xJ[1])], xJ[1][k % len(xJ[1])]), 1)
    return (Y - problem.T) @ np.linalg.inv(problem.L).T


def _source_at(problem, X):
    '''The source and its gradient at points X (n, 2): AI (C_I, n) and grad (C_I, n, 2), with
    LDDMM's interpolation of the source (bilinear, border padding: along an axis on which a
    point lies outside the source the gradient is 0).'''
    I = problem.I
    idx = []
    for d, x in enumerate(problem.xI):
        n, h = len(x), float(x[1] - x[0]) if len(x) > 1 else 1.0
        u = (X[:, d] - float(x[0])) / h
        outside = (u < 0) | (u > n - 1)
        u = np.clip(u, 0, n - 1)
        i = np.clip(np.floor(u).astype(np.int64), 0, max(n - 2, 0))
        idx.append((i, np.minimum(i + 1, n - 1), u - i, outside, h))
    (i0, j0, f0, o0, h0), (i1, j1, f1, o1, h1) = idx
    AI = np.empty((I.shape[0], len(X)))
    grad = np.empty((I.shape[0], len(X), 2))
    for k, Ik in enumerate(I):
        a, b, c, d = Ik[i0, i1], Ik[i0, j1], Ik[j0, i1], Ik[j0, j1]
        AI[k] = (1 - f0) * ((1 - f1) * a + f1 * b) + f0 * ((1 - f1) * c + f1 * d)
        grad[k, :, 0] = np.where(o0, 0.0, ((1 - f1) * (c - a) + f1 * (d - b)) / h0)
        grad[k, :, 1] = np.where(o1, 0.0, ((1 - f0) * (b - a) + f0 * (d - c)) / h1)
    return AI, grad


def _cells(xv, P):
    '''Bilinear cell (i0, i1), fractions (f0, f1) of points P (n, 2) on the velocity grid xv and
    whether they lie in it (LDDMM samples v with zero padding; points outside are dropped).'''
    out = []
    inside = np.ones(len(P), bool)
    for d, x in enumerate(xv):
        n, h = len(x), float(x[1] - x[0])
        u = (P[:, d] - float(x[0])) / h
        inside &= (u >= 0) & (u <= n - 1)
        i = np.clip(np.floor(u).astype(np.int64), 0, n - 2)
        out += [i, np.clip(u - i, 0.0, 1.0)]
    return out, inside


# corners of a velocity grid cell and the (a, b) pairs a <= b of corners
_CORNERS = ((0, 0), (0, 1), (1, 0), (1, 1))
_PAIRS = [(a, b) for a in range(4) for b in range(a, 4)]


class _Curvature:
    '''Gauss-Newton curvature of LDDMM's 2D energy at v = 0 and the problem's affine.

    The image term is taken with matching weights WM = 1 (their upper bound) and the contrast
    of a perfectly correlated source, |c_k|^2 = sum_c var J_c / var AI_k for source channel k
    (averaged over the source channels), the assumptions of curvature_proxy.  A displacement
    u_j of target pixel j's sample point X_j (source coordinates) then changes the image term
    by u_j^T S_j u_j / 2 with

        S_j = (sum_c var J_c) / sigmaM^2 * mean_k g_kj g_kj^T / var AI_k,   g_kj = grad I_k(X_j),

    and a displacement u_i of landmark i (source coordinates, mapped by L) changes the landmark
    term by u_i^T S_i u_i / 2 with S_i = L^T L / sigmaP^2.  Both are accumulated over the
    bilinear velocity grid cells (moments of the corner weights) for velocity_curvature, and
    into the 6x6 Hessian of (L, T) for affine_hessian.
    '''

    def __init__(self, problem, velocity=True, affine=True):
        P = problem
        N = int(np.prod(P.J.shape[1:]))
        J = P.J.reshape(P.J.shape[0], -1)
        vJ = float(sum(np.var(Jc) for Jc in J if np.ptp(Jc) > 0))
        self.problem = P
        Li = np.linalg.inv(P.L)
        xv = P.xv
        lm = len(P.pointsI) > 0
        # pass 1: variance of the source channels on the sample points, and the box of
        # velocity grid cells that holds the sample points and landmarks
        CI = P.I.shape[0]
        cnt, mean, m2 = 0, np.zeros(CI), np.zeros(CI)
        lo, hi = np.array([len(xv[0]), len(xv[1])]), np.array([-1, -1])

        def grow(X):
            (i0, _, i1, _), inside = _cells(xv, X)
            if inside.any():
                lo[:] = np.minimum(lo, [i0[inside].min(), i1[inside].min()])
                hi[:] = np.maximum(hi, [i0[inside].max(), i1[inside].max()])
        for s in range(0, N, _CHUNK):
            X = _target_points(P, s, min(N, s + _CHUNK))
            AI, _ = _source_at(P, X)
            n, mu = AI.shape[1], AI.mean(1)
            m2 += ((AI - mu[:, None]) ** 2).sum(1) + (mu - mean) ** 2 * cnt * n / (cnt + n)
            mean += (mu - mean) * n / (cnt + n)
            cnt += n
            if velocity:
                grow(X)
        if velocity and lm:
            grow(P.pointsI)
        var = m2 / max(cnt, 1)
        use = [k for k in range(CI) if var[k] > 0 and np.ptp(P.I[k]) > 0]
        w = vJ / P.sigmaM ** 2 / len(use) if use and vJ > 0 else 0.0
        self.M, self.box = None, None
        if velocity and np.all(hi >= lo):
            self.box = (int(lo[0]), int(lo[1]), int(hi[0] - lo[0] + 1), int(hi[1] - lo[1] + 1))
            self.M = np.zeros((len(_PAIRS), 3, self.box[2] * self.box[3]))
        self.H = np.zeros((6, 6)) if affine else None
        chunks = list(range(0, N, _CHUNK)) if w > 0 else []
        # pass 2: the tensors S, accumulated
        for s in chunks:
            X = _target_points(P, s, min(N, s + _CHUNK))
            _, grad = _source_at(P, X)
            S = np.zeros((len(X), 3))  # S00, S01, S11
            for k in use:
                g = grad[k]
                S += np.stack((g[:, 0] * g[:, 0], g[:, 0] * g[:, 1], g[:, 1] * g[:, 1]), 1) / var[k]
            S *= w
            self._add(X, S)
            if affine:
                # a change dTheta of Theta = [L | T] moves X_j by -L^-1 dTheta Z_j, Z_j = (X_j, 1):
                # the Hessian (row-major Theta) is sum_j U_j (x) Z_j Z_j^T with U_j = L^-T S_j L^-1
                (p, q), (r, t) = Li
                U = np.stack((p * p * S[:, 0] + 2 * p * r * S[:, 1] + r * r * S[:, 2],
                              p * q * S[:, 0] + (p * t + r * q) * S[:, 1] + r * t * S[:, 2],
                              q * q * S[:, 0] + 2 * q * t * S[:, 1] + t * t * S[:, 2]), 1)
                ZZ = np.stack((X[:, 0] * X[:, 0], X[:, 0] * X[:, 1], X[:, 0], X[:, 1] * X[:, 1], X[:, 1],
                               np.ones(len(X))), 1)
                self._add_affine(U.T @ ZZ)
        if lm:
            LtL = P.L.T @ P.L / P.sigmaP ** 2
            self._add(P.pointsI, np.tile([LtL[0, 0], LtL[0, 1], LtL[1, 1]], (len(P.pointsI), 1)))
            if affine:
                Z = np.concatenate((P.pointsI, np.ones((len(P.pointsI), 1))), 1)
                self.H += np.kron(np.eye(2), Z.T @ Z) / P.sigmaP ** 2
        if affine:
            order = [0, 1, 3, 4, 2, 5]  # (L00, L01, T0, L10, L11, T1) -> (L00, L01, L10, L11, T0, T1)
            self.H = self.H[np.ix_(order, order)]

    def _add_affine(self, R):
        '''Add sum_j U_j (x) Z_j Z_j^T to H from R = sum_j u_j z_j^T (3 x 6), u_j the unique
        entries (00, 01, 11) of U_j and z_j those (00, 01, 02, 11, 12, 22) of Z_j Z_j^T.'''
        sym2 = ((0, 1), (1, 2))
        sym3 = ((0, 1, 2), (1, 3, 4), (2, 4, 5))
        for a in range(2):
            for b in range(2):
                for s in range(3):
                    for t in range(3):
                        self.H[3 * a + s, 3 * b + t] += R[sym2[a][b], sym3[s][t]]

    def _add(self, X, S):
        '''Add the tensors S (n, 3) at points X to the cell moments.'''
        if self.M is None:
            return
        (i0, f0, i1, f1), inside = _cells(self.problem.xv, X)
        b0, b1, n0, n1 = self.box
        cell = (i0[inside] - b0) * n1 + (i1[inside] - b1)
        f0, f1, S = f0[inside], f1[inside], S[inside]
        w = ((1 - f0) * (1 - f1), (1 - f0) * f1, f0 * (1 - f1), f0 * f1)
        for p, (a, b) in enumerate(_PAIRS):
            ww = w[a] * w[b]
            for c in range(3):
                self.M[p, c] += np.bincount(cell, weights=ww * S[:, c], minlength=n0 * n1)

    def apply(self, u):
        '''H u for a field u (n0, n1, 2) on the velocity grid, or for each field of a block
        (n0, n1, 2, B): the Hessian of the energy in the nodes' displacement.'''
        out = np.zeros_like(u)
        b0, b1, n0, n1 = self.box
        M = self.M.reshape((len(_PAIRS), 3, n0, n1) + (1,) * (u.ndim - 3))
        uc = [u[b0 + da:b0 + da + n0, b1 + db:b1 + db + n1] for da, db in _CORNERS]
        acc = [np.zeros((n0, n1) + u.shape[2:]) for _ in _CORNERS]
        for p, (a, b) in enumerate(_PAIRS):
            for x, y in ((a, b), (b, a)) if a != b else ((a, b),):
                acc[x][:, :, 0] += M[p, 0] * uc[y][:, :, 0] + M[p, 1] * uc[y][:, :, 1]
                acc[x][:, :, 1] += M[p, 1] * uc[y][:, :, 0] + M[p, 2] * uc[y][:, :, 1]
        for (da, db), t in zip(_CORNERS, acc):
            out[b0 + da:b0 + da + n0, b1 + db:b1 + db + n1] += t
        return out


def velocity_curvature(problem, curvature=None):
    '''lambda_v: the largest eigenvalue of K H_v (2D), H_v the Hessian of the image and landmark
    terms in v at v = 0 (as in _Curvature) and K the kernel that smooths the gradient.

    At v = 0 a change dv moves every sample point by -sum_t dv_t / nt, so the largest
    eigenvalue over v is that over a field u shared by the nt time steps divided by nt.  It is
    that of the symmetric K^1/2 H K^1/2, found by subspace iteration with Rayleigh-Ritz on a
    block of EIG_BLOCK vectors from a fixed pseudo-random start.  (A single power iteration can
    settle on the second eigenvalue for many steps when the first is close to it and the start
    has little of its eigenvector.)  The velocity step v <- v - epV (K dE/dv) is stable for
    epV (lambda_v + DV / sigmaR^2) < 2, DV / sigmaR^2 being the regularizer's contribution.
    Returns 0 without image curvature or landmarks in the velocity grid.'''
    c = curvature if curvature is not None else _Curvature(problem, affine=False)
    if c.M is None or not np.any(c.M):
        return 0.0
    n = 2 * problem.K.size
    B = min(EIG_BLOCK, n)
    shape = problem.K.shape + (2, B)
    root = np.sqrt(problem.K)[..., None, None]

    def op(Y):  # K^1/2 H K^1/2 Y for a block Y (n, B)
        u = np.fft.ifft2(np.fft.fft2(Y.reshape(shape), axes=(0, 1)) * root, axes=(0, 1)).real
        return np.fft.ifft2(np.fft.fft2(c.apply(u), axes=(0, 1)) * root, axes=(0, 1)).real.reshape(n, B)

    Y, _ = np.linalg.qr(np.random.default_rng(0).standard_normal((n, B)))
    lam = 0.0
    for _ in range(EIG_ITERATIONS):
        Z = op(Y)
        new = float(np.linalg.eigvalsh(0.5 * (Y.T @ Z + Z.T @ Y))[-1])
        done = abs(new - lam) <= EIG_TOL * abs(new)
        lam = new
        if done or not lam > 0:
            break
        Y, _ = np.linalg.qr(Z)
    return max(lam, 0.0) / problem.nt


def affine_hessian(problem, curvature=None):
    '''The 6x6 Hessian of the image and landmark terms in (L00, L01, L10, L11, T0, T1) at
    v = 0 (as in _Curvature).'''
    c = curvature if curvature is not None else _Curvature(problem, velocity=False)
    return c.H


def stable_steps(problem, steps):
    '''Cap default steps where gradient descent on the energy would be unstable.

    `steps` holds the steps to cap (the ones the caller did not give).  epV is lowered to
    STEP_LIMIT_V / (lambda_v + DV / sigmaR^2) if it is larger (velocity_curvature).  The
    default ones of epL and epT are multiplied by a common factor so that the largest
    eigenvalue of D H D <= STEP_LIMIT_AFFINE, with H the affine Hessian (affine_hessian)
    restricted to their entries and D^2 their steps divided by 10 (the engines divide the
    affine steps by 10 while the diffeomorphism is on), or by 1 if diffeo_start > 0.

    The curvature is that of a perfectly correlated source with WM = 1, so the caps hold for
    any matching weights; a step that is already small enough is returned unchanged.  The
    curvature of the image term is intensive in v (it does not grow with the tissue area) but
    the steps of curvature_steps are not, so the cap matters for small images, and for
    landmarks, small sigmaR, and coordinates far from the origin (the affine).

    In 3D (fixed default steps, whose stability for the image term is as in version 1.0) only
    the regularizer's part is used: epV <= STEP_LIMIT_V / (DV / sigmaR^2), which matters only
    for a very small sigma_disp with sigmaM well below the target's intensity scale.
    '''
    steps = dict(steps)
    if problem.dim == 3:
        reg = problem.DV / problem.sigmaR ** 2
        if 'epV' in steps and steps['epV'] * reg > STEP_LIMIT_V:
            steps['epV'] = STEP_LIMIT_V / reg
        return steps
    affine = [k for k in ('epL', 'epT') if k in steps]
    if 'epV' not in steps and not affine:
        return steps
    curv = _Curvature(problem, velocity='epV' in steps, affine=bool(affine))
    if 'epV' in steps:
        lam = velocity_curvature(problem, curv) + problem.DV / problem.sigmaR ** 2
        if lam > 0 and steps['epV'] * lam > STEP_LIMIT_V:
            steps['epV'] = STEP_LIMIT_V / lam
    if affine:
        idx = ([0, 1, 2, 3] if 'epL' in steps else []) + ([4, 5] if 'epT' in steps else [])
        fac = 10.0 if problem.diffeo_start <= 0 else 1.0
        d = np.sqrt(np.array([steps['epL'] if i < 4 else steps['epT'] for i in idx]) / fac)
        H = curv.H[np.ix_(idx, idx)]
        lam = float(np.linalg.eigvalsh(d[:, None] * H * d[None, :])[-1])
        if lam > STEP_LIMIT_AFFINE:
            for k in affine:
                steps[k] *= STEP_LIMIT_AFFINE / lam
    return steps
