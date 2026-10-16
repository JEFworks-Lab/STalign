'''Pure PyTorch engine for LDDMM and LDDMM_3D_to_slice.

This engine runs on any torch device and dtype.  It is the fallback when the compiled engine
is missing or declines a problem.  It computes the same iteration as the original STalign
code, with the corrections listed in CHANGES.md and the optional affine step control
(``_lddmm_common.Problem``), and agrees with the reference implementation in
tests/reference.py to rounding.

How it differs from the original (all equal up to rounding):

* No autograd.  The gradient is chained by hand through the semi-Lagrangian flow.  The
  forward pass uses ``torch.grid_sampler``, and the backward pass calls
  ``aten.grid_sampler_{2,3}d_backward`` directly: grid gradient only for the image, grid and
  input gradients for the velocity steps.  The affine gradient is the closed form
  dE/dA = -Ai^T dE/dAi Ai^T.
* The flow is integrated in grid_sample's normalised coordinates with a pre-scaled velocity,
  in preallocated "planar" (component, batch, point) buffers.  On macOS every fresh multi-MB
  allocation page-faults, which costs more than the arithmetic.
* CPU parallelism.  torch's CPU grid_sample kernels parallelise over the batch dimension
  only, so the target points are split into a fixed number of chunks that form the batch.
  The chunk count does not depend on the thread count, so results do not either.
* Crop (CPU).  The velocity steps sample, and splat their gradient into, only the bounding
  box of the sample points.  The box is bounded analytically from the affine image of the
  target box grown by sum_t max|v_t| / nt.  Corners outside the velocity grid contribute zero
  either way, so cropping is exact.  For a 3D atlas and a 2D slice the box is a thin slab.
* Smoothing.  With the regularizer 0.5/sigmaR^2 sum_t |L v_t|^2, the smoothed regularizer
  gradient K * dER/dv is exactly DV/sigmaR^2 * v, so only the matching gradient is
  transformed: v <- v - epV (K * dEM/dv + DV/sigmaR^2 v).  The spectrum of v follows the
  same linear recurrence, so the reported ER needs no transform either; it is recomputed
  from v every 50 updates.  2D grids up to 12k cells use dense real DFT matrices, because
  their sides are often prime.  Larger 2D grids and 3D grids use torch.fft per volume on a
  small thread pool, because pocketfft is single threaded.  In 3D the gradient lives on the
  crop slab, so its forward transform is three small dense DFTs.  Only increasing ``dim``
  orders are used: MPS computes rfftn/irfftn with a permuted dim order wrongly.
* With user-supplied muA / muB the exp() terms of WA / WB are constant and computed once (and,
  with the mixture EM, the squared distances to a fixed class mean).
* Exact landmark map: the fixed-point sweeps run on all landmarks at once and freeze each one
  as it converges (as the C++ engine solves them one by one).  The Jacobians for the Newton
  step and the implicit-function gradient come from one grid_sampler backward pass with a
  batch of two unit output gradients.
'''
import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

_BWD = {2: torch.ops.aten.grid_sampler_2d_backward, 3: torch.ops.aten.grid_sampler_3d_backward}
_BILINEAR = 0
_ZEROS = 0
_BORDER = 1

# Number of point chunks used as grid_sample's batch dimension on CPU (see module docstring).
_CHUNKS = 16
# 2D velocity grids up to this many cells are transformed with dense DFT matrices.
_DENSE_DFT_MAX_CELLS = 12000
# The tracked spectrum of v (used only for the reported ER) is recomputed this often.
_RESYNC = 50
# Exact landmark map: a fixed-point solve stops when no landmark moves by more than this
# fraction of a velocity grid cell, after at most _LANDMARK_MAX_ITER sweeps (as the C++ engine).
_LANDMARK_TOL = 1e-10
_LANDMARK_MAX_ITER = 100
# Mixture EM: floor of the estimated matching variance.
_VARIANCE_FLOOR = 1e-12


# =========================================================================== threads
def choose_threads(npoints, dim, max_threads):
    '''Threads for a problem with ``npoints`` target pixels, at most ``max_threads``.

    grid_sample over the target pixels dominates each iteration.  Small problems are bound by
    the latency of many small ops, where extra threads only add overhead.  The steps were
    measured on an Apple M1 Ultra.
    '''
    work = npoints * (1 if dim == 2 else 3)
    if work < 10_000:
        n = 1
    elif work < 40_000:
        n = 4
    elif work < 200_000:
        n = 6
    elif work < 1_000_000:
        n = 8
    else:
        n = 16
    return max(1, min(n, max_threads))


class _TorchThreads:
    '''Set torch's intra-op thread count inside a with block and restore it afterwards.'''

    def __init__(self, n):
        self.n = n

    def __enter__(self):
        self.old = torch.get_num_threads()
        if self.n:
            torch.set_num_threads(self.n)

    def __exit__(self, *exc):
        torch.set_num_threads(self.old)


# FFT worker pools, by size.  A forked child inherits the dict but not the worker threads, so
# the pools are dropped whenever the process id changes.
_POOLS = {}
_POOLS_PID = os.getpid()
_POOL_LOCK = threading.Lock()


def _reset_pools_in_child():
    global _POOL_LOCK
    _POOL_LOCK = threading.Lock()  # the parent's lock may have been held while forking
    _POOLS.clear()


if hasattr(os, 'register_at_fork'):
    os.register_at_fork(after_in_child=_reset_pools_in_child)


def _single_thread_worker():
    # pool workers run serial torch ops (pocketfft is serial anyway); no nested thread teams
    torch.set_num_threads(1)


def _get_pool(n):
    global _POOLS_PID
    with _POOL_LOCK:
        if _POOLS_PID != os.getpid():
            _POOLS.clear()
            _POOLS_PID = os.getpid()
        pool = _POOLS.get(n)
        if pool is None:
            keep = torch.get_num_threads()
            pool = ThreadPoolExecutor(n, thread_name_prefix='stalign_fft', initializer=_single_thread_worker)
            list(pool.map(lambda _: None, range(4 * n)))  # start the workers now
            torch.set_num_threads(keep)  # a worker's set_num_threads may change the global default
            _POOLS[n] = pool
        return pool


def _pool_map(f, x, nthreads):
    '''f applied to chunks of x along the leading axis, in parallel on the FFT pool.'''
    P = x.shape[0]
    nch = min(P, nthreads)
    if nch <= 1:
        return f(x)
    bounds = [(P * i) // nch for i in range(nch + 1)]
    parts = _get_pool(nthreads).map(lambda i: f(x[bounds[i]:bounds[i + 1]]), range(nch))
    return torch.cat(list(parts), 0)


# =========================================================================== small helpers
def _to_A(L, T):
    d = L.shape[0]
    O = torch.zeros((1, d + 1), device=L.device, dtype=L.dtype)
    O[0, d] = 1.0
    return torch.cat((torch.cat((L, T[:, None]), 1), O))


def _col_weights(n, nh, device, dtype):
    '''Multiplicity of each column of a real FFT's half spectrum in the full spectrum.'''
    w = torch.full((nh,), 2.0, device=device, dtype=dtype)
    w[0] = 1.0
    if n % 2 == 0:
        w[-1] = 1.0
    return w


def _dft_mats(n, device, dtype):
    '''Dense DFT matrices for an axis of length n (twiddles from exact integer phases).

    Returns (Fr, Gr, F): x @ Fr == view_as_real(rfft(x)) with (re, im) interleaved, Fr (n, 2h);
    view_as_real(Z) @ Gr == irfft(Z, n), Gr (2h, n); F @ x == fft(x), F (n, n) complex.
    '''
    h = n // 2 + 1
    k = np.arange(n)
    ph = 2.0 * np.pi * ((k[:, None] * k[None, :]) % n) / n
    c, s = np.cos(ph), np.sin(ph)
    Fr = np.empty((n, 2 * h))
    Fr[:, 0::2] = c[:, :h]
    Fr[:, 1::2] = -s[:, :h]
    w = np.full(h, 2.0)
    w[0] = 1.0
    if n % 2 == 0:
        w[-1] = 1.0
    Gr = np.empty((2 * h, n))
    Gr[0::2] = w[:, None] * c[:h] / n
    Gr[1::2] = -w[:, None] * s[:h] / n
    cdt = torch.complex128 if dtype == torch.float64 else torch.complex64
    t = lambda a, d: torch.as_tensor(a).to(d).to(device)
    return t(Fr, dtype), t(Gr, dtype), t(c - 1j * s, cdt)


def _gview(g, D):
    '''Planar (D, B, M) point buffer -> grid_sample grid view (B, 1[, 1], M, D).'''
    B, M = g.shape[1], g.shape[2]
    return g.view((D, B) + (1,) * (D - 1) + (M,)).permute((1,) + tuple(range(2, D + 1)) + (D + 1, 0))


def _cview(x, D):
    '''(B, C, M) -> (B, C, 1[, 1], M), the layout of grid_sample's input and output.'''
    B, C, M = x.shape
    return x.view((B, C) + (1,) * (D - 1) + (M,))


# =========================================================================== forward model + gradient
class _Model:
    '''Forward model and hand-written gradient of the matching energy (2D or 3D).

    ``vi`` is the velocity in the internal layout (nt, D, n_0, .., n_{D-1}), with components
    in grid_sample order: component k is the displacement along spatial axis D-1-k.  Point
    buffers are planar (D, B, M) and hold grid_sample's normalised coordinates relative to
    the current crop box.  Points are padded to B*M; the padding has zero weight.
    '''

    def __init__(self, xI, I, xJ, J, xv, nt, sigmaM, chunks, crop):
        D = len(xv)
        dev, dt = J.device, J.dtype
        self.D, self.nt, self.dev, self.dt = D, nt, dev, dt
        self.CI, self.CJ = I.shape[0], J.shape[0]
        self.nJ = tuple(J.shape[1:])
        N = int(np.prod(self.nJ))
        B = max(1, min(int(chunks), N))
        M = -(-N // B)
        self.N, self.B, self.M, self.Np = N, B, M, B * M
        self.sigmaM = sigmaM
        self.nv = [len(x) for x in xv]
        self.xv0 = [float(x[0]) for x in xv]
        self.xvs = [float(x[-1] - x[0]) for x in xv]
        self.xI0 = [float(x[0]) for x in xI]
        self.xIs = [float(x[-1] - x[0]) for x in xI]
        self.xI, self.xJ = xI, xJ
        self.XJ = None  # target meshgrid, only kept while v == 0 (see _exact_image_grid)
        # target box (centre, half width) per axis, for the crop bound
        self.cJ = [0.5 * (float(x.min()) + float(x.max())) for x in xJ]
        self.hJ = [0.5 * (float(x.max()) - float(x.min())) for x in xJ]
        self.crop = bool(crop)
        self.xv0_g = torch.tensor(self.xv0[::-1], device=dev, dtype=dt)
        CI, CJ, Np = self.CI, self.CJ, self.Np
        self.gbuf = torch.zeros((nt + 1, D, B, M), device=dev, dtype=dt)
        self.gI = torch.zeros((D, B, M), device=dev, dtype=dt)  # the padded tail must stay finite
        self.G = torch.empty((B, D, M), device=dev, dtype=dt)
        self.Gpl = torch.empty((D, Np), device=dev, dtype=dt)
        # stacked per-point rows [AI (CI) | J (CJ) | 1] for the contrast normal equations
        self.k1 = 1 + CI
        self.Cbuf = torch.zeros((B, 1 + CI + CJ, M), device=dev, dtype=dt)
        self.Cbuf[:, -1] = 1.0
        Jp = torch.zeros((CJ, Np), device=dev, dtype=dt)
        Jp[:, :N] = J.reshape(CJ, N)
        self.Cbuf[:, CI:CI + CJ] = Jp.view(CJ, B, M).transpose(0, 1)
        self.Jflat = J.reshape(CJ, N)
        self.WAbuf = torch.empty((B, CI, M), device=dev, dtype=dt)
        self.negI = -torch.eye(CJ, device=dev, dtype=dt)
        self.rbuf = torch.empty((B, CJ, M), device=dev, dtype=dt)
        self.rWbuf = torch.empty((B, CJ, M), device=dev, dtype=dt)
        self.gobuf = torch.empty((B, CI, M), device=dev, dtype=dt)
        self.Wp = torch.zeros(Np, device=dev, dtype=dt)  # matching weights, zero padded
        self.Wb = self.Wp.view(B, 1, M)
        self.WM = self.Wp[:N].view(self.nJ)  # view shaped like J[0]
        self.Iexp = I[None].expand((B,) + tuple(I.shape))
        self.eye = torch.eye(self.k1, device=dev, dtype=dt)
        self.vgf = torch.zeros((nt, D) + tuple(self.nv), device=dev, dtype=dt)
        # affine gradient reduction over the last target axis: [1, x_last]
        self.Xlast = torch.stack((torch.ones_like(xJ[-1]), xJ[-1]), 1)
        self.frame = None
        self.AIb = None

    def set_WM(self, WM):
        self.WM.copy_(WM)
        self.cW = self.WM.sum()
        self.cWJ = (self.Jflat * self.WM.reshape(1, -1)).sum(1)

    def _frame(self, Ai, vi):
        '''Crop box (per grid component k) and the map from full-grid to crop-box normalised
        coordinates, g_c = alpha * g + beta.'''
        D, nt = self.D, self.nt
        a_l, b_l = [], []
        if self.crop:
            Aih = Ai.detach().to('cpu').tolist()
            vmin, vmx = torch.aminmax(vi.reshape(nt, D, -1), dim=2)
            vmax = torch.maximum(vmx, -vmin).to('cpu').tolist()  # max |v_t| per component
        for k in range(D):
            d = D - 1 - k
            n = self.nv[d]
            a, b = 0, n - 1
            if self.crop:
                c = Aih[d][D] + sum(Aih[d][e] * self.cJ[e] for e in range(D))
                h = sum(abs(Aih[d][e]) * self.hJ[e] for e in range(D))
                fl = sum(vmax[t][k] for t in range(nt)) / nt
                sc = (n - 1) / self.xvs[d]
                lo = (c - h - fl - self.xv0[d]) * sc
                hi = (c + h + fl - self.xv0[d]) * sc
                if math.isfinite(lo) and math.isfinite(hi) and hi - lo < 4 * n:
                    a = min(max(0, math.floor(lo) - 1), n - 2)
                    b = max(min(n - 1, math.floor(hi) + 2), a + 1)
            a_l.append(a)
            b_l.append(b)
        alpha = [(self.nv[D - 1 - k] - 1) / (b_l[k] - a_l[k]) for k in range(D)]
        beta = [(self.nv[D - 1 - k] - 1 - 2 * a_l[k]) / (b_l[k] - a_l[k]) - 1.0 for k in range(D)]
        slices = tuple(slice(a_l[D - 1 - d], b_l[D - 1 - d] + 1) for d in range(D))  # axis order
        return dict(alpha=alpha, beta=beta, slices=slices)

    def _exact_image_grid(self, Ai):
        '''Image sample grid computed with the original's operation sequence (only while v == 0).

        Bilinear/trilinear interpolation has a one-sided derivative on grid lines.  With an
        exact initial affine (90 degree rotations, integer scales, commensurate pixel sizes)
        and v == 0, many target pixels land exactly on image grid lines.  Which side is taken
        then depends on the last bit of the coordinate, so it is computed as the original
        does: X = Ai[:D,:D] @ XJ + Ai[:D,D], then (X - x0) / (x[-1] - x0) * 2 - 1.  Once v or A
        moves such ties have measure zero.  The same tie can occur at velocity-grid nodes
        (a warm start with v != 0, an exact affine and xv on the target lattice); it is not
        replayed, since the original's result there also flips with the last bit of the input.
        '''
        D, N = self.D, self.N
        if self.XJ is None:
            self.XJ = torch.stack(torch.meshgrid(*self.xJ, indexing='ij'), -1)
        X = (Ai[:D, :D] @ self.XJ[..., None])[..., 0] + Ai[:D, D]
        gI = self.gI.view(D, self.Np)
        for d in range(D):
            ph = X[..., d] - self.xI[d][0]
            ph /= self.xI[d][-1] - self.xI[d][0]
            ph *= 2.0
            ph -= 1.0
            gI[D - 1 - d, :N] = ph.reshape(-1)

    def evaluate(self, Ai, vi, need_vgrad=True, v_is_zero=False):
        '''Forward pass and gradient at (A, v).

        Returns dict(Lgrad, Tgrad, EM, vgc, frame): vgc is dE/dv on the crop box (physical
        units; None unless need_vgrad) and frame['slices'] the crop box in axis order.  The
        sample points, AI and the residual stay in the buffers until the next call.'''
        D, nt, B, M, N, Np = self.D, self.nt, self.B, self.M, self.N, self.Np
        CI, CJ, k1 = self.CI, self.CJ, self.k1
        dev, dt = self.dev, self.dt
        fr = self.frame = self._frame(Ai, vi)
        al, be = fr['alpha'], fr['beta']
        # scale from physical displacement to crop-normalised coordinates (grid order)
        sc = [2.0 * al[k] / self.xvs[D - 1 - k] for k in range(D)]
        sc_t = torch.tensor(sc, device=dev, dtype=dt)
        al_t = torch.tensor(al, device=dev, dtype=dt)
        be_t = torch.tensor(be, device=dev, dtype=dt)
        sI = [self.xvs[D - 1 - k] / (al[k] * self.xIs[D - 1 - k]) for k in range(D)]
        cI = [-be[k] * sI[k] + (2.0 * (self.xv0[D - 1 - k] - self.xI0[D - 1 - k]) + self.xvs[D - 1 - k]) / self.xIs[D - 1 - k] - 1.0
              for k in range(D)]
        sI_t = torch.tensor(sI, device=dev, dtype=dt)
        cI_t = torch.tensor(cI, device=dev, dtype=dt)
        # prescaled, cropped velocity: g_t = g_{t+1} - interp(vs_t, g_{t+1})
        sv = (sc_t / nt).view((1, D) + (1,) * D)
        vs = vi[(slice(None), slice(None)) + fr['slices']] * sv
        # g[nt] = Mg x + cg over the target grid
        Aig = Ai[:D].flip(0)  # row k <-> axis D-1-k
        Mg = sc_t[:, None] * Aig[:, :D]
        cg = sc_t * (Aig[:, D] - self.xv0_g) - al_t + be_t
        gl = self.gbuf
        gt = gl[nt].view(D, Np)[:, :N].view((D,) + self.nJ)
        terms = []
        for e in range(D):
            if self.nJ[e] == 1:
                cg = cg + Mg[:, e] * self.xJ[e][0]
            else:
                shp = [D] + [1] * D
                shp[1 + e] = self.nJ[e]
                terms.append((Mg[:, e:e + 1] * self.xJ[e][None, :]).view(shp))
        cshape = (D,) + (1,) * D
        if len(terms) == 0:
            gt.copy_(cg.view(cshape).expand_as(gt))
        elif len(terms) == 1:
            torch.add(terms[0], cg.view(cshape), out=gt)
        else:
            torch.add(terms[0] + cg.view(cshape), terms[1], out=gt)
            for tm in terms[2:]:
                gt.add_(tm)
        # backward flow of the sample points
        vexp = [vs[t][None].expand((B,) + tuple(vs[t].shape)) for t in range(nt)]
        for t in range(nt - 1, -1, -1):
            out = torch.grid_sampler(vexp[t], _gview(gl[t + 1], D), _BILINEAR, _ZEROS, True)
            torch.sub(gl[t + 1], out.view(B, D, M).transpose(0, 1), out=gl[t])
        if v_is_zero:
            self._exact_image_grid(Ai)
        else:
            self.XJ = None
            torch.addcmul(cI_t.view(D, 1, 1), gl[0], sI_t.view(D, 1, 1), out=self.gI)
        AIb = self.AIb = torch.grid_sampler(self.Iexp, _gview(self.gI, D), _BILINEAR, _BORDER, True).view(B, CI, M)
        # contrast (no gradient through the coefficients): with B = [1; AI], solve
        # (B diag(W) B^T + 0.1 Id) c = B diag(W) J^T.  The W-only sums (sum W, sum W J) change
        # only with W and are cached by set_WM; the AI rows come from one bmm.
        Cb = self.Cbuf
        Cb[:, :CI].copy_(AIb)
        WA = torch.mul(AIb, self.Wb, out=self.WAbuf)
        rows = torch.bmm(WA, Cb.transpose(1, 2)).sum(0)  # (CI, k2): [W AI AI^T | W AI J^T | W AI]
        top = torch.cat((self.cW.view(1, 1), rows[:, -1].view(1, CI), self.cWJ.view(1, CJ)), 1)
        bot = torch.cat((rows[:, -1:], rows[:, :CI], rows[:, CI:CI + CJ]), 1)
        NE = torch.cat((top, bot), 0)  # (k1, k1 + CJ) = [BB | BJ]
        coeffs = torch.linalg.solve(NE[:, :k1] + 0.1 * self.eye, NE[:, k1:])  # (k1, CJ)
        # residual r = fAI - J = [c_AI^T | -I | c_0] [AI; J; 1]
        Wr = torch.cat((coeffs[1:].T, self.negI, coeffs[0].view(CJ, 1)), 1)
        r = torch.matmul(Wr, Cb, out=self.rbuf)  # (B, CJ, M)
        rW = torch.mul(r, self.Wb, out=self.rWbuf)
        EM = torch.dot(r.view(-1), rW.view(-1)) / 2.0 / self.sigmaM ** 2
        go = torch.matmul(coeffs[1:] / self.sigmaM ** 2, rW, out=self.gobuf)  # (B, CI, M) = dEM/dAI
        # backward: image (grid gradient only)
        _, gg = _BWD[D](_cview(go, D), self.Iexp, _gview(self.gI, D), _BILINEAR, _BORDER, True, [False, True])
        G = torch.mul(gg.view(B, M, D).transpose(1, 2), sI_t.view(1, D, 1), out=self.G)
        # backward through the flow
        vgc = torch.empty((nt, D) + tuple(vs.shape[2:]), device=dev, dtype=dt) if need_vgrad else None
        mask = [bool(need_vgrad), True]
        for t in range(nt):
            gin, gg = _BWD[D](_cview(G, D), vexp[t], _gview(gl[t + 1], D), _BILINEAR, _ZEROS, True, mask)
            if need_vgrad:
                torch.sum(gin, 0, out=vgc[t])
            G.sub_(gg.view(B, M, D).transpose(1, 2))
        if need_vgrad:
            vgc.mul_(-sv)  # dEM/dv on the crop box, physical units
        # affine gradient: sum_j G_j x_j^T over the target grid
        self.Gpl.view(D, B, M).copy_(G.transpose(0, 1))
        Gv = self.Gpl[:, :N].view((D,) + self.nJ)
        Y = torch.matmul(Gv, self.Xlast)  # (D, *nJ[:-1], 2): sums over the last axis of [1, x_last] G
        red = tuple(range(1, D))
        Y0 = Y[..., 0]
        cols = []
        for e in range(D - 1):
            shp = [1] * (D - 1)
            shp[e] = self.nJ[e]
            cols.append((Y0 * self.xJ[e].view(shp)).sum(dim=red))
        cols.append(Y[..., 1].sum(dim=red))
        gAi = torch.zeros((D + 1, D + 1), device=dev, dtype=dt)
        gAi[:D, :D] = (sc_t[:, None] * torch.stack(cols, 1)).flip(0)
        gAi[:D, D] = (sc_t * Y0.sum(dim=red)).flip(0)
        gA = -Ai.T @ gAi @ Ai.T
        return dict(Lgrad=gA[:D, :D], Tgrad=gA[:D, D], EM=EM, vgc=vgc, frame=fr)

    def full_vgrad(self, vgc, slices):
        '''dEM/dv on the whole velocity grid from its crop-box part (zero outside).'''
        vgrad = self.vgf
        if self.crop:
            vgrad.zero_()
        vgrad[(slice(None), slice(None)) + slices] = vgc
        return vgrad

    def residual(self):
        '''fAI - J of the last evaluation, (CJ, *nJ).'''
        return self.rbuf.transpose(0, 1).reshape(self.CJ, self.Np)[:, :self.N].reshape((self.CJ,) + self.nJ)

    def deformed_source(self):
        '''AI of the last evaluation, (CI, *nJ).'''
        return self.AIb.transpose(0, 1).reshape(self.CI, self.Np)[:, :self.N].reshape((self.CI,) + self.nJ)

    def sample_points(self):
        '''Physical coordinates of the image sample points of the last evaluation, (*nJ, D).'''
        D, fr = self.D, self.frame
        g = self.gbuf[0].reshape(D, self.Np)[:, :self.N]
        out = torch.empty((self.N, D), device=self.dev, dtype=self.dt)
        for k in range(D):
            d = D - 1 - k
            gf = (g[k] - fr['beta'][k]) / fr['alpha'][k]
            out[:, d] = self.xv0[d] + (gf + 1.0) * (self.xvs[d] / 2.0)
        return out.view(self.nJ + (D,))


def _points_grad(pIf, pJf, A, vi, xv0_g, xvs_g, nt, sigmaP, vgrad):
    '''Landmark energy EP and its gradients (2D).  The landmarks flow forward in time
    (t = 0..nt-1) and are then mapped by A.  Adds dEP/dv into vgrad if it is given.
    Returns (transformed pointsI, EP, dEP/dL, dEP/dT).'''
    P = pIf.shape[0]
    p = pIf
    grids = []
    for t in range(nt):
        g = ((p - xv0_g) / xvs_g * 2.0 - 1.0).view(1, 1, P, 2)
        out = torch.grid_sampler(vi[t][None], g, _BILINEAR, _ZEROS, True)  # (1, 2, 1, P)
        grids.append(g)
        p = p + out[0, :, 0, :].T / nt
    Alf = A[:2, :2].flip(0).flip(1)
    PA = p @ Alf.T + A[:2, 2].flip(0)
    d = PA - pJf
    EP = torch.sum(d ** 2) / 2.0 / sigmaP ** 2
    q = d / sigmaP ** 2
    gL = (q.T @ p).flip(0).flip(1)
    gT = q.sum(0).flip(0)
    h = q @ Alf
    for t in range(nt - 1, -1, -1):
        go = (h / nt).T.reshape(1, 2, 1, P).contiguous()
        gin, gg = _BWD[2](go, vi[t][None], grids[t], _BILINEAR, _ZEROS, True, [vgrad is not None, True])
        if vgrad is not None:
            vgrad[t] += gin[0]
        h = h + gg[0, 0] * 2.0 / xvs_g
    return PA.flip(1), EP, gL, gT


def _landmark_tol(xv, dtype):
    '''Stopping tolerance of the exact landmark map: _LANDMARK_TOL of the smallest velocity grid
    cell, but at least a few ulps of the grid's coordinates.'''
    h = min(float(x[-1] - x[0]) / (len(x) - 1) for x in xv)
    R = max(max(abs(float(x[0])), abs(float(x[-1]))) for x in xv)
    return max(_LANDMARK_TOL * h, 64.0 * torch.finfo(dtype).eps * R)


def _points_grad_exact(pIf, pJf, A, vi, xv0_g, xvs_g, nt, sigmaP, vgrad, tol):
    '''As _points_grad, with the exact landmark map (see ``_lddmm_common.Problem``).

    Each time step solves z = p_t + v_t(z)/nt by fixed-point iteration from the Euler step (a
    landmark stops once a sweep moves it by at most tol), then takes one Newton step
    p_{t+1} = z + M^-1 (p_t + v_t(z)/nt - z) with M = I - Dv_t(z)/nt, which gives the solution
    up to rounding.  The backward pass applies the implicit function theorem with the same z
    and M: lambda = M^-T h is the gradient at p_t, and lambda/nt is splatted into dEP/dv_t at
    z.  Where M is exactly singular (det M = 0, or NaN) that step is the Euler step
    p_{t+1} = p_t + v_t(p_t)/nt instead, with its gradient (as the C++ engine does).  Returns
    (transformed pointsI, EP, dEP/dL, dEP/dT, number of solves that did not converge within
    _LANDMARK_MAX_ITER sweeps or fell back to the Euler step).'''
    P = pIf.shape[0]
    grid = lambda p: ((p - xv0_g) / xvs_g * 2.0 - 1.0).view(1, 1, P, 2)
    vel = lambda t, p: torch.grid_sampler(vi[t][None], grid(p), _BILINEAR, _ZEROS, True)[0, :, 0, :].T
    eye = torch.eye(2, device=pIf.device, dtype=pIf.dtype)
    unit = eye.view(2, 2, 1, 1).expand(2, 2, 1, P).contiguous()  # grad_output e_k for batch entry k
    p, ps, zs, Ms, euler = pIf, [], [], [], []
    unconverged = 0
    for t in range(nt):
        ps.append(p)
        z = p + vel(t, p) / nt
        z_euler = z
        active = torch.ones(P, dtype=torch.bool, device=p.device)
        for _ in range(_LANDMARK_MAX_ITER):
            zn = p + vel(t, z) / nt
            moved = (zn - z).abs().amax(1)
            z = torch.where(active[:, None], zn, z)
            active &= moved > tol
            if not bool(active.any()):
                break
        unconverged += int(active.sum())
        # Jacobian Dv[p, k, d] = d v_k / d z_d (both in grid_sample order), from one backward
        # pass with a batch of two unit output gradients
        _, gg = _BWD[2](unit, vi[t][None].expand((2,) + tuple(vi[t].shape)), grid(z).expand(2, 1, P, 2),
                        _BILINEAR, _ZEROS, True, [False, True])
        M = eye - gg[:, 0].transpose(0, 1) * (2.0 / xvs_g) / nt
        # det M with the C++ engine's formula (its M is this one with both axes flipped)
        singular = ~((M[:, 0, 0] * M[:, 1, 1] - M[:, 0, 1] * M[:, 1, 0]).abs() > 0)
        if bool(singular.any()):
            unconverged += int((singular & ~active).sum())
            M = torch.where(singular[:, None, None], eye, M)
            euler.append(singular)
        else:
            euler.append(None)
        zs.append(z)
        Ms.append(M)
        p = z + torch.linalg.solve(M, (p + vel(t, z) / nt - z)[..., None])[..., 0]
        if euler[-1] is not None:
            p = torch.where(singular[:, None], z_euler, p)
    Alf = A[:2, :2].flip(0).flip(1)
    PA = p @ Alf.T + A[:2, 2].flip(0)
    d = PA - pJf
    EP = torch.sum(d ** 2) / 2.0 / sigmaP ** 2
    q = d / sigmaP ** 2
    gL = (q.T @ p).flip(0).flip(1)
    gT = q.sum(0).flip(0)
    h = q @ Alf
    for t in range(nt - 1, -1, -1):
        lam = torch.linalg.solve(Ms[t].transpose(1, 2), h[..., None])[..., 0]
        singular = euler[t]
        if singular is not None:  # the Euler steps: splat h/nt at p_t, h <- h + Dv_t(p_t)^T h/nt
            go = torch.where(singular[:, None], h / nt, torch.zeros_like(h)).T.reshape(1, 2, 1, P).contiguous()
            gin, gg = _BWD[2](go, vi[t][None], grid(ps[t]), _BILINEAR, _ZEROS, True, [vgrad is not None, True])
            if vgrad is not None:
                vgrad[t] += gin[0]
            lam = torch.where(singular[:, None], h + gg[0, 0] * 2.0 / xvs_g, lam)
        if vgrad is not None:
            go = (lam / nt).T.reshape(1, 2, 1, P).contiguous()
            if singular is not None:
                go = torch.where(singular.view(1, 1, 1, P), torch.zeros_like(go), go)
            gin, _ = _BWD[2](go, vi[t][None], grid(zs[t]), _BILINEAR, _ZEROS, True, [True, False])
            vgrad[t] += gin[0]
        h = lam
    return PA.flip(1), EP, gL, gT, unconverged


# =========================================================================== smoothing
class _Smooth2D:
    '''Smoothing K * dEM/dv and the regularizer ER for 2D velocities, P = nt*2 slices (n0, n1).

    Spectra are half spectra along axis 1.  mode 'matmul' keeps them as real arrays
    (n0, P, 2*h1) with (re, im) interleaved, computed with dense DFT matrices:
      forward : R = [cos; -sin] @ x  (axis 0),  Y = R_re @ Fr + R_im @ FrS  (axis 1)
      inverse : Q = [cos | sin]/n0 @ [Y; Y S]   (axis 0),  x = Q @ Gr  (c2r, axis 1)
    where S maps each pair (a, b) -> (-b, a).  Modes 'pool' and 'torch' keep complex spectra
    (P, n0, h1) from torch.fft, per slice on the thread pool or in one call.
    '''

    def __init__(self, nv, LL, K, DV, sigmaR, mode, nthreads):
        dev, dt = K.device, K.dtype
        n0, n1 = nv
        h1 = n1 // 2 + 1
        self.n0, self.n1, self.h1, self.mode, self.nthreads = n0, n1, h1, mode, nthreads
        self.DVs = float(DV / sigmaR ** 2)
        Kh = K[:, :h1]
        ERw = LL[:, :h1] * _col_weights(n1, h1, dev, dt) * (DV / 2.0 / n0 / n1 / sigmaR ** 2)
        if mode == 'matmul':
            Fr1, self.Gr1, _ = _dft_mats(n1, dev, dt)
            Fr1S = torch.empty_like(Fr1)
            Fr1S[:, 0::2] = -Fr1[:, 1::2]
            Fr1S[:, 1::2] = Fr1[:, 0::2]
            self.Fr1, self.Fr1S = Fr1, Fr1S
            k = np.arange(n0)
            ph = 2.0 * np.pi * ((k[:, None] * k[None, :]) % n0) / n0
            c, s = np.cos(ph), np.sin(ph)
            self.F0s = torch.as_tensor(np.concatenate((c, -s), 0)).to(dt).to(dev)  # (2 n0, n0)
            self.G0c = torch.as_tensor(np.concatenate((c, s), 1) / n0).to(dt).to(dev)  # (n0, 2 n0)
            self.Kw = Kh.repeat_interleave(2, dim=1)[:, None, :].contiguous()  # (n0, 1, 2 h1)
            self.ERw = ERw.repeat_interleave(2, dim=1)[:, None, :].contiguous()
        else:
            self.Kw = Kh.contiguous()
            self.ERw = ERw.contiguous()
        self.V = None  # tracked spectrum of v, for the reported ER
        self.nupd = 0

    def _fwd(self, x):
        '''Half spectrum of x (P, n0, n1).'''
        if self.mode == 'matmul':
            P, n0, n1 = x.shape
            R = self.F0s @ x.transpose(0, 1).reshape(n0, P * n1)  # (2 n0, P n1)
            Y = R[:n0].view(n0 * P, n1) @ self.Fr1
            Y.addmm_(R[n0:].view(n0 * P, n1), self.Fr1S)
            return Y.view(n0, P, 2 * self.h1)
        if self.mode == 'pool':
            return _pool_map(torch.fft.rfft2, x, self.nthreads)
        return torch.fft.rfft2(x)

    def _inv(self, Y, P):
        '''Inverse of _fwd, (P, n0, n1).'''
        n0, n1 = self.n0, self.n1
        if self.mode == 'matmul':
            YS = torch.empty((2, n0, P, 2 * self.h1), device=Y.device, dtype=Y.dtype)
            YS[0] = Y
            torch.neg(Y[..., 1::2], out=YS[1][..., 0::2])
            YS[1][..., 1::2] = Y[..., 0::2]
            Q = self.G0c @ YS.view(2 * n0, -1)  # (n0, P 2h1)
            return (Q.view(n0 * P, -1) @ self.Gr1).view(n0, P, n1).transpose(0, 1)
        irfft = lambda z: torch.fft.irfft2(z, s=(n0, n1))
        if self.mode == 'pool':
            return _pool_map(irfft, Y, self.nthreads)
        return irfft(Y)

    def _power(self, V):
        return V * V if self.mode == 'matmul' else V.abs().square_()

    def energy(self, vi):
        '''ER(v) for vi (nt, 2, n0, n1).  Uses the tracked spectrum (recomputed every _RESYNC updates).'''
        if self.V is None or self.nupd >= _RESYNC:
            self.V = self._fwd(vi.reshape(-1, self.n0, self.n1))
            self.nupd = 0
        return torch.sum(self._power(self.V) * self.ERw)

    def direction(self, vgrad, vi):
        '''K * dEM/dv + DV/sigmaR^2 v (== K * dE/dv) and the spectrum K * F(dEM/dv).'''
        P = vgrad.shape[0] * vgrad.shape[1]
        KY = self._fwd(vgrad.reshape(P, self.n0, self.n1)) * self.Kw
        out = self._inv(KY, P).reshape(vgrad.shape)
        return out.add_(vi, alpha=self.DVs), KY

    def step(self, vgrad, vi, epV, crop=None):
        '''The velocity update v - epV * K * dE/dv (crop is always None in 2D).'''
        out, KY = self.direction(vgrad, vi)
        if self.V is not None:  # the spectrum of v follows the same recurrence
            self.V.mul_(1.0 - epV * self.DVs).sub_(KY, alpha=epV)
            self.nupd += 1
        return out.mul_(-epV).add_(vi)


class _Smooth3D:
    '''Smoothing K * dEM/dv and the regularizer ER for 3D velocities, P = nt*3 volumes (n0, n1, n2).

    Spectra are half along axis 1: F = fft_2 o rfft_01, F^-1 = irfft_01 o ifft_2 (increasing
    dims only, see the module docstring).  Mode 'pool' runs them per volume on the thread
    pool, mode 'torch' in one call.  A gradient that is zero outside the crop box is
    transformed with three small dense DFTs instead.
    '''

    def __init__(self, nv, LL, K, DV, sigmaR, mode, nthreads):
        dev, dt = K.device, K.dtype
        n0, n1, n2 = nv
        h1 = n1 // 2 + 1
        self.nv, self.h1, self.mode, self.nthreads = nv, h1, mode, nthreads
        self.DVs = float(DV / sigmaR ** 2)
        self.Kh = K[:, :h1, :].contiguous()
        self.ERw = (LL[:, :h1, :] * _col_weights(n1, h1, dev, dt)[:, None]
                    * (DV / 2.0 / n0 / n1 / n2 / sigmaR ** 2)).contiguous()
        self.Fy, _, _ = _dft_mats(n1, dev, dt)  # (n1, 2 h1): rfft along axis 1, interleaved
        _, _, self.Fx = _dft_mats(n2, dev, dt)  # (n2, n2) complex, symmetric
        _, _, self.Fz = _dft_mats(n0, dev, dt)  # (n0, n0) complex
        self.V = None
        self.nupd = 0

    def _map(self, f, x):
        return _pool_map(f, x, self.nthreads) if self.mode == 'pool' else f(x)

    def _fwd(self, x):
        return self._map(lambda y: torch.fft.fft(torch.fft.rfftn(y, dim=(1, 2)), dim=3), x)

    def _inv(self, X):
        n0, n1 = self.nv[0], self.nv[1]
        X = self._map(lambda y: torch.fft.ifft(y, dim=3), X)
        return self._map(lambda y: torch.fft.irfftn(y, s=(n0, n1), dim=(1, 2)), X)

    def _fwd_crop(self, vgc, slices):
        '''Spectrum of a gradient that is zero outside the crop box (vgc: (nt, 3, cz, cy, cx)).'''
        s0, s1, s2 = slices
        P = vgc.shape[0] * vgc.shape[1]
        cz, cy, cx = vgc.shape[2:]
        Y = vgc.reshape(P, cz, cy, cx).transpose(2, 3) @ self.Fy[s1]  # (P, cz, cx, 2 h1): rfft along y
        Yc = torch.view_as_complex(Y.view(P, cz, cx, self.h1, 2))
        X = Yc.transpose(2, 3) @ self.Fx[s2]  # (P, cz, h1, n2): DFT along x
        Z = self.Fz[:, s0] @ X.reshape(P, cz, -1)  # (P, n0, h1 n2): DFT along z
        return Z.view(P, self.nv[0], self.h1, self.nv[2])

    def energy(self, vi):
        '''ER(v) for vi (nt, 3, n0, n1, n2).  Uses the tracked spectrum (recomputed every _RESYNC updates).'''
        if self.V is None or self.nupd >= _RESYNC:
            self.V = self._fwd(vi.reshape((-1,) + tuple(self.nv)))
            self.nupd = 0
        return torch.sum(self.V.abs().square_() * self.ERw)

    def direction(self, vgrad, vi, crop=None):
        '''K * dEM/dv + DV/sigmaR^2 v (== K * dE/dv) and the spectrum K * F(dEM/dv).
        With crop = (vgc, slices) the gradient is given on its crop box only.'''
        if crop is not None:
            H = self._fwd_crop(*crop)
        else:
            H = self._fwd(vgrad.reshape((-1,) + tuple(self.nv)))
        H.mul_(self.Kh)
        out = self._inv(H).reshape(vi.shape)
        return out.add_(vi, alpha=self.DVs), H

    def step(self, vgrad, vi, epV, crop=None):
        '''The velocity update v - epV * K * dE/dv.'''
        out, H = self.direction(vgrad, vi, crop)
        if self.V is not None:  # the spectrum of v follows the same recurrence
            self.V.mul_(1.0 - epV * self.DVs).sub_(H, alpha=epV)
            self.nupd += 1
        return out.mul_(-epV).add_(vi)


# =========================================================================== engine
class TorchEngine:
    '''LDDMM iterations in pure torch, on the problem's device and dtype.

    Parameters
    ----------
    problem : Problem
        The fit (see ``_lddmm_common.Problem``).
    num_threads : int
        Maximum number of CPU threads.  Fewer are used for small problems.
    crop, fft : optional
        Implementation choices, for tests.  ``crop`` (default: on for CPU) restricts the
        velocity steps to the bounding box of the sample points.  ``fft`` picks the
        transforms: 'matmul' (dense DFTs, 2D), 'pool' (torch.fft per volume on a thread pool,
        CPU) or 'torch'.  The default is 'matmul' for 2D CPU grids up to 12k cells, 'pool'
        for other CPU problems, and 'torch' on other devices.  All choices give the same
        results to rounding.
    '''

    def __init__(self, problem, num_threads, *, crop=None, fft=None):
        P = self.problem = problem
        dev, dt = P.device, P.dtype
        cpu = dev.type == 'cpu'
        D = self.dim = P.dim
        self.nthreads = choose_threads(int(np.prod(P.J.shape[1:])), D, num_threads) if cpu else None
        if fft is None:
            ncells = int(np.prod(P.K.shape))
            fft = ('matmul' if D == 2 and ncells <= _DENSE_DFT_MAX_CELLS else 'pool') if cpu else 'torch'
        if fft not in ('matmul', 'pool', 'torch') or (fft == 'matmul' and D == 3) or (fft == 'pool' and not cpu):
            raise ValueError(f'fft={fft!r} is not available for this problem')
        tensor = lambda x: torch.as_tensor(x).to(device=dev, dtype=dt)
        with _TorchThreads(self.nthreads):
            xI = [tensor(x) for x in P.xI]
            xJ = [tensor(x) for x in P.xJ]
            xv = [tensor(x) for x in P.xv]
            self.J = tensor(P.J)
            self.model = _Model(xI, tensor(P.I), xJ, self.J, xv, P.nt, P.sigmaM,
                                _CHUNKS if cpu else 1, cpu if crop is None else crop)
            self.model.set_WM(tensor(P.WM))
            self.WA, self.WB = tensor(P.WA).clone(), tensor(P.WB).clone()
            self.L, self.T = tensor(P.L).clone(), tensor(P.T).clone()
            # internal velocity layout (nt, D, *nv) with components in grid_sample order
            self.vi = tensor(P.v).flip(-1).permute((0, D + 1) + tuple(range(1, D + 1))).contiguous()
            Smooth = _Smooth2D if D == 2 else _Smooth3D
            self.smooth = Smooth(tuple(len(x) for x in P.xv), tensor(P.LL), tensor(P.K), P.DV, P.sigmaR,
                                 fft, self.nthreads or 1)
            self.muA = None if P.muA is None else tensor(P.muA)
            self.muB = None if P.muB is None else tensor(P.muB)
            self.EA = self.EB = None  # cached exp() terms for user-supplied muA / muB
            self.has_points = P.pointsI.shape[0] > 0
            self.lm_exact = self.has_points and P.landmark_map == 'exact'
            self.lm_unconverged = 0
            if self.has_points:
                self.pIf = tensor(P.pointsI).flip(1).contiguous()
                self.pJf = tensor(P.pointsJ).flip(1).contiguous()
                self.xv0_g = torch.stack((xv[1][0], xv[0][0]))
                self.xvs_g = torch.stack((xv[1][-1] - xv[1][0], xv[0][-1] - xv[0][0]))
                self.lm_tol = _landmark_tol(P.xv, dt)
            # mixture EM (see _lddmm_common.Problem): priors, class means, variances (vM, vA, vB)
            # once estimated, and the squared distances to a class mean that stays fixed
            self.mix = P.mixture
            if self.mix is not None:
                pi = tensor(np.asarray(self.mix['pi_init'], dtype=np.float64))
                self.mix_pi = pi / pi.sum()
                self.mix_mu = {'A': tensor(self.mix['muA']), 'B': tensor(self.mix['muB'])}
                self.mix_est = {'A': P.muA is None and not self.mix['fix_muA'],
                                'B': P.muB is None and not self.mix['fix_muB']}
                self.mix_var = None
                self.mix_d2 = {}
            self.Esave = torch.zeros((max(P.niter, 1), 4 if self.has_points else 3), device=dev, dtype=dt)
            # affine step control: one multiplier per affine parameter (L row-major, then T)
            # and the previous gradient, see _lddmm_common.Problem
            self.step_m = self.step_gprev = None
            if P.affine_step_control:
                self.step_m = torch.ones(D * D + D, device=dev, dtype=dt)
                self.step_gprev = torch.zeros(D * D + D, device=dev, dtype=dt)
        self.v_zero = not bool(self.vi.any())
        self.it = 0
        self.A_eval = self.vi_eval = self.pointsIt = None

    # ---- the iteration ------------------------------------------------------
    def run(self, n):
        '''Advance n iterations.'''
        with _TorchThreads(self.nthreads):
            for _ in range(n):
                self._iterate()

    def _gradient(self, A, need_vgrad):
        '''Matching and landmark energies and their gradients at (A, v).

        Returns dict(Lgrad, Tgrad, EM, EP, vgrad, crop).  When need_vgrad, the v gradient is
        either vgrad (full grid, 2D) or crop = (gradient on the crop box, its slices) (3D).'''
        P, m = self.problem, self.model
        res = m.evaluate(torch.linalg.inv(A), self.vi, need_vgrad, self.v_zero)
        res['EP'] = res['vgrad'] = res['crop'] = None
        if need_vgrad:
            slices = res['frame']['slices']
            if self.dim == 3 and m.crop:
                res['crop'] = (res['vgc'], slices)
            else:
                res['vgrad'] = m.full_vgrad(res['vgc'], slices)
        if self.lm_exact:
            self.pointsIt, res['EP'], gL, gT, bad = _points_grad_exact(
                self.pIf, self.pJf, A, self.vi, self.xv0_g, self.xvs_g, P.nt, P.sigmaP, res['vgrad'], self.lm_tol)
            self.lm_unconverged += bad
        elif self.has_points:
            self.pointsIt, res['EP'], gL, gT = _points_grad(self.pIf, self.pJf, A, self.vi, self.xv0_g, self.xvs_g,
                                                            P.nt, P.sigmaP, res['vgrad'])
        if self.has_points:
            res['Lgrad'] = res['Lgrad'] + gL
            res['Tgrad'] = res['Tgrad'] + gT
        return res

    def _iterate(self):
        P, it = self.problem, self.it
        diffeo = it >= P.diffeo_start
        A = _to_A(self.L, self.T)
        res = self._gradient(A, need_vgrad=diffeo)
        # energy history [E, EM, ER(, EP)]; as in the original, E is recorded without EP
        if it >= self.Esave.shape[0]:
            self.Esave = torch.cat((self.Esave, torch.zeros_like(self.Esave)))
        ER = self.smooth.energy(self.vi)
        self.Esave[it, :3] = torch.stack((res['EM'] + ER, res['EM'], ER))
        if res['EP'] is not None:
            self.Esave[it, 3] = res['EP']
        # gradient step; the step sizes drop 10x once the diffeomorphism is on
        fac = 1.0 + diffeo * 9
        if self.step_m is None:
            self.L = self.L - (P.epL / fac) * res['Lgrad']
            self.T = self.T - (P.epT / fac) * res['Tgrad']
        else:
            mL, mT = self._step_multipliers(res['Lgrad'], res['Tgrad'])
            self.L = self.L - (P.epL / fac) * mL * res['Lgrad']
            self.T = self.T - (P.epT / fac) * mT * res['Tgrad']
        self.A_eval, self.vi_eval = A, self.vi
        if diffeo:
            self.vi = self.smooth.step(res['vgrad'], self.vi, P.epV, res['crop'])
            self.v_zero = False
        if self.mix is not None:
            if it >= self.mix['start'] and (it - self.mix['start']) % self.mix['every'] == 0:
                self._mixture_step()
        elif not it % 5:
            self._update_weights(it)
        self.it += 1

    def _step_multipliers(self, Lgrad, Tgrad):
        '''Affine step control: update the per-parameter multipliers from the sign of the new
        gradient times the previous one; returns them shaped like L and T.'''
        P, D = self.problem, self.dim
        g = torch.cat((Lgrad.reshape(-1), Tgrad.reshape(-1)))
        s = g * self.step_gprev
        m = self.step_m
        down = torch.clamp(m * P.step_shrink, min=P.step_floor)
        up = torch.clamp(m * P.step_grow, max=P.step_cap)
        self.step_m = torch.where(s < 0, down, torch.where(s > 0, up, m))
        self.step_gprev = g
        return self.step_m[:D * D].view(D, D), self.step_m[D * D:]

    def _update_weights(self, it):
        '''M step for the class means, then (from iteration 50) the E step for the weights.'''
        P, m, J = self.problem, self.model, self.J
        sdims = tuple(range(-self.dim, 0))
        if P.muA is None:
            self.muA = torch.sum(self.WA * J, dim=sdims) / torch.sum(self.WA)
        if P.muB is None:
            self.muB = torch.sum(self.WB * J, dim=sdims) / torch.sum(self.WB)
        if it < 50:
            return
        grid = (Ellipsis,) + (None,) * self.dim
        C = J.shape[0]
        W = torch.stack((m.WM, self.WA, self.WB))
        pi = torch.sum(W, dim=tuple(range(1, self.dim + 1)))
        pi += torch.max(pi) * 1e-6
        pi /= torch.sum(pi)
        WM = pi[0] * torch.exp(-torch.sum(m.residual() ** 2, 0) / 2.0 / P.sigmaM ** 2) / np.sqrt(2.0 * np.pi * P.sigmaM ** 2) ** C
        if P.muA is None or self.EA is None:
            self.EA = torch.exp(-torch.sum((self.muA[grid] - J) ** 2, 0) / 2.0 / P.sigmaA ** 2)
        if P.muB is None or self.EB is None:
            self.EB = torch.exp(-torch.sum((self.muB[grid] - J) ** 2, 0) / 2.0 / P.sigmaB ** 2)
        WA = pi[1] * self.EA / np.sqrt(2.0 * np.pi * P.sigmaA ** 2) ** C
        WB = pi[2] * self.EB / np.sqrt(2.0 * np.pi * P.sigmaB ** 2) ** C
        WS = WM + WB + WA
        WS += torch.max(WS) * 1e-6
        WM /= WS
        WB /= WS
        WA /= WS
        m.set_WM(WM)
        self.WA, self.WB = WA, WB

    def _mixture_step(self):
        '''E-step and M-step of the mixture EM from this iteration's residual (see
        ``_lddmm_common.Problem``).'''
        P, m, J, mix = self.problem, self.model, self.J, self.mix
        C = J.shape[0]
        grid = (Ellipsis,) + (None,) * self.dim
        sdims = tuple(range(-self.dim, 0))
        d2M = torch.sum(m.residual() ** 2, 0)
        if self.mix_var is None:  # first E-step: the matching variance of the unweighted residual
            vM = max(float(d2M.mean()) / C, _VARIANCE_FLOOR)
            self.mix_var = (vM, (P.sigmaA / P.sigmaM) ** 2 * vM, (P.sigmaB / P.sigmaM) ** 2 * vM)
        vM, vA, vB = self.mix_var
        lp = torch.log(self.mix_pi)
        logl = [lp[0] - d2M / (2 * vM) - 0.5 * C * math.log(vM)]
        for k, c, var in ((1, 'A', vA), (2, 'B', vB)):
            if c not in mix['classes']:
                logl.append(torch.full_like(d2M, -math.inf))
                continue
            d2 = self.mix_d2.get(c)
            if d2 is None:
                d2 = torch.sum((J - self.mix_mu[c][grid]) ** 2, 0)
                if not self.mix_est[c]:
                    self.mix_d2[c] = d2
            logl.append(lp[k] - d2 / (2 * var) - 0.5 * C * math.log(var))
        W = torch.softmax(torch.stack(logl), 0)
        m.set_WM(W[0])
        self.WA, self.WB = W[1], W[2]
        sums = W.sum(dim=tuple(range(1, self.dim + 1)))
        pi = torch.clamp(sums / sums.sum(), min=1e-4)
        if pi[0] < mix['pi_min']:
            pi = torch.cat((pi.new_tensor([mix['pi_min']]), pi[1:] / pi[1:].sum() * (1.0 - mix['pi_min'])))
        self.mix_pi = pi / pi.sum()
        sM, sA, sB = (float(x) for x in sums)
        for c, Wc, s in (('A', W[1], sA), ('B', W[2], sB)):
            if self.mix_est[c] and s > 1e-3:
                self.mix_mu[c] = torch.sum(Wc * J, dim=sdims) / s
        if sM > 0:
            vM = max(float(torch.sum(W[0] * d2M)) / (C * sM), _VARIANCE_FLOOR)
        self.mix_var = (vM, (P.sigmaA / P.sigmaM) ** 2 * vM, (P.sigmaB / P.sigmaM) ** 2 * vM)

    # ---- outputs ------------------------------------------------------------
    def _v_out(self, vi):
        '''Internal velocity layout -> original layout (nt, *nv, D) (a copy).'''
        D = self.dim
        return vi.permute((0,) + tuple(range(2, D + 2)) + (1,)).flip(-1).contiguous()

    def snapshot(self):
        '''Current state in the original layouts (copies); see ``_lddmm_common.run_lddmm``.'''
        m = self.model
        snap = {'L': self.L.clone(), 'T': self.T.clone(), 'A_eval': self.A_eval.clone(),
                'v': self._v_out(self.vi), 'v_eval': self._v_out(self.vi_eval),
                'WM': m.WM.clone(), 'WB': self.WB.clone(), 'WA': self.WA.clone(),
                'AI': m.deformed_source().clone(), 'fAI': m.residual() + self.J,
                'Esave': self.Esave[:self.it].clone()}
        if self.dim == 3:
            snap['Xs'] = m.sample_points()
        if self.has_points:
            snap['pointsIt'] = self.pointsIt.clone()
        if self.lm_exact:
            snap['landmark_unconverged'] = self.lm_unconverged
        if self.step_m is not None:
            snap['step_multipliers'] = self.step_m.clone()
        return snap

    def gradient(self):
        '''Gradient at the current state, without updating it (for tests).

        Returns dict(Lgrad, Tgrad, vgrad = d(EM + EP)/dv, direction = K * dE/dv (the update
        direction, regularizer included), EM, ER, EP) in the original layouts.'''
        with _TorchThreads(self.nthreads):
            res = self._gradient(_to_A(self.L, self.T), need_vgrad=True)
            vgrad = res['vgrad'] if res['crop'] is None else self.model.full_vgrad(*res['crop'])
            direction, _ = self.smooth.direction(vgrad, self.vi)
            return dict(Lgrad=res['Lgrad'].clone(), Tgrad=res['Tgrad'].clone(), vgrad=self._v_out(vgrad),
                        direction=self._v_out(direction), EM=res['EM'], ER=self.smooth.energy(self.vi), EP=res['EP'])
