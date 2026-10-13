'''Shared front-end of the LDDMM functions: inputs, kernels, threads, figures and backends.

The public functions in ``_lddmm.py`` turn their arguments into a :class:`Problem` with
:func:`prepare_problem` and hand it to :func:`run_lddmm`.  That function picks a backend
engine, runs the iterations in segments, draws the progress figures in between if asked, and
assembles the returned dict.

An engine implements::

    Engine(problem, num_threads)  # raise Unsupported from __init__ to decline a problem
    engine.run(n)                 # advance n iterations; the iteration counter persists
    engine.snapshot()             # dict of the current state, see run_lddmm

The compiled engine is ``_lddmm_cpp.CppEngine`` (CPU, float64).  The pure torch engine,
``_lddmm_torch.TorchEngine``, runs everywhere else and is the fallback.
'''
import dataclasses
import importlib
import os
import warnings
from typing import Optional

import numpy as np
import torch
from torch.nn.functional import grid_sample

# Default sigmaR of LDDMM_3D_to_slice (change it here only).  The corrected 3D regularizer
# (Fourier transform over all three axes) is much weaker than the old one at the same sigmaR:
# at the old default 1e8 the rms velocity of the 3D tutorial fits became up to twice as large
# as with the old code.  At 2e6 it is about as large as with the old code.
DEFAULT_SIGMAR_3D = 2e6

# Affine step control (affine_step_control=True; default in LDDMM_3D_to_slice only).  Every
# affine parameter (the d*d entries of L, row-major, then the d entries of T) has a step
# multiplier m, initially 1.  After each gradient, m is multiplied by AFFINE_STEP_SHRINK (but
# kept >= AFFINE_STEP_FLOOR) when the parameter's gradient changed sign since the previous
# iteration (the last step overshot), and by AFFINE_STEP_GROW (but kept <= AFFINE_STEP_CAP)
# when it kept its sign; a zero product leaves m unchanged.  The step is then m * epL/fac or
# m * epT/fac, never larger than without the control.  This is the sign rule of Rprop applied
# to the gradient steps.
AFFINE_STEP_SHRINK = 0.5
AFFINE_STEP_GROW = 1.05
AFFINE_STEP_FLOOR = 1e-3
AFFINE_STEP_CAP = 1.0

BACKENDS = ('auto', 'cpp', 'torch')

# Iterations per engine.run call.  Short segments keep long fits responsive to Ctrl-C.
_SEGMENT = 100


class Unsupported(Exception):
    '''Raised by an engine's constructor for a problem it cannot run.'''


@dataclasses.dataclass(eq=False, repr=False)
class Problem:
    '''Everything a backend engine needs for one LDDMM run.

    Arrays are float64 C-contiguous numpy arrays that belong to the problem; engines must not
    modify them.  ``L``, ``T`` and ``v`` are the initial state and ``WM``, ``WB``, ``WA`` the
    initial weights.  In 3D the target has a slice axis: ``J`` is (C_J, 1, H, W) and ``xJ``
    is [array([z]), y, x].  ``muA`` / ``muB`` are None when they are estimated.

    With ``affine_step_control`` the engines scale each affine step by a per-parameter
    multiplier, updated every iteration before the affine update::

        g = concat(dE/dL (row-major), dE/dT);  s = g * g_prev
        m = where(s < 0, max(step_shrink * m, step_floor),
                  where(s > 0, min(step_grow * m, step_cap), m));  g_prev = g
        L -= (epL / fac) * m[:d*d].reshape(d, d) * dE/dL;  T -= (epT / fac) * m[d*d:] * dE/dT

    with m = 1 and g_prev = 0 initially (fac = 10 once the diffeomorphism is on, else 1).
    The state belongs to the engine and persists across ``engine.run`` calls; engines report
    m as ``snapshot()['step_multipliers']`` (only with the control on).
    '''
    dim: int
    xI: list
    I: np.ndarray
    xJ: list
    J: np.ndarray
    L: np.ndarray
    T: np.ndarray
    v: np.ndarray
    xv: list
    LL: np.ndarray
    K: np.ndarray
    DV: float
    WM: np.ndarray
    WB: np.ndarray
    WA: np.ndarray
    muA: Optional[np.ndarray]
    muB: Optional[np.ndarray]
    pointsI: np.ndarray
    pointsJ: np.ndarray
    a: float
    p: float
    nt: int
    niter: int
    diffeo_start: int
    epL: float
    epT: float
    epV: float
    sigmaM: float
    sigmaB: float
    sigmaA: float
    sigmaR: float
    sigmaP: float
    device: torch.device
    dtype: torch.dtype
    affine_step_control: bool = False
    step_shrink: float = AFFINE_STEP_SHRINK
    step_grow: float = AFFINE_STEP_GROW
    step_floor: float = AFFINE_STEP_FLOOR
    step_cap: float = AFFINE_STEP_CAP


# =========================================================================== inputs
def as_float64(x):
    '''Copy of x (numpy array, torch tensor on any device, list or scalar) as a float64
    C-contiguous numpy array.'''
    if isinstance(x, torch.Tensor):
        x = x.detach().to('cpu', torch.float64).numpy()
    return np.array(x, dtype=np.float64, order='C', copy=True)


def affine(L, T):
    '''Affine matrix [[L, T], [0, 1]] as a numpy array.'''
    d = L.shape[0]
    A = np.eye(d + 1)
    A[:d, :d] = L
    A[:d, d] = T
    return A


def velocity_grid(xI, expand, a):
    '''Velocity grid: the box of the source grid xI grown by ``expand``, spacing a/2.

    Computed in float64 with the same operations as the original STalign.'''
    f64 = torch.float64
    minv = torch.as_tensor([x[0] for x in xI], dtype=f64)
    maxv = torch.as_tensor([x[-1] for x in xI], dtype=f64)
    minv, maxv = (minv + maxv) * 0.5 + 0.5 * torch.tensor([-1.0, 1.0], dtype=f64)[..., None] * (maxv - minv) * expand
    return [torch.arange(m, M, a * 0.5, dtype=f64).numpy() for m, M in zip(minv, maxv)]


def kernels(xv, a, p):
    '''Fourier symbol of the regularizer on the velocity grid.

    Returns (LL, K, DV): LL is the symbol of (1 - a^2 Laplacian)^(2p) (finite differences),
    K = 1/LL is the smoothing kernel applied to the gradient, and DV the voxel volume.'''
    f64 = torch.float64
    xv = [torch.as_tensor(x, dtype=f64) for x in xv]
    dv = torch.as_tensor([x[1] - x[0] for x in xv], dtype=f64)
    fv = [torch.arange(len(x), dtype=f64) / len(x) / d for x, d in zip(xv, dv)]
    FV = torch.stack(torch.meshgrid(fv, indexing='ij'), -1)
    LL = (1.0 + 2.0 * a ** 2 * torch.sum((1.0 - torch.cos(2.0 * np.pi * FV * dv)) / dv ** 2, -1)) ** (p * 2.0)
    K = 1.0 / LL
    return LL.numpy(), K.numpy(), float(torch.prod(dv))


def coerce_mu(mu, C, name):
    '''A user-supplied class mean (muA or muB) as a float64 array with one value per channel.

    Accepts ints, floats, lists, numpy arrays and torch tensors.  Size C is used as is and
    size 1 is broadcast.  Another size whose entries are all equal warns and uses that value:
    older STalign versions broadcast such a mean against every channel, which multiplied the
    class's squared distance by its size.  Anything else raises ValueError.
    '''
    if mu is None:
        return None
    mu = as_float64(mu).reshape(-1)
    if mu.size == C:
        return mu
    if mu.size == 1:
        return np.full(C, mu[0])
    if mu.size > 1 and np.all(mu == mu[0]):
        warnings.warn(f'{name} has {mu.size} entries but the target image has {C} channel(s); all entries are '
                      f'equal, so {name}[0] is used for every channel. (Older STalign versions broadcast it, '
                      f'which made this class {mu.size}x narrower than sigma.) Pass one value per channel '
                      f'to silence this warning.', stacklevel=4)
        return np.full(C, mu[0])
    raise ValueError(f'{name} must have 1 or {C} entries (one per channel of the target image), got {mu.size}')


def prepare_problem(dim, xI, I, xJ, J, pointsI=None, pointsJ=None, L=None, T=None, A=None, v=None, xv=None,
                    a=500.0, p=2.0, expand=2.0, nt=3, niter=5000, diffeo_start=0, epL=2e-8, epT=2e-1, epV=2e3,
                    sigmaM=1.0, sigmaB=2.0, sigmaA=5.0, sigmaR=5e5, sigmaP=2e1, device='cpu',
                    dtype=torch.float64, muA=None, muB=None, affine_step_control=False):
    '''Validate and normalize the arguments of LDDMM (dim=2) or LDDMM_3D_to_slice (dim=3).

    Follows the original conventions: A excludes L and T, v requires xv, and the velocity grid
    is built from xI when neither is given.  2D landmarks must come in pairs; in 3D they are
    dropped (the caller warns).  The constants of the affine step control are read from this
    module (AFFINE_STEP_*) at call time.
    '''
    device = torch.device(device)
    if not (isinstance(dtype, torch.dtype) and dtype.is_floating_point):
        raise ValueError(f'dtype must be a floating point torch dtype, got {dtype!r}')
    if not isinstance(affine_step_control, (bool, np.bool_)):
        raise ValueError(f'affine_step_control must be True or False, got {affine_step_control!r}')
    step = dict(step_shrink=float(AFFINE_STEP_SHRINK), step_grow=float(AFFINE_STEP_GROW),
                step_floor=float(AFFINE_STEP_FLOOR), step_cap=float(AFFINE_STEP_CAP))
    if not (0.0 < step['step_shrink'] < 1.0 and step['step_grow'] >= 1.0
            and 0.0 < step['step_floor'] <= step['step_cap'] <= 1.0):
        raise ValueError(f'the affine step control needs 0 < AFFINE_STEP_SHRINK < 1, AFFINE_STEP_GROW >= 1 and '
                         f'0 < AFFINE_STEP_FLOOR <= AFFINE_STEP_CAP <= 1, got {step}')

    # affine
    if A is not None:
        if L is not None or T is not None:
            raise Exception('If specifying A, you must not specify L or T')
        A = as_float64(A)
        L, T = A[:dim, :dim], A[:dim, -1]
    L = np.eye(dim) if L is None else as_float64(L)
    T = np.zeros(dim) if T is None else as_float64(T).reshape(-1)
    if L.shape != (dim, dim) or T.shape != (dim,):
        raise ValueError(f'L must be {dim}x{dim} and T must have {dim} entries, got shapes {L.shape} and {T.shape}')

    # images and their pixel locations
    I = as_float64(I)
    J = as_float64(J)
    if dim == 3:
        if J.ndim == 3:
            J = J[:, None]  # add the slice axis
        if len(xJ) == 2:
            xJ = [[0.0], xJ[0], xJ[1]]
    xI = [as_float64(x).reshape(-1) for x in xI]
    xJ = [as_float64(x).reshape(-1) for x in xJ]
    for name, x, img in (('I', xI, I), ('J', xJ, J)):
        if len(x) != dim or img.ndim != dim + 1:
            raise ValueError(f'{name} must be (channels, {dim} spatial axes) with {dim} pixel location arrays; '
                             f'got shape {img.shape} and {len(x)} arrays')
        if img.shape[1:] != tuple(len(xi) for xi in x):
            raise ValueError(f'shape of {name} {img.shape} does not match its pixel locations '
                             f'{tuple(len(xi) for xi in x)}')

    # velocity field
    if v is not None and xv is not None:
        v = as_float64(v)
        xv = [as_float64(x).reshape(-1) for x in xv]
        if len(xv) != dim or v.ndim != dim + 2 or v.shape[1:] != tuple(len(x) for x in xv) + (dim,):
            raise ValueError(f'v must have shape (nt, {", ".join("len(xv[%d])" % i for i in range(dim))}, {dim}), '
                             f'got {v.shape} with xv lengths {tuple(len(x) for x in xv)}')
        nt = v.shape[0]
    elif v is None and xv is None:
        xv = velocity_grid(xI, expand, a)
        v = np.zeros((int(nt),) + tuple(len(x) for x in xv) + (dim,))
    else:
        raise Exception('If inputting an initial v, must input both xv and v')
    if nt < 1 or min(len(x) for x in xv) < 2:
        raise ValueError('need nt >= 1 and at least 2 velocity grid points along each axis')
    LL, K, DV = kernels(xv, a, p)

    # landmarks (2D only)
    if dim == 3 or (pointsI is None and pointsJ is None):
        pointsI = pointsJ = np.zeros((0, 2))
    elif pointsI is None or pointsJ is None:
        raise Exception('Must specify corresponding sets of points or none at all')
    else:
        pointsI = as_float64(pointsI)
        pointsJ = as_float64(pointsJ)
        if pointsI.ndim != 2 or pointsI.shape[1] != 2 or pointsI.shape != pointsJ.shape:
            raise ValueError(f'pointsI and pointsJ must both be N x 2, got {pointsI.shape} and {pointsJ.shape}')

    niter = int(niter)
    if niter < 0:
        raise ValueError(f'niter must be >= 0, got {niter}')
    C = J.shape[0]
    return Problem(
        dim=dim, xI=xI, I=I, xJ=xJ, J=J, L=L, T=T, v=v, xv=xv, LL=LL, K=K, DV=DV,
        WM=np.full(J.shape[1:], 0.5), WB=np.full(J.shape[1:], 0.4), WA=np.full(J.shape[1:], 0.1),
        muA=coerce_mu(muA, C, 'muA'), muB=coerce_mu(muB, C, 'muB'), pointsI=pointsI, pointsJ=pointsJ,
        a=float(a), p=float(p), nt=int(nt), niter=niter, diffeo_start=int(diffeo_start),
        epL=float(epL), epT=float(epT), epV=float(epV), sigmaM=float(sigmaM), sigmaB=float(sigmaB),
        sigmaA=float(sigmaA), sigmaR=float(sigmaR), sigmaP=float(sigmaP), device=device, dtype=dtype,
        affine_step_control=bool(affine_step_control), **step)


# =========================================================================== threads
def resolve_num_threads(num_threads=None):
    '''Maximum number of CPU threads for a fit.

    The ``num_threads`` argument if given, else the environment variable STALIGN_NUM_THREADS,
    else torch.get_num_threads() (which follows torch.set_num_threads and OMP_NUM_THREADS).
    Engines never use more; they may use fewer for small problems.
    '''
    source = 'num_threads'
    if num_threads is None:
        env = os.environ.get('STALIGN_NUM_THREADS', '').strip()
        if env:
            num_threads, source = env, 'STALIGN_NUM_THREADS'
        else:
            num_threads = torch.get_num_threads()
    try:
        n = int(num_threads)
    except (TypeError, ValueError):
        n = 0
    if n < 1 or isinstance(num_threads, bool):
        raise ValueError(f'{source} must be a positive integer, got {num_threads!r}')
    return n


# =========================================================================== backends
def _sibling(name):
    '''Import a sibling module, in the package or with the STalign directory on sys.path.'''
    if __package__:
        return importlib.import_module('.' + name, __package__)
    return importlib.import_module(name)


def select_engine(problem, backend='auto', num_threads=None):
    '''Construct the engine for a problem.

    backend='auto' uses the compiled engine when it is installed and supports the problem
    (CPU, float64), else the torch engine.  backend='cpp' raises RuntimeError when the
    compiled engine is unavailable or declines the problem; backend='torch' always uses torch.
    '''
    if backend not in BACKENDS:
        raise ValueError(f'backend must be one of {BACKENDS}, got {backend!r}')
    nthreads = resolve_num_threads(num_threads)
    if backend in ('auto', 'cpp'):
        try:
            return _sibling('_lddmm_cpp').CppEngine(problem, nthreads)
        except (ImportError, Unsupported) as e:
            if backend == 'cpp':
                raise RuntimeError(f"backend='cpp' cannot run this problem: {type(e).__name__}: {e}") from e
    return _sibling('_lddmm_torch').TorchEngine(problem, nthreads)


def backend_info():
    '''Which LDDMM backends are available.

    Returns
    -------
    info : dict
        'auto': the backend that backend='auto' uses for CPU float64 problems ('cpp' or 'torch');
        'cpp': whether the compiled extension can be imported; 'cpp_error': why not (or None);
        'num_threads': the default maximum number of threads (see resolve_num_threads).
    '''
    try:
        _sibling('_lddmm_cpp')
        cpp, err = True, None
    except ImportError as e:
        cpp, err = False, f'{type(e).__name__}: {e}'
    return {'auto': 'cpp' if cpp else 'torch', 'cpp': cpp, 'cpp_error': err, 'num_threads': resolve_num_threads()}


# =========================================================================== the fit
def draw_iterations(draw, niter):
    '''Iterations after which the progress figures are drawn.

    'end' draws once after the last iteration; True every 10 iterations (the original
    behavior, iterations 0, 10, 20, ...); a positive int N every N iterations; False, None
    or 0 never.'''
    if draw is None or draw is False:
        return []
    if isinstance(draw, str):
        if draw != 'end':
            raise ValueError(f"draw must be 'end', True, False, None or an int, got {draw!r}")
        return [niter - 1] if niter > 0 else []
    every = 10 if draw is True else int(draw)
    if every < 0:
        raise ValueError(f'draw must be >= 0, got {draw!r}')
    return list(range(0, niter, every)) if every else []


def run_lddmm(problem, draw='end', num_threads=None, backend='auto'):
    '''Run the fit described by ``problem`` and return the public result dict.

    The result holds the last evaluated pair: 'A' and 'v' are the affine and the velocity
    used by the last iteration's forward pass, 'WM', 'WB', 'WA' the weights after the last
    iteration and, in 3D, 'Xs' the sample points of that forward pass.  With niter=0 it is
    the initial state.  Tensors are on the requested device and dtype.
    '''
    engine = select_engine(problem, backend, num_threads)
    if problem.niter == 0:
        return _initial_result(problem)
    draws = set(draw_iterations(draw, problem.niter))
    figures = None
    done = 0
    for stop in sorted({it + 1 for it in draws} | {problem.niter}):
        while done < stop:
            n = min(stop - done, _SEGMENT)
            engine.run(n)
            done += n
        if stop - 1 in draws:
            try:
                if figures is None:
                    figures = _Figures(problem)
                figures.update(engine.snapshot())
            except Exception as e:  # a plotting problem must never abort a long fit
                warnings.warn(f'LDDMM drawing failed ({type(e).__name__}: {e}); continuing without figures',
                              stacklevel=3)
                draws = set()
    return _result(problem, engine.snapshot())


def _result(problem, snap):
    out = {'A': snap['A_eval'], 'v': snap['v_eval'], 'xv': problem.xv,
           'WM': snap['WM'], 'WB': snap['WB'], 'WA': snap['WA']}
    if problem.dim == 3:
        out['Xs'] = snap['Xs']
    return {k: ([_tensor(x, problem) for x in val] if k == 'xv' else _tensor(val, problem)) for k, val in out.items()}


def _initial_result(problem):
    snap = {'A_eval': affine(problem.L, problem.T), 'v_eval': problem.v,
            'WM': problem.WM, 'WB': problem.WB, 'WA': problem.WA}
    if problem.dim == 3:
        snap['Xs'] = initial_sample_points(problem)
    return _result(problem, snap)


def _tensor(x, problem):
    return torch.as_tensor(x).to(device=problem.device, dtype=problem.dtype)


def initial_sample_points(problem):
    '''Sample points of a forward pass at the initial A and v, (*nJ, d) (the 3D 'Xs' for niter=0).

    Same steps as the original: X = A^-1 x on the target grid, then the backward flow
    X <- X - v_t(X)/nt for t = nt-1, ..., 0 with trilinear (zero padded) interpolation.'''
    f64 = torch.float64
    d, nt = problem.dim, problem.nt
    XJ = torch.stack(torch.meshgrid(*[torch.as_tensor(x, dtype=f64) for x in problem.xJ], indexing='ij'), -1)
    Ai = torch.linalg.inv(torch.as_tensor(affine(problem.L, problem.T)))
    X = (Ai[:d, :d] @ XJ[..., None])[..., 0] + Ai[:d, d]
    v = torch.as_tensor(problem.v)
    lo = torch.tensor([x[0] for x in problem.xv], dtype=f64)
    span = torch.tensor([x[-1] - x[0] for x in problem.xv], dtype=f64)
    perm = (0, d + 1) + tuple(range(1, d + 1))  # (nt, *nv, d) -> (nt, d, *nv)
    vc = v.permute(perm)
    for t in range(nt - 1, -1, -1):
        g = ((X - lo) / span * 2.0 - 1.0).flip(-1)  # grid_sample wants (x, y[, z]) order
        X = X - grid_sample(vc[t][None], g[None], align_corners=True)[0].permute(tuple(range(1, d + 1)) + (0,)) / nt
    return X


# =========================================================================== figures
def extent_from_x(xs):
    '''imshow extent (left, right, bottom, top) for pixel locations xs = [rows, cols].'''
    d0, d1 = xs[0][1] - xs[0][0], xs[1][1] - xs[1][0]
    return (float(xs[1][0] - d1 / 2.0), float(xs[1][-1] + d1 / 2.0),
            float(xs[0][-1] + d0 / 2.0), float(xs[0][0] - d0 / 2.0))


def _numpy(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


class _Figures:
    '''The original's progress figures.

    A figure of the smoothing kernel, a 2x3 panel (deformed source, contrast-transformed
    source, target, velocity at the first time step, weights WM/WA/WB as RGB, error) and a
    1x3 panel of the energy history.  In 3D the images are shown at the target slice and the
    velocity at the middle plane of the velocity grid.
    '''

    def __init__(self, problem):
        import matplotlib.pyplot as plt
        self.dim = problem.dim
        lead = slice(None) if problem.dim == 2 else slice(1, None)  # 3D: drop the slice axis
        self.extentV = extent_from_x(problem.xv[lead])
        self.extentJ = extent_from_x(problem.xJ[lead])
        Ki = np.fft.ifftn(problem.K).real
        if problem.dim == 3:
            Ki = Ki[Ki.shape[0] // 2]
        fig, ax = plt.subplots()
        ax.imshow(Ki, vmin=0.0, extent=self.extentV)
        ax.set_title('smoothing kernel')
        fig.canvas.draw()
        self.fig, ax = plt.subplots(2, 3)
        self.ax = ax.ravel()
        self.figE, axE = plt.subplots(1, 3)
        self.axE = axE.ravel()
        self.J = problem.J
        self.Jmax = float(np.max(problem.J))
        self.pointsJ = problem.pointsJ

    def update(self, snap):
        s = {k: _numpy(val) for k, val in snap.items()}
        AI, fAI, J, Jmax = s['AI'], s['fAI'], self.J, self.Jmax
        clip = lambda x: np.clip(x, 0.0, 1.0)
        sdims = tuple(range(1, AI.ndim))
        AIn = (AI - AI.min(axis=sdims, keepdims=True)) / (AI.max(axis=sdims, keepdims=True) - AI.min(axis=sdims, keepdims=True))
        W = np.stack((s['WM'], s['WA'], s['WB']), -1)
        v0 = s['v'][0]
        vmax = np.max(np.abs(v0))
        v0 = clip((v0 / vmax if vmax > 0 else v0) * 0.5 + 0.5)
        if self.dim == 2:
            chan = lambda x: x.transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
            v0 = np.concatenate((v0, np.zeros_like(v0[..., :1])), -1)
            limits = {}
        else:
            chan = lambda x: x[0, 0]  # first channel at the target slice
            W, v0 = W[0], v0[v0.shape[0] // 2]
            limits = dict(vmin=0, vmax=1)
        pI = s.get('pointsIt', np.zeros((0, 2)))
        pJ = self.pointsJ
        panels = [(0, chan(AIn), 'space tformed source', {}, (pI,)),
                  (1, clip(chan(fAI) / Jmax), 'contrast tformed source', limits, (pI,)),
                  (5, chan(clip((fAI - J) / Jmax * 3.0)) * 0.5 + 0.5, 'Error', {}, (pI, pJ)),
                  (2, chan(J) / Jmax, 'Target', limits, (pJ,)),
                  (4, clip(W), 'Weights', {}, ()),
                  (3, v0, 'velocity', {}, ())]
        for i, img, title, kw, points in panels:
            ax = self.ax[i]
            ax.cla()
            ax.imshow(img, extent=self.extentV if i == 3 else self.extentJ, **kw)
            if self.dim == 2:
                for pts in points:
                    ax.scatter(pts[:, 1], pts[:, 0])
            ax.set_title(title)
        E = s['Esave']
        for ax, y, legend in ((self.axE[0], E, ['E', 'EM', 'ER', 'EP']), (self.axE[1], E[:, :2], ['E', 'EM']),
                              (self.axE[2], E[:, 2], ['ER'])):
            ax.cla()
            ax.plot(y)
            ax.legend(legend)
            with warnings.catch_warnings():  # ER is 0 while v is 0, which log scale ignores
                warnings.simplefilter('ignore', UserWarning)
                ax.set_yscale('log')
        self.fig.canvas.draw()
        self.figE.canvas.draw()
