'''C++ backend of LDDMM and LDDMM_3D_to_slice (CPU, float64).

:class:`CppEngine` implements the engine interface of ``_lddmm_common`` on top of the
compiled extension ``_stalign_cpp`` (sources in STalign/_cpp, built by setup.py).  A whole
iteration (sample points, deformed source, contrast fit, analytic gradients, FFT smoothing,
parameter updates and EM weights) runs in C++ on a pool of std::threads with the GIL
released.  The arithmetic follows the torch implementation in float64 and its result does not
depend on the number of threads.

Importing this module raises ImportError when the extension is not built, and the engine
raises ``Unsupported`` for problems it cannot run; ``select_engine`` then uses the torch
engine instead.
'''
import importlib
import os
import sys

import torch

if __package__:
    from ._lddmm_common import Unsupported
else:  # the STalign directory is on sys.path (the tutorials' "option B" import)
    from _lddmm_common import Unsupported


def _import_extension():
    '''The compiled module next to this file, imported at most once per process.

    This file may be imported both as STalign._lddmm_cpp and, with the STalign directory on
    sys.path, as _lddmm_cpp.  Initializing the same shared library a second time under the
    other name fails, so an already imported copy is reused.
    '''
    here = os.path.dirname(os.path.abspath(__file__))
    for name in ('STalign._stalign_cpp', '_stalign_cpp'):
        module = sys.modules.get(name)
        path = getattr(module, '__file__', None)
        if path and os.path.dirname(os.path.abspath(path)) == here:
            return module
    if __package__:
        return importlib.import_module('._stalign_cpp', __package__)
    return importlib.import_module('_stalign_cpp')


_ext = _import_extension()

# Binding interface this module is written for.  A stale build of the extension (e.g. an old
# in-place build next to newer sources) counts as missing, so the torch engine is used.
_API_VERSION = 4
if getattr(_ext, 'API_VERSION', None) != _API_VERSION:
    raise ImportError(f'the C++ extension {_ext.__file__} is out of date (API version '
                      f'{getattr(_ext, "API_VERSION", None)}, expected {_API_VERSION}); rebuild it with '
                      f'"pip install ." or "python setup.py build_ext --inplace"')


class CppEngine:
    '''LDDMM engine running on the compiled extension.

    Parameters
    ----------
    problem : _lddmm_common.Problem
        The fit, as built by ``prepare_problem``.  Its arrays are read, never modified;
        the source and target images are used in place (not copied).
    num_threads : int
        Number of threads (>= 1).  The thread pool is shared by all engines of a process
        and re-created in a child process after fork.

    Raises
    ------
    Unsupported
        For a device other than the CPU, a dtype other than float64, more channels or time
        steps than the engine supports, or a source image with fewer than 2 pixels along an
        axis.
    '''

    def __init__(self, problem, num_threads):
        P = problem
        if P.device.type != 'cpu' or P.dtype != torch.float64:
            raise Unsupported(f'the C++ engine runs on the CPU in float64 only, not on '
                              f'device={P.device} with dtype={P.dtype}')
        limits = ((P.I.shape[0], _ext.MAX_CI, 'source image channels'),
                  (P.J.shape[0], _ext.MAX_CJ, 'target image channels'),
                  (P.v.shape[0], _ext.MAX_NT, 'time steps nt'))
        for n, nmax, what in limits:
            if n > nmax:
                raise Unsupported(f'the C++ engine supports at most {nmax} {what}, got {n}')
        if min(P.I.shape[1:]) < 2:
            raise Unsupported('the C++ engine needs at least 2 source pixels along each axis')
        self._dim = P.dim
        self._landmarks = P.dim == 2 and P.pointsI.shape[0] > 0
        self._landmarks_exact = self._landmarks and P.landmark_map == 'exact'
        muA, muB, mix = P.muA, P.muB, {}
        if P.mixture is not None:  # the engine gets the initial class means and the options
            m = P.mixture
            muA, muB = m['muA'], m['muB']
            mix = dict(mixture=True, mix_start=m['start'], mix_every=m['every'], mix_classes=m['classes'],
                       mix_pi_min=m['pi_min'], mix_pi_init=list(m['pi_init']),
                       mix_est_muA=P.muA is None and not m['fix_muA'],
                       mix_est_muB=P.muB is None and not m['fix_muB'])
        Engine = _ext.Engine2D if P.dim == 2 else _ext.Engine3D
        self._eng = Engine(
            P.I, P.xI, P.J, P.xJ, P.v, P.xv, P.L, P.T, P.K, P.LL, P.WM, P.WA, P.WB,
            muA=muA, muB=muB,
            pointsI=P.pointsI if self._landmarks else None,
            pointsJ=P.pointsJ if self._landmarks else None,
            sigmaM=P.sigmaM, sigmaA=P.sigmaA, sigmaB=P.sigmaB, sigmaR=P.sigmaR, sigmaP=P.sigmaP,
            epL=P.epL, epT=P.epT, epV=P.epV, diffeo_start=P.diffeo_start, DV=P.DV,
            affine_step_control=bool(P.affine_step_control), step_shrink=P.step_shrink,
            step_grow=P.step_grow, step_floor=P.step_floor, step_cap=P.step_cap,
            num_threads=int(num_threads), landmark_exact=P.landmark_map == 'exact', **mix)
        self._step_control = bool(P.affine_step_control)

    def run(self, n):
        '''Advance n iterations.  The iteration counter persists across calls.'''
        self._eng.run(int(n))

    def snapshot(self):
        '''Current state and the last evaluation, as new numpy arrays owned by the caller.

        See ``run_lddmm`` in ``_lddmm_common`` for the keys.
        '''
        e = self._eng
        snap = {'L': e.L, 'T': e.T, 'A_eval': e.A_eval, 'v': e.v, 'v_eval': e.v_eval,
                'WM': e.WM, 'WB': e.WB, 'WA': e.WA, 'AI': e.AI, 'fAI': e.fAI,
                'Esave': e.Esave[:, :4 if self._landmarks else 3]}
        if self._dim == 3:
            snap['Xs'] = e.Xs
        if self._landmarks:
            snap['pointsIt'] = e.pointsIt
        if self._landmarks_exact:
            snap['landmark_unconverged'] = e.landmark_unconverged
        if self._step_control:
            snap['step_multipliers'] = e.step_multipliers
        return snap

    def gradient(self):
        '''Gradients at the current state, without updating it (for tests).

        Returns
        -------
        dict
            'Lgrad', 'Tgrad': dE/dL and dE/dT; 'vgrad': d(EM + EP)/dv; 'direction': K * dE/dv
            including the regularizer (the direction the velocity update subtracts, times epV);
            'EM', 'ER', 'EP' (None without landmarks).
        '''
        g = self._eng.gradient()
        if not self._landmarks:
            g['EP'] = None
        return g
