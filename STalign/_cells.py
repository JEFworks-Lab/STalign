'''Alignment of two sets of cells (align_cells; see its docstring).

The cells are rasterized into density images at a fine pixel size with a kernel of a fixed
physical width, and LDDMM is run coarse to fine on mean-pooled copies of these images: each
level starts from the affine transform and velocity field of the previous one.
'''
import inspect

import numpy as np
import torch

try:
    from ._lddmm import LDDMM
    from ._prefit import pool_image
    from ._raster import rasterize
except ImportError:
    from _lddmm import LDDMM
    from _prefit import pool_image
    from _raster import rasterize


def align_cells(xI, yI, xJ, yJ, dx=15.0, blur_um=30.0, levels=((4, 500), (2, 500)), *, mixture='em',
                **lddmm_kwargs):
    ''' Align two sets of cells with LDDMM on density images, coarse to fine.

    Both sets of cells are rasterized with rasterize at pixel size dx and a Gaussian kernel of
    standard deviation blur_um (each image covers its own cells). LDDMM is then run on these
    images mean-pooled by the factor of each level in turn, from coarse to fine. Each level
    starts from the previous level's affine transform and velocity field, so the result is a
    single map from source to target that the transform functions use as LDDMM's.

    Parameters
    ----------
    xI, yI : numpy arrays of length N
        x and y locations of the source cells.
    xJ, yJ : numpy arrays of length M
        x and y locations of the target cells, in the same units as the source cells
        (for example micrometers).
    dx : float
        Pixel size of the density images, in the units of the cell locations (default 15.0).
    blur_um : float
        Standard deviation of the rasterization kernel, in the units of the cell locations
        (default 30.0, i.e. 2 pixels of 15). See rasterize.
    levels : sequence of (factor, niter) pairs
        The pyramid, coarsest level first: at each level LDDMM runs niter iterations on the
        images mean-pooled by the integer factor (the source by the factor that gives it about
        the same pixel size as the target under the current affine). The default
        ((4, 500), (2, 500)) runs 500 iterations at 4*dx = 60 and then 500 at 2*dx = 30.
    mixture : 'em', None or dict, keyword only
        LDDMM's mixture argument, passed to every level. The default is 'em', unlike LDDMM's
        (None): LDDMM's mixture EM, which estimates the noise of the matching term from the
        residual and gives a low matching weight to target pixels that look like background
        where the deformed source has tissue, as where the target section lacks cells that
        the source has. mixture=None turns it off (LDDMM's weights of earlier versions, with
        the fixed variances sigmaM^2, sigmaA^2 and sigmaB^2); a dict selects 'em' with some of
        its options changed (see LDDMM). The value you pass is used as given.
        On 64 synthetic cell-to-cell test problems (see CHANGES.md), the default
        cut the geometric mean error of the 8 partial sections from 15.2 to 4.3 um (all 8
        better), but raised that of the 6 sections of uneven cell density from 7.9 to 12.8 um
        (by up to 99%) and made 5 of 8 serial heart sections 10-57% worse; the other 42
        changed by less than 2%. Pass mixture=None where the target section is complete.
    **lddmm_kwargs
        Keyword arguments of LDDMM, for example pointsI and pointsJ, sigmaM, a, num_threads
        or backend. They apply at every level, except as noted below. Everything else
        takes LDDMM's defaults at each level, so the parameters that LDDMM derives from the
        images when they are not given are derived at each level from its pooled images.

        * niter: set by levels (passing niter is an error).
        * L, T or A, v and xv (the initial state), prefit and diffeo_start: apply to the
          first level, which is an ordinary LDDMM call. Each later level starts from the
          previous level's A and v on the same velocity grid xv, without a pre-fit, and
          deforms from its first iteration.
        * sigmaP (only with landmarks): multiplied by the level's pooling factor, so that the
          balance between the landmark and the image terms does not change with the
          number of pixels.
        * draw: applies to the last level; the other levels are not drawn.

    Returns
    -------
    out : dict
        LDDMM's result for the last level ('A', 'v', 'xv', 'WM', 'WB', 'WA', ...), with two
        more entries:

        'rasters' : dict
            The density images and their pixel locations: 'XI', 'YI', 'I' for the source and
            'XJ', 'YJ', 'J' for the target, as returned by rasterize. The fitted map takes
            [YI, XI] to [YJ, XJ] (row, column order, as LDDMM's xI and xJ).
        'levels' : list of dict
            One per level: 'factor' and 'source_factor' (the pooling factors of the target
            and the source images), 'niter', and the pooled pixel locations 'xI' and 'xJ'.
            The weights 'WM', 'WB' and 'WA' are on the last level's 'xJ'.

        If LDDMM reports its affine pre-fit ('prefit'), the record is the first level's.

    Raises
    ------
    ValueError
        If the cell coordinates do not match in length, levels is empty or invalid, or a
        level pools an image to fewer than 2 pixels along an axis.
    TypeError
        If niter is passed.

    Notes
    -----
    Re-rasterizing at a fine pixel size with a fixed physical kernel and fitting a two level
    pyramid was the most accurate configuration on our synthetic test problems (see
    CHANGES.md). Images from other sources (H&E, Visium, an atlas) go to LDDMM directly.

    The matching weights are not carried over from one level to the next: each level starts
    from LDDMM's initial weights.

    Examples
    --------
    >>> out = STalign.align_cells(xI, yI, xJ, yJ, L=L, T=T)
    >>> tpointsI = STalign.transform_points_source_to_target(out['xv'], out['v'], out['A'],
    ...                                                      np.stack([yI, xI], 1))
    '''
    if 'niter' in lddmm_kwargs:
        raise TypeError('align_cells sets the iterations of each level from levels; do not pass niter')
    levels = _check_levels(levels)
    for name, x, y in (('source', xI, yI), ('target', xJ, yJ)):
        if np.ndim(x) != 1 or np.shape(x) != np.shape(y):
            raise ValueError(f'the {name} cell coordinates must be two 1D arrays of the same length, got shapes '
                             f'{np.shape(x)} and {np.shape(y)}')

    XI, YI, I = rasterize(xI, yI, dx=dx, blur_um=blur_um, draw=0)
    XJ, YJ, J = rasterize(xJ, yJ, dx=dx, blur_um=blur_um, draw=0)
    xI_, xJ_ = [YI, XI], [YJ, XJ]

    defaults = {k: p.default for k, p in inspect.signature(LDDMM).parameters.items()
                if p.default is not inspect.Parameter.empty}
    kw = dict(lddmm_kwargs, mixture=mixture)
    first = {k: kw.pop(k) for k in ('L', 'T', 'A', 'v', 'xv', 'prefit', 'diffeo_start') if k in kw}
    draw = kw.pop('draw', defaults['draw'])
    landmarks = kw.get('pointsI') is not None or kw.get('pointsJ') is not None
    if landmarks:
        sigmaP = kw.pop('sigmaP', defaults['sigmaP'])

    linear = _linear_part(first)
    results, info = [], []
    for li, (f, n) in enumerate(levels):
        fI = max(1, int(round(f/np.sqrt(abs(np.linalg.det(linear))))))
        xIc, Ic = _pool(xI_, I, fI)
        xJc, Jc = _pool(xJ_, J, f)
        lvl = dict(kw, niter=n, draw=draw if li == len(levels) - 1 else False)
        if landmarks:
            lvl['sigmaP'] = sigmaP*f
        if li == 0:
            lvl.update(first)
        else:
            prev = results[-1]
            lvl.update(A=prev['A'], v=prev['v'], xv=prev['xv'], diffeo_start=0, prefit=False)
        results.append(LDDMM(xIc, Ic, xJc, Jc, **lvl))
        linear = _numpy(results[-1]['A'])[:2, :2]
        info.append(dict(factor=f, source_factor=fI, niter=n, xI=xIc, xJ=xJc))

    out = dict(results[-1])
    if 'prefit' in results[0]:
        out['prefit'] = results[0]['prefit']
    out['rasters'] = dict(XI=XI, YI=YI, I=I, XJ=XJ, YJ=YJ, J=J)
    out['levels'] = info
    return out


def _check_levels(levels):
    ''' levels as a list of (factor, niter) int pairs, factor >= 1 and niter >= 0. '''
    try:
        pairs = [tuple(p) for p in levels]
        out = [(int(f), int(n)) for f, n in pairs]
        ok = bool(out) and out == pairs and all(f >= 1 and n >= 0 for f, n in out)
    except (TypeError, ValueError):
        ok = False
    if not ok:
        raise ValueError(f'levels must be a non-empty sequence of (factor, niter) pairs of integers with '
                         f'factor >= 1 and niter >= 0, got {levels!r}')
    return out


def _numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _linear_part(first):
    ''' The linear part of the initial affine given by L or A (the identity if neither). '''
    if first.get('A') is not None:
        return np.asarray(_numpy(first['A']), dtype=np.float64)[:2, :2]
    if first.get('L') is not None:
        return np.asarray(_numpy(first['L']), dtype=np.float64).reshape(2, 2)
    return np.eye(2)


def _pool(x, I, f):
    ''' Mean-pool the image I (C, H, W) with pixel locations x = [rows, cols] by the factor f
    (_prefit.pool_image, which pads each axis at its end with the edge values to a multiple of
    f), after checking that every axis keeps at least 2 pixels.
    '''
    n = [len(xi) for xi in x]
    if any(-(-m // f) < 2 for m in n):
        raise ValueError(f'a pooling factor of {f} leaves fewer than 2 pixels along an axis of a {n[0]} x {n[1]} '
                         f'pixel image: use smaller factors in levels, or a smaller dx')
    return pool_image(x, I, f)
