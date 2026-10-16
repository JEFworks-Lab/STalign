"""Tests of the fast rasterize / rasterize_with_signal / normalize (STalign/_raster.py).

The fast implementation is compared with `reference_rasterize` below: the
original per-cell loop of STalign (before the speedup) with the approved fixes
applied (abs only for wavelet_magnitude; normalize of a constant gives t_min).
Criterion: identical grids, shape, layout and NaN / inf pattern, and finite
values within 1e-12 * max|W|.  When the frozen original (version 1.0's
STalign/STalign.py) is available, the reference is also checked against it bit
for bit.  It is the file named by the environment variable STALIGN_ORIGINAL,
for example from `git show b2068ed:STalign/STalign.py` (as the CI workflow
does), or else dev/ref/stalign_orig.py.
"""
import importlib.util
import os
import sys
import tracemalloc
import warnings

import numpy as np
import pytest

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from STalign import STalign  # noqa: E402
from STalign import _raster  # noqa: E402

TOL = 1e-12


# ---------------------------------------------------------------------------
# reference: the original per-cell loop (draw removed), with the fixes
# ---------------------------------------------------------------------------
def reference_normalize(arr, t_min=0, t_max=1):
    diff = t_max - t_min
    diff_arr = np.max(arr) - np.min(arr)
    min_ = np.min(arr)
    if diff_arr == 0:
        diff_arr = 1
    return ((arr - min_)/diff_arr * diff) + t_min


def reference_rasterize(x, y, g=np.ones(1), dx=30.0, blur=1.0, expand=1.1, wavelet_magnitude=False,
                        use_windowing=True, s=None, signal=False):
    """rasterize (signal=False) or rasterize_with_signal (signal=True, with s)."""
    if not isinstance(blur, list):
        blur = [blur]
    nb = len(blur)
    blur = np.array(blur)
    maxblur = np.max(blur)
    if s is not None:
        s = np.array(s)
        if s.ndim == 1:
            s = s[..., None]
    minx, maxx, miny, maxy = np.min(x), np.max(x), np.min(y), np.max(y)
    minx, maxx = (minx+maxx)/2.0 - (maxx-minx)/2.0*expand, (minx+maxx)/2.0 + (maxx-minx)/2.0*expand
    miny, maxy = (miny+maxy)/2.0 - (maxy-miny)/2.0*expand, (miny+maxy)/2.0 + (maxy-miny)/2.0*expand
    X_ = np.arange(minx, maxx, dx)
    Y_ = np.arange(miny, maxy, dx)
    X = np.stack(np.meshgrid(X_, Y_))
    W = np.zeros((X.shape[1], X.shape[2], nb if s is None else s.shape[1]))
    if not signal:
        g = np.resize(g, len(x))
        if not (g == 1.0).all():
            g = reference_normalize(g)
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        for count, (x_, y_) in enumerate(zip(x, y)):
            if not use_windowing:
                rows, cols = slice(None), slice(None)
            else:
                r = int(np.ceil(maxblur*4))
                col = np.round((x_ - X_[0])/dx).astype(int)
                row = np.round((y_ - Y_[0])/dx).astype(int)
                row0 = np.minimum(np.maximum(row - r, 0), W.shape[0]-1)
                row1 = np.minimum(np.maximum(row + r, 0), W.shape[0]-1)
                col0 = np.minimum(np.maximum(col - r, 0), W.shape[1]-1)
                col1 = np.minimum(np.maximum(col + r, 0), W.shape[1]-1)
                rows, cols = slice(row0, row1+1), slice(col0, col1+1)
            k = np.exp(- ((X[0][rows, cols, None] - x_)**2 + (X[1][rows, cols, None] - y_)**2)/(2.0*(dx*blur*2)**2))
            if signal:
                k /= np.sum(k, axis=(0, 1), keepdims=True)*dx**2
            else:
                k /= np.sum(k, axis=(0, 1), keepdims=True)
                k *= g[count]
            if wavelet_magnitude:
                for i in reversed(range(nb)):
                    if i == 0:
                        continue
                    k[..., i] = k[..., i] - k[..., i-1]
            if signal:
                k = k * (1.0 if s is None else s[count])
            W[rows, cols, :] += k
    if wavelet_magnitude:
        W = np.abs(W)
    return X_, Y_, W.transpose((-1, 0, 1))


def reference_signal(x, y, s=None, **kw):
    return reference_rasterize(x, y, s=s, signal=True, **kw)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def assert_same(out, ref, tol=TOL):
    np.testing.assert_array_equal(out[0], ref[0])
    np.testing.assert_array_equal(out[1], ref[1])
    assert out[0].dtype == ref[0].dtype and out[1].dtype == ref[1].dtype
    W, R = out[2], ref[2]
    assert W.shape == R.shape and W.dtype == R.dtype
    assert W.strides == R.strides  # a transposed (H, W, C) array, like the original
    np.testing.assert_array_equal(np.isnan(W), np.isnan(R))
    np.testing.assert_array_equal(np.isposinf(W), np.isposinf(R))
    np.testing.assert_array_equal(np.isneginf(W), np.isneginf(R))
    fin = np.isfinite(R)
    if fin.any():
        scale = np.max(np.abs(R[fin]))
        err = np.max(np.abs(W[fin] - R[fin]))
        assert err <= tol*max(scale, 1e-300), f'max abs error {err:.3e}, max |W| {scale:.3e}'


def quiet(fn, *args, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return fn(*args, **kw)


def cells(n, seed=0, extent=3000.0, dtype=np.float64):
    """Clustered, tissue-like cell coordinates."""
    rng = np.random.default_rng(seed)
    c = rng.uniform(0, extent, (12, 2))
    k = rng.integers(0, 12, n)
    xy = c[k] + rng.normal(0, extent/10, (n, 2))
    return xy[:, 0].astype(dtype), xy[:, 1].astype(dtype)


def frozen_original():
    '''Version 1.0's STalign.py, if available (STALIGN_ORIGINAL or dev/ref/stalign_orig.py).'''
    path = os.environ.get('STALIGN_ORIGINAL') or os.path.join(REPO, 'dev', 'ref', 'stalign_orig.py')
    if not os.path.exists(path):
        pytest.skip('the frozen original STalign.py is not available (set STALIGN_ORIGINAL)')
    module = sys.modules.get('stalign_orig')
    if module is None or os.path.abspath(getattr(module, '__file__', '')) != os.path.abspath(path):
        spec = importlib.util.spec_from_file_location('stalign_orig', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules['stalign_orig'] = module
    return module


# ---------------------------------------------------------------------------
# rasterize: exactness
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('dx,blur', [(30.0, 1.5), (10.0, 1.0), (25.0, 3.3), (30.0, 0.5)])
def test_density(dx, blur):
    x, y = cells(1500, seed=1)
    assert_same(STalign.rasterize(x, y, dx=dx, blur=blur, draw=0), reference_rasterize(x, y, dx=dx, blur=blur))


@pytest.mark.parametrize('wavelet', [False, True])
def test_multi_blur(wavelet):
    x, y = cells(1500, seed=2)
    kw = dict(dx=30.0, blur=[2.0, 1.0, 0.5], wavelet_magnitude=wavelet)
    assert_same(STalign.rasterize(x, y, draw=0, **kw), reference_rasterize(x, y, **kw))


def test_blur_tuple_or_array_is_multiscale():
    # the original only accepted a list (a tuple or array raised a broadcasting error)
    x, y = cells(500, seed=3)
    ref = STalign.rasterize(x, y, dx=20.0, blur=[1.9, 0.7], draw=0)
    for blur in [(1.9, 0.7), np.array([1.9, 0.7])]:
        out = STalign.rasterize(x, y, dx=20.0, blur=blur, draw=0)
        np.testing.assert_array_equal(out[2], ref[2])
    with pytest.raises(ValueError):
        STalign.rasterize(x, y, blur=-1.0, draw=0)


def test_wavelet_requires_sorted_blur():
    x, y = cells(100, seed=4)
    with pytest.raises(Exception, match='sorted from greatest to least'):
        STalign.rasterize(x, y, blur=[1.0, 2.0], draw=0, wavelet_magnitude=True)


def test_weights():
    x, y = cells(1500, seed=5)
    g = np.random.default_rng(5).gamma(2.0, 3.0, x.size)
    assert_same(STalign.rasterize(x, y, g=g, dx=30.0, blur=1.5, draw=0),
                reference_rasterize(x, y, g=g, dx=30.0, blur=1.5))


def test_signed_weights_wavelet():
    x, y = cells(1200, seed=6)
    g = np.random.default_rng(6).normal(0, 1, x.size)
    kw = dict(g=g, dx=25.0, blur=[1.5, 0.75], wavelet_magnitude=True)
    assert_same(STalign.rasterize(x, y, draw=0, **kw), reference_rasterize(x, y, **kw))


@pytest.mark.parametrize('g', [np.ones(7, dtype=int), np.array([1.0, 3.0, 2.0]), np.arange(5), [1, 2]])
def test_weight_resize(g):
    # a short g is cycled (np.resize); an all-ones g skips normalize
    x, y = cells(800, seed=7)
    assert_same(STalign.rasterize(x, y, g=g, dx=30.0, blur=1.0, draw=0),
                reference_rasterize(x, y, g=g, dx=30.0, blur=1.0))


def test_nan_weight():
    # one NaN makes normalize(g) all NaN, so every window is NaN
    x, y = cells(600, seed=9, extent=6000)
    g = np.random.default_rng(9).random(x.size)
    g[17] = np.nan
    out = quiet(STalign.rasterize, x, y, g=g, dx=30.0, blur=1.0, draw=0)
    assert_same(out, quiet(reference_rasterize, x, y, g=g, dx=30.0, blur=1.0))
    assert np.isnan(out[2]).any() and not np.isnan(out[2]).all()


@pytest.mark.parametrize('expand', [0.9, 0.5, 0.2])
def test_cells_outside_the_image(expand):
    # windows clipped down to 1-pixel windows at the border
    x, y = cells(1200, seed=10)
    assert_same(STalign.rasterize(x, y, dx=30.0, blur=1.5, expand=expand, draw=0),
                reference_rasterize(x, y, dx=30.0, blur=1.5, expand=expand))


def test_kernel_underflow_far_outside():
    # the original's kernel underflows (0/0 = NaN window): exact per-cell path
    rng = np.random.default_rng(11)
    x = np.concatenate([rng.uniform(0, 1000, 300), [-30000.0, 31000.0, 500.0, -1400.0]])
    y = np.concatenate([rng.uniform(0, 1000, 300), [500.0, -20000.0, 40000.0, 500.0]])
    out = quiet(STalign.rasterize, x, y, dx=30.0, blur=1.0, expand=0.05, draw=0)
    assert np.isnan(out[2]).any()
    assert_same(out, quiet(reference_rasterize, x, y, dx=30.0, blur=1.0, expand=0.05))


def test_pixel_centers_and_half_pixels():
    # cells on pixel centers and half-pixel positions (np.round rounds half to even), on the border
    g = np.arange(0, 900.0, 15.0)
    X, Y = np.meshgrid(g, g[:40])
    x, y = X.ravel(), Y.ravel()
    for blur in (1.0, 0.6):
        assert_same(STalign.rasterize(x, y, dx=30.0, blur=blur, expand=1.0, draw=0),
                    reference_rasterize(x, y, dx=30.0, blur=blur, expand=1.0))


@pytest.mark.parametrize('blur', [0.01, 0.0])
def test_degenerate_blur(blur):
    # blur so small that every kernel underflows (exact path for all cells)
    x, y = cells(300, seed=16)
    assert_same(quiet(STalign.rasterize, x, y, dx=30.0, blur=blur, draw=0),
                quiet(reference_rasterize, x, y, dx=30.0, blur=blur))


def test_large_blur_small_image():
    # windows larger than the image (clipped on all sides)
    x, y = cells(300, seed=15, extent=500)
    assert_same(STalign.rasterize(x, y, dx=50.0, blur=4.0, draw=0), reference_rasterize(x, y, dx=50.0, blur=4.0))


@pytest.mark.parametrize('wavelet', [False, True])
def test_legacy_whole_image(wavelet):
    x, y = cells(150, seed=19, extent=1500)
    g = np.random.default_rng(19).random(x.size)
    kw = dict(g=g, dx=60.0, blur=[1.5, 1.0], wavelet_magnitude=wavelet, use_windowing=False)
    assert_same(STalign.rasterize(x, y, draw=0, **kw), reference_rasterize(x, y, **kw))


def test_few_cells():
    for x, y in [(np.array([0.0, 100.0]), np.array([0.0, 50.0])),
                 (np.array([0.0, 100.0, 37.0]), np.array([0.0, 50.0, 12.0]))]:
        assert_same(STalign.rasterize(x, y, dx=10.0, blur=1.0, draw=0), reference_rasterize(x, y, dx=10.0, blur=1.0))
        kw = dict(g=np.array([3.0, 1.0, 2.0])[:x.size], dx=7.0, blur=[1.0, 0.5], wavelet_magnitude=True)
        assert_same(STalign.rasterize(x, y, draw=0, **kw), reference_rasterize(x, y, **kw))


@pytest.mark.parametrize('use_windowing', [True, False])
def test_empty_image_raises_index_error(use_windowing):
    # a single cell or collinear cells give an empty grid (IndexError, as before)
    for x, y in [(np.array([5.0]), np.array([7.0])), (np.array([5.0, 90.0, 300.0]), np.array([7.0, 7.0, 7.0]))]:
        with pytest.raises(IndexError, match='grid is empty'):
            STalign.rasterize(x, y, dx=30.0, draw=0, use_windowing=use_windowing)
        with pytest.raises(IndexError, match='grid is empty'):
            STalign.rasterize_with_signal(x, y, dx=30.0, use_windowing=use_windowing)


def test_dense_tile():
    # thousands of cells in one tile (tile padding to multiples of _CHUNK)
    rng = np.random.default_rng(13)
    x = np.concatenate([rng.normal(500, 20, 3000), rng.uniform(0, 3000, 300)])
    y = np.concatenate([rng.normal(800, 20, 3000), rng.uniform(0, 3000, 300)])
    assert_same(STalign.rasterize(x, y, dx=30.0, blur=1.5, draw=0), reference_rasterize(x, y, dx=30.0, blur=1.5))


def test_small_batches(monkeypatch):
    # many batches (split inside tiles and by the tile budget), tiny tiles and chunks
    x, y = cells(2000, seed=14)
    s = np.random.default_rng(14).random((x.size, 5))
    monkeypatch.setattr(_raster, '_BATCH_CELLS', 100)
    monkeypatch.setattr(_raster, '_BATCH_BYTES', 1 << 14)
    monkeypatch.setattr(_raster, '_CHUNK', 3)
    monkeypatch.setattr(_raster, '_TILE', 1)
    monkeypatch.setattr(_raster, '_CHAN_GROUP', 2)
    kw = dict(dx=30.0, blur=[1.5, 0.5], wavelet_magnitude=True)
    assert_same(STalign.rasterize(x, y, draw=0, **kw), reference_rasterize(x, y, **kw))
    assert_same(STalign.rasterize_with_signal(x, y, s=s, dx=30.0, blur=1.0),
                reference_signal(x, y, s=s, dx=30.0, blur=1.0))
    assert_same(STalign.rasterize(x[:200], y[:200], dx=90.0, blur=1.0, use_windowing=False, draw=0),
                reference_rasterize(x[:200], y[:200], dx=90.0, blur=1.0, use_windowing=False))


@pytest.mark.parametrize('dtype', [np.float32, np.int64])
def test_coordinate_dtypes(dtype):
    x, y = cells(1000, seed=17, dtype=dtype)
    assert_same(STalign.rasterize(x, y, dx=30.0, blur=1.5, draw=0), reference_rasterize(x, y, dx=30.0, blur=1.5))


def test_mismatched_lengths():
    # cells are zip(x, y, g): extra coordinates only enter the grid
    x, y = cells(700, seed=22)
    g = np.random.default_rng(22).random(800)
    assert_same(STalign.rasterize(x, y[:650], g=g, dx=30.0, blur=1.0, draw=0),
                reference_rasterize(x, y[:650], g=g, dx=30.0, blur=1.0))
    assert_same(STalign.rasterize_with_signal(x[:600], y, s=g[:600], dx=30.0, blur=1.0),
                reference_signal(x[:600], y, s=g[:600], dx=30.0, blur=1.0))


def test_pandas_and_list_inputs():
    pd = pytest.importorskip('pandas')
    x, y = cells(500, seed=18)
    ref = reference_rasterize(x, y, dx=30.0, blur=1.5)
    assert_same(STalign.rasterize(pd.Series(x), pd.Series(y), dx=30.0, blur=1.5, draw=0), ref)
    assert_same(STalign.rasterize(list(x), list(y), dx=30.0, blur=1.5, draw=0), ref)


def test_pandas_nan_coordinates_are_ignored():
    # pandas min/max skip NaN, so the grid comes from the other cells; the NaN cells are left out
    pd = pytest.importorskip('pandas')
    x, y = cells(500, seed=23)
    xn, yn = x.copy(), y.copy()
    xn[[3, 100]] = np.nan
    yn[7] = np.nan
    keep = np.isfinite(xn) & np.isfinite(yn)
    with pytest.warns(UserWarning, match='3 cells with NaN coordinates'):
        out = STalign.rasterize(pd.Series(xn), pd.Series(yn), dx=30.0, blur=1.5, draw=0)
    assert_same(out, reference_rasterize(x[keep], y[keep], dx=30.0, blur=1.5))


def test_draw():
    x, y = cells(300, seed=20)
    ref = reference_rasterize(x, y, dx=30.0, blur=[1.0, 0.5])
    for draw in (True, 100):
        out = STalign.rasterize(x, y, dx=30.0, blur=[1.0, 0.5], draw=draw)
        assert len(out) == 4 and isinstance(out[3], matplotlib.figure.Figure)
        assert_same(out[:3], ref)
    out = STalign.rasterize(x, y)  # default draw=10000 -> figure
    assert len(out) == 4
    assert len(STalign.rasterize(x, y, draw=False)) == 3
    assert len(STalign.rasterize_with_signal(x, y)) == 3  # default draw=0
    assert len(STalign.rasterize_with_signal(x, y, draw=1)) == 4
    plt.close('all')


def test_memory_sparse_cells_on_a_large_image(monkeypatch):
    # sparse cells on a large image: temporaries are bounded by _BATCH_BYTES,
    # not by the number of tiles (peak = image buffer + output + small batch)
    monkeypatch.setattr(_raster, '_BATCH_BYTES', 1 << 22)
    rng = np.random.default_rng(24)
    x, y = rng.uniform(0, 60000, 3000), rng.uniform(0, 60000, 3000)
    tracemalloc.start()
    try:
        out = STalign.rasterize(x, y, dx=30.0, blur=4.0, draw=0)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < 2.6*out[2].nbytes, f'peak {peak/1e6:.0f} MB for a {out[2].nbytes/1e6:.0f} MB image'


# ---------------------------------------------------------------------------
# rasterize_with_signal
# ---------------------------------------------------------------------------
def test_signal_none():
    x, y = cells(1200, seed=30)
    assert_same(STalign.rasterize_with_signal(x, y, dx=30.0, blur=1.5), reference_signal(x, y, dx=30.0, blur=1.5))


def test_signal_none_multiblur_wavelet():
    x, y = cells(1200, seed=31)
    kw = dict(dx=20.0, blur=[2.0, 1.0], wavelet_magnitude=True)
    assert_same(STalign.rasterize_with_signal(x, y, **kw), reference_signal(x, y, **kw))


@pytest.mark.parametrize('M', [1, 3])
@pytest.mark.parametrize('wavelet', [False, True])
def test_signal_columns(M, wavelet):
    # wavelet_magnitude with a signal only takes the absolute value (one blur): one channel per column
    x, y = cells(1200, seed=33)
    s = np.random.default_rng(33).poisson(3.0, (x.size, M))  # int counts
    kw = dict(s=s if M > 1 else s[:, 0], dx=25.0, blur=1.2, wavelet_magnitude=wavelet)
    out = STalign.rasterize_with_signal(x, y, **kw)
    assert out[2].shape[0] == M
    assert_same(out, reference_signal(x, y, **kw))


@pytest.mark.parametrize('use_windowing', [True, False])
def test_signal_columns_signed_wavelet(use_windowing):
    x, y = cells(300, seed=34, extent=2000)
    s = np.random.default_rng(34).normal(0, 1, (x.size, 3))
    kw = dict(s=s, dx=60.0, blur=1.5, wavelet_magnitude=True, use_windowing=use_windowing)
    out = STalign.rasterize_with_signal(x, y, **kw)
    assert_same(out, reference_signal(x, y, **kw))
    assert (out[2] >= 0).all()


def test_signal_many_columns():
    # more columns than one matrix product group (_CHAN_GROUP)
    x, y = cells(800, seed=39)
    s = np.random.default_rng(39).gamma(1.0, 1.0, (x.size, 21))
    kw = dict(s=s, dx=30.0, blur=1.5, expand=0.8)
    assert_same(STalign.rasterize_with_signal(x, y, **kw), reference_signal(x, y, **kw))


def test_signal_keeps_sign():
    # the image of a signed signal keeps its sign: W(-s) == -W(s), W(s) == W(s+) - W(s-)
    x, y = cells(1000, seed=35)
    s = np.random.default_rng(35).normal(0, 1, (x.size, 2))
    out = STalign.rasterize_with_signal(x, y, s=s, dx=30.0, blur=1.5)
    assert_same(out, reference_signal(x, y, s=s, dx=30.0, blur=1.5))
    assert (out[2] < 0).any()
    neg = STalign.rasterize_with_signal(x, y, s=-s, dx=30.0, blur=1.5)
    np.testing.assert_allclose(neg[2], -out[2], rtol=0, atol=1e-14*np.abs(out[2]).max())
    pos = STalign.rasterize_with_signal(x, y, s=np.maximum(s, 0), dx=30.0, blur=1.5)[2]
    pos -= STalign.rasterize_with_signal(x, y, s=np.maximum(-s, 0), dx=30.0, blur=1.5)[2]
    np.testing.assert_allclose(pos, out[2], rtol=0, atol=1e-13*np.abs(out[2]).max())


def test_signal_nan_and_inf():
    x, y = cells(600, seed=36)
    s = np.random.default_rng(36).random((x.size, 3))
    s[5, 1] = np.nan
    s[40, 0] = np.inf
    s[41, 2] = -np.inf
    s[42] = [np.nan, np.inf, 1.0]
    out = quiet(STalign.rasterize_with_signal, x, y, s=s, dx=30.0, blur=1.0)
    assert np.isnan(out[2]).any() and np.isinf(out[2]).any()
    assert_same(out, quiet(reference_signal, x, y, s=s, dx=30.0, blur=1.0))


def test_signal_rows_and_errors():
    x, y = cells(400, seed=37)
    s = np.random.default_rng(37).random((x.size + 10, 2))
    # extra rows are ignored; lists are accepted
    assert_same(STalign.rasterize_with_signal(x, y, s=s.tolist(), dx=30.0, blur=1.0),
                reference_signal(x, y, s=s, dx=30.0, blur=1.0))
    with pytest.raises(IndexError):
        STalign.rasterize_with_signal(x, y, s=s[:100], dx=30.0, blur=1.0)
    with pytest.raises(Exception, match='only have one blur'):
        STalign.rasterize_with_signal(x, y, s=s, blur=[2.0, 1.0])


# ---------------------------------------------------------------------------
# blur_um: the kernel width in the units of x and y
# ---------------------------------------------------------------------------
def test_default_blur_is_one():
    x, y = cells(800, seed=50)
    s = np.random.default_rng(50).random((x.size, 2))
    for fn, kw in ((STalign.rasterize, {}), (STalign.rasterize_with_signal, {}),
                   (STalign.rasterize_with_signal, dict(s=s))):
        out = fn(x, y, dx=30.0, draw=0, **kw)
        for a, b in zip(out, fn(x, y, dx=30.0, blur=1.0, draw=0, **kw)):
            np.testing.assert_array_equal(a, b)
        for a, b in zip(out, fn(x, y, dx=30.0, blur=None, blur_um=None, draw=0, **kw)):
            np.testing.assert_array_equal(a, b)
    assert_same(STalign.rasterize(x, y, draw=0), reference_rasterize(x, y, blur=1.0))


@pytest.mark.parametrize('dx,blur_um', [(15.0, 30.0), (30.0, 45.0), (10.0, [40.0, 20.0]), (7.0, 13.0)])
def test_blur_um_is_blur_in_pixels(dx, blur_um):
    # bit for bit the call with blur = blur_um/(2*dx)
    x, y = cells(1500, seed=51)
    s = np.random.default_rng(51).random(x.size) - 0.5
    blur = list(np.asarray(blur_um)/(2*dx)) if isinstance(blur_um, list) else blur_um/(2*dx)
    calls = [(STalign.rasterize, dict(draw=0)), (STalign.rasterize_with_signal, {})]
    if isinstance(blur_um, list):
        calls.append((STalign.rasterize, dict(draw=0, wavelet_magnitude=True)))
    else:
        calls.append((STalign.rasterize_with_signal, dict(s=s, use_windowing=False)))
    for fn, kw in calls:
        a = fn(x, y, dx=dx, blur_um=blur_um, **kw)
        b = fn(x, y, dx=dx, blur=blur, **kw)
        assert len(a) == len(b) == 3
        for u, v in zip(a, b):
            np.testing.assert_array_equal(u, v)
    if dx == 15.0:  # 30 um at 15 um pixels is the default blur of 1 pixel
        a = STalign.rasterize(x, y, dx=dx, blur_um=blur_um, draw=0)
        assert_same(a, reference_rasterize(x, y, dx=dx, blur=1.0))


def test_blur_um_is_the_kernel_standard_deviation():
    # one cell at a pixel center, whole-image kernel: the image's standard deviation is blur_um
    # at any pixel size (the two other cells have weight 0 and only set the image extent)
    x = np.array([0.0, 1500.0, 3000.0])
    y = np.array([0.0, 1500.0, 3000.0])
    for dx in (10.0, 15.0, 30.0):
        X, Y, W = STalign.rasterize(x, y, g=np.array([0.0, 1.0, 0.0]), dx=dx, blur_um=60.0, draw=0,
                                    use_windowing=False)
        w = W[0]/W[0].sum()
        mx, my = (w.sum(0)*X).sum(), (w.sum(1)*Y).sum()
        assert abs(mx - 1500.0) < 1e-6 and abs(my - 1500.0) < 1e-6
        sx = np.sqrt((w.sum(0)*(X - mx)**2).sum())
        sy = np.sqrt((w.sum(1)*(Y - my)**2).sum())
        assert abs(sx - 60.0) < 1e-6 and abs(sy - 60.0) < 1e-6, (dx, sx, sy)


def test_blur_and_blur_um_together_raise():
    x, y = cells(100, seed=52)
    for fn in (STalign.rasterize, STalign.rasterize_with_signal):
        for blur in (1.0, 2.0, [2.0, 1.0]):
            with pytest.raises(ValueError, match='not both'):
                fn(x, y, dx=15.0, blur=blur, blur_um=30.0, draw=0)
    with pytest.raises(ValueError, match='non-negative'):
        STalign.rasterize(x, y, dx=15.0, blur_um=-30.0, draw=0)


# ---------------------------------------------------------------------------
# normalize
# ---------------------------------------------------------------------------
def test_normalize():
    a = np.array([[3.0, 1.0], [2.0, 5.0]])
    np.testing.assert_array_equal(STalign.normalize(a), (a - 1.0)/4.0)
    np.testing.assert_array_equal(STalign.normalize(a, 2, 4), (a - 1.0)/4.0*2 + 2)
    # constant input maps to t_min (it used to be 0/0 = NaN)
    for c in (np.zeros(5), np.full((2, 3), 7.0), np.full(4, 3, dtype=int)):
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            out = STalign.normalize(c, -1, 1)
        assert out.shape == c.shape and (out == -1).all()
    assert np.isnan(STalign.normalize(np.array([1.0, np.nan]))).all()


def test_constant_weights_give_zero_image():
    # normalize(constant g) = 0, so a constant g other than 1 gives zeros (it used to give NaN windows)
    x, y = cells(500, seed=8)
    for g in (np.zeros(x.size), np.full(x.size, 2.0)):
        out = STalign.rasterize(x, y, g=g, dx=30.0, blur=1.0, draw=0)
        assert (out[2] == 0).all()
        assert_same(out, reference_rasterize(x, y, g=g, dx=30.0, blur=1.0))


# ---------------------------------------------------------------------------
# the reference is the original code (bit for bit, where no fix applies)
# ---------------------------------------------------------------------------
def test_reference_is_the_original():
    S = frozen_original()
    x, y = cells(400, seed=40)
    g = np.random.default_rng(40).gamma(2.0, 1.0, x.size)
    s = np.random.default_rng(41).random((x.size, 2))
    cases = [
        (S.rasterize, reference_rasterize, dict(g=g, dx=30.0, blur=[2.0, 1.0], wavelet_magnitude=True)),
        (S.rasterize, reference_rasterize, dict(dx=30.0, blur=1.5, expand=0.5)),
        (S.rasterize, reference_rasterize, dict(g=g, dx=90.0, blur=1.0, use_windowing=False)),
        (S.rasterize_with_signal, reference_signal, dict(s=s, dx=30.0, blur=1.5)),
        (S.rasterize_with_signal, reference_signal, dict(dx=30.0, blur=[1.5, 0.5], wavelet_magnitude=True)),
    ]
    for orig, ref, kw in cases:
        a = orig(x, y, draw=0, **kw)
        b = ref(x, y, **kw)
        for u, v in zip(a, b):
            np.testing.assert_array_equal(u, v)
    # where the fixes apply: abs of a signed signal, NaN for a constant g
    s = s - 0.5
    np.testing.assert_array_equal(S.rasterize_with_signal(x, y, s=s, dx=30.0, blur=1.5)[2],
                                  np.abs(reference_signal(x, y, s=s, dx=30.0, blur=1.5)[2]))
    assert np.isnan(quiet(S.rasterize, x, y, g=np.zeros(x.size), draw=0)[2]).any()
