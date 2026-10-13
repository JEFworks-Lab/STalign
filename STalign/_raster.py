''' Fast rasterization of cell coordinates into images.

Implementation of STalign.normalize, STalign.rasterize and
STalign.rasterize_with_signal (see their docstrings for the parameters).  The
results equal those of the original per-cell loop up to floating point
rounding (relative differences ~1e-15).

How it works
------------
Each cell adds a Gaussian evaluated on a (2r+1) x (2r+1) pixel window around
its nearest pixel (clipped at the image border), normalized to unit sum over
that window and multiplied by the cell's weight.  On a rectangular window the
Gaussian and its sum are separable, so a cell contributes exactly

    w * outer(ky, kx) / (sum(ky) * sum(kx))

where ky and kx are the 1D kernels on the cell's row and column windows.  The
image is therefore Ky^T diag(w) Kx, with Ky (cells x rows) and Kx (cells x
columns) banded.  It is evaluated with dense matrix products on tiles:

1. cells are binned into G x G pixel tiles by the start of their window and
   sorted by tile,
2. each cell's 1D kernels (2(2r+1) exponentials instead of (2r+1)^2) are
   written into tile-local frames of F = G + 2r pixels,
3. for each tile, the F x F block Ky^T diag(w) Kx is one small matrix product;
   tiles with the same (padded) number of cells go through one batched
   np.matmul,
4. the overlapping F x F blocks are added into the image (G >= 2r, so a block
   only overlaps the three neighbouring tiles).

Cells are processed in batches that bound the temporary memory.  A few rare
cells are handled with the original per-cell code so that even degenerate
output matches: cells whose kernel underflows (far outside the image, or a
tiny blur), where the original divides 0 by 0, and cells with infinite
weights.  Cells with NaN weights make their window NaN.
'''

import warnings

import numpy as np
from numpy.lib.stride_tricks import as_strided

_TILE = 16              # minimum tile size in pixels (tiles are G = max(_TILE, 2r) pixels)
_CHUNK = 32             # tiles with more cells than this are padded to a multiple of _CHUNK
_BATCH_CELLS = 1 << 16  # maximum number of cells per batch
_BATCH_BYTES = 1 << 26  # bound on the large per-batch temporaries (bytes, each)
_CHAN_GROUP = 16        # signal columns per matrix product
# A cell is handled by the exact per-cell code when the largest kernel value
# in its window is below exp(-_FALLBACK_EXPONENT): the original's 2D kernel
# loses precision to underflow there, the separable one does not.
_FALLBACK_EXPONENT = 600.0


def normalize(arr, t_min=0, t_max=1):
    ''' Linearly map arr to [t_min, t_max] (see STalign.normalize). '''
    diff = t_max - t_min
    diff_arr = np.max(arr) - np.min(arr)
    min_ = np.min(arr)
    if diff_arr == 0:
        # constant input maps to t_min (instead of 0/0 = NaN)
        diff_arr = 1
    norm_arr = ((arr - min_)/diff_arr * diff) + t_min
    return norm_arr


def rasterize(x, y, g=np.ones(1), dx=30.0, blur=None, expand=1.1, draw=10000, wavelet_magnitude=False,
              use_windowing=True, *, blur_um=None):
    ''' Density (or weighted density) image of cells (see STalign.rasterize). '''
    blur = _blurs(_blur_pixels(blur, blur_um, dx))
    _check_wavelet(blur, wavelet_magnitude)
    X_, Y_ = _grid(x, y, dx, expand)
    x = np.asarray(x)
    y = np.asarray(y)
    # g is cycled to the number of cells (np.resize) and min-max normalized
    # unless it is all ones
    g = _resize(g, x.shape[0])
    if not (g == 1.0).all():
        g = normalize(g)
    n = min(x.shape[0], y.shape[0])  # cells are zip(x, y, g)
    w = np.asarray(g[:n], dtype=np.float64).reshape(n, 1)
    W = _accumulate(x[:n], y[:n], w, X_, Y_, dx, blur, wavelet_magnitude, use_windowing, signal=False)
    return _finish(X_, Y_, W, blur.size, draw, wavelet_magnitude)


def rasterize_with_signal(x, y, s=None, dx=30.0, blur=None, expand=1.1, draw=0, wavelet_magnitude=False,
                          use_windowing=True, *, blur_um=None):
    ''' Image of a per-cell signal (see STalign.rasterize_with_signal). '''
    blur = _blurs(_blur_pixels(blur, blur_um, dx))
    if blur.size > 1 and s is not None:
        raise Exception('when using a signal, we can only have one blur')
    if s is not None:
        s = np.array(s)
        if s.ndim == 1:
            s = s[..., None]
    _check_wavelet(blur, wavelet_magnitude)
    X_, Y_ = _grid(x, y, dx, expand)
    x = np.asarray(x)
    y = np.asarray(y)
    n = min(x.shape[0], y.shape[0])  # cells are zip(x, y)
    if s is None:
        w = np.ones((n, 1))
    else:
        if s.shape[0] < n:
            raise IndexError(f's has {s.shape[0]} rows but there are {n} cells')
        w = np.asarray(s[:n], dtype=np.float64).reshape(n, s.shape[1])
    W = _accumulate(x[:n], y[:n], w, X_, Y_, dx, blur, wavelet_magnitude, use_windowing, signal=True)
    return _finish(X_, Y_, W, blur.size, draw, wavelet_magnitude)


# ---------------------------------------------------------------------------
# setup shared by both functions
# ---------------------------------------------------------------------------
def _blur_pixels(blur, blur_um, dx):
    ''' The blur in pixels: blur as given (1.0 if None), or blur_um/(2*dx).

    blur_um is the kernel standard deviation in the units of x and y; the
    kernel standard deviation is 2*blur pixels, i.e. 2*blur*dx in those units.
    '''
    if blur_um is None:
        return 1.0 if blur is None else blur
    if blur is not None:
        raise ValueError('give the kernel width either as blur (in pixels) or as blur_um (in the units of x '
                         'and y), not both')
    return np.asarray(blur_um, dtype=np.float64)/(2.0*dx)


def _blurs(blur):
    ''' Blur(s) as a 1D float array: a list, tuple or array gives one channel per entry. '''
    blur = np.atleast_1d(np.asarray(blur, dtype=np.float64))
    if blur.ndim != 1 or blur.size == 0:
        raise ValueError('blur must be a number or a non-empty list of numbers')
    if np.any(blur < 0):
        raise ValueError('blur must be non-negative')
    return blur


def _check_wavelet(blur, wavelet_magnitude):
    if wavelet_magnitude and np.any(blur != np.sort(blur)[::-1]):
        raise Exception('When using wavelet magnitude, blurs must be sorted from greatest to least')


def _grid(x, y, dx, expand):
    ''' Pixel centers X_, Y_ (expression by expression as in the original).

    np.min / np.max are applied to the inputs as given, so pandas inputs skip
    NaN like they always did.
    '''
    minx = np.min(x)
    maxx = np.max(x)
    miny = np.min(y)
    maxy = np.max(y)
    minx,maxx = (minx+maxx)/2.0 - (maxx-minx)/2.0*expand, (minx+maxx)/2.0 + (maxx-minx)/2.0*expand
    miny,maxy = (miny+maxy)/2.0 - (maxy-miny)/2.0*expand, (miny+maxy)/2.0 + (maxy-miny)/2.0*expand
    X_ = np.arange(minx,maxx,dx)
    Y_ = np.arange(miny,maxy,dx)
    if X_.size == 0 or Y_.size == 0:
        raise IndexError(f'the image grid is empty ({Y_.size} x {X_.size} pixels): the cells must span a '
                         'nonzero extent in x and y, and dx and expand must be positive')
    return X_, Y_


def _resize(a, n):
    ''' np.resize(a, n) for 1D output, without np.resize's slow concatenate of n copies. '''
    a = np.ravel(a)
    if a.size == 0 or n == 0:
        return np.zeros_like(a, shape=(n,))
    if a.size >= n:
        return a[:n].copy()
    return np.tile(a, -(-n // a.size))[:n]


def _windows(x, y, X_, Y_, dx, r):
    ''' First and last row / column of each cell's window, clipped to the image as in the original. '''
    H, Wd = Y_.size, X_.size
    col = np.round((x - X_[0])/dx).astype(int)
    row = np.round((y - Y_[0])/dx).astype(int)
    row0 = np.minimum(np.maximum(row - r, 0), H - 1)
    row1 = np.minimum(np.maximum(row + r, 0), H - 1)
    col0 = np.minimum(np.maximum(col - r, 0), Wd - 1)
    col1 = np.minimum(np.maximum(col + r, 0), Wd - 1)
    return row0, row1, col0, col1


def _channels(B, Cw):
    ''' Number of channels, and the blur and weight column used by each.

    rasterize: Cw == 1, one channel per blur.  rasterize_with_signal: one
    channel per column of s (then B == 1), or per blur if s is None.
    '''
    C = max(B, Cw)
    bsel = np.arange(C) if B > 1 else np.zeros(C, dtype=int)
    wsel = np.arange(C) if Cw > 1 else np.zeros(C, dtype=int)
    return C, bsel, wsel


# ---------------------------------------------------------------------------
# accumulation
# ---------------------------------------------------------------------------
def _accumulate(x, y, w, X_, Y_, dx, blur, wavelet, use_windowing, signal):
    ''' Sum over cells of kernel times weight, as an (H, Wd, C) array (before any abs).

    w is (N, Cw).  signal selects rasterize_with_signal's normalization (each
    kernel divided by its sum times dx**2) over rasterize's (by its sum).
    '''
    H, Wd = Y_.size, X_.size
    B = blur.size
    C, _, _ = _channels(B, w.shape[1])
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    keep = np.isfinite(x) & np.isfinite(y)
    if not keep.all():
        # only possible for inputs whose min/max skip NaN (e.g. pandas)
        warnings.warn(f'ignoring {np.count_nonzero(~keep)} cells with NaN coordinates')
        x, y, w = x[keep], y[keep], w[keep]
    W = np.zeros((H, Wd, C))
    if x.size == 0:
        return W
    r = int(np.ceil(np.max(blur)*4))  # window radius in pixels (the largest blur sets it for all channels)
    den = 2.0*(dx*blur*2)**2           # kernel exp(-d**2/den), standard deviation 2*blur pixels

    # weights: +-inf cells go to the exact per-cell path; NaN entries make the
    # cell's window NaN in the affected channels
    ws = w*(1.0/dx**2) if signal else w
    exact = []
    nan_cells = None
    finite = np.isfinite(ws)
    if not finite.all():
        inf_cells = np.isinf(ws).any(1)
        exact.append(np.flatnonzero(inf_cells))
        nan_cells = np.flatnonzero(~finite.all(1) & ~inf_cells)
        nan_cols = np.isnan(ws[nan_cells])
        ws = np.where(finite, ws, 0.0)
        ws[inf_cells] = 0.0

    if use_windowing:
        raw, underflow = _accumulate_tiles(x, y, ws, X_, Y_, dx, r, den, C)
    else:
        raw, underflow = _accumulate_full(x, y, ws, X_, Y_, den, C)
    W[...] = np.moveaxis(raw, 0, -1)
    if wavelet and B > 1:
        # the original replaces kernel i by kernel i - kernel i-1 for every
        # cell; this is linear, so difference the accumulated channels
        W[..., 1:] -= np.moveaxis(raw[:-1], 0, -1)
    del raw
    exact.append(underflow)
    cells = np.unique(np.concatenate(exact))
    if cells.size:
        # these cells contributed nothing above (zero weight or zeroed kernel)
        _exact_cells(cells, x, y, w, X_, Y_, dx, blur, r, signal, wavelet, use_windowing, W)
    if nan_cells is not None and nan_cells.size:
        _nan_windows(x[nan_cells], y[nan_cells], nan_cols, X_, Y_, dx, r, use_windowing, W)
    return W


def _row_windows(a, L):
    ''' Writable (len(a) - L + 1, L) view of the 1D array a whose row i is a[i:i+L]. '''
    return as_strided(a, shape=(a.shape[0] - L + 1, L), strides=(a.strides[0], a.strides[0]))


def _padded_counts(cnt):
    ''' Slots per tile: cell counts rounded up to a power of two (small tiles) or a multiple of _CHUNK. '''
    pad = -(-cnt//_CHUNK)*_CHUNK
    small = cnt <= _CHUNK
    pad[small] = 2**np.ceil(np.log2(cnt[small])).astype(int)
    return pad


def _accumulate_tiles(x, y, w, X_, Y_, dx, r, den, C):
    ''' Windowed kernels, summed with tiled matrix products.

    Returns raw (C, H, Wd) and the indices of the cells whose kernel
    underflows (they contribute nothing to raw).
    '''
    H, Wd = Y_.size, X_.size
    N = x.size
    Cw = w.shape[1]
    G = max(_TILE, 2*r)
    F = G + 2*r
    nc = min(Cw, _CHAN_GROUP)  # channels per matrix product
    ntr, ntc = -(-H//G), -(-Wd//G)
    # a cell with an unclipped window has a pixel within dx/2 in x and y, so
    # it can only underflow when blur is tiny: then check every cell (the
    # bound on its smallest exponent is doubled for safety)
    check_all = 2.0*2.0*(0.5*dx)**2/den.min() > _FALLBACK_EXPONENT

    row0, row1, col0, col1 = _windows(x, y, X_, Y_, dx, r)
    key = (row0//G)*ntc + col0//G
    order = np.argsort(key.astype(np.uint16) if ntr*ntc <= 65536 else key, kind='stable')  # radix sort
    key = key[order]
    tile_start = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    tile_of_cell = np.cumsum(np.r_[True, key[1:] != key[:-1]]) - 1

    # batches of consecutive (sorted) cells, so each tile of a batch is one run
    # of cells.  The slot frames (about 2 slots per cell, nc + 1 frames of F
    # values per slot) and the per-tile products (nc*F*F values per tile)
    # bound the batch size.
    max_cells = max(256, min(_BATCH_CELLS, _BATCH_BYTES//(16*F*(nc + 1))))
    max_tiles = max(1, _BATCH_BYTES//(8*nc*F*F))
    bounds = [0]
    while bounds[-1] < N:
        s = bounds[-1]
        e = min(N, s + max_cells)
        t = tile_of_cell[s] + max_tiles
        if t < tile_start.size:
            e = min(e, tile_start[t])
        bounds.append(int(e))

    # image as a grid of G x G blocks, plus one block row and column for the
    # overhang of the last tiles' frames
    img = np.zeros((C, ntr + 1, G, ntc + 1, G))
    underflow = []
    for s, e in zip(bounds[:-1], bounds[1:]):
        o = order[s:e]
        flag = _tile_batch(x[o], y[o], row0[o], row1[o], col0[o], col1[o], key[s:e], w[o],
                           X_, Y_, r, den, G, ntc, check_all, img)
        underflow.append(o[flag])
    raw = img.reshape(C, (ntr + 1)*G, (ntc + 1)*G)[:, :H, :Wd]
    return raw, np.concatenate(underflow)


def _tile_batch(x, y, row0, row1, col0, col1, key, w, X_, Y_, r, den, G, ntc, check_all, img):
    ''' Add a batch of cells, sorted by tile key, into img (C, ntr+1, G, ntc+1, G).

    Returns a boolean mask of the cells whose kernel underflows (they are
    left out).
    '''
    n = x.size
    B = den.size
    Cw = w.shape[1]
    L = 2*r + 1
    F = G + 2*r
    j = np.arange(L)

    # squared distances (grid - coordinate)**2 on each cell's row and column
    # windows; positions past the end of a clipped window are set to inf so
    # that their kernel value is exp(-inf) = 0
    dy2 = _row_windows(np.concatenate([Y_, np.zeros(L)]), L)[row0] - y[:, None]
    dy2 *= dy2
    dx2 = _row_windows(np.concatenate([X_, np.zeros(L)]), L)[col0] - x[:, None]
    dx2 *= dx2
    clipped_y = np.flatnonzero(row1 - row0 < 2*r)
    if clipped_y.size:
        dy2[clipped_y] = np.where(j > (row1 - row0)[clipped_y, None], np.inf, dy2[clipped_y])
    clipped_x = np.flatnonzero(col1 - col0 < 2*r)
    if clipped_x.size:
        dx2[clipped_x] = np.where(j > (col1 - col0)[clipped_x, None], np.inf, dx2[clipped_x])
    flag = np.zeros(n, dtype=bool)
    check = np.arange(n) if check_all else np.union1d(clipped_y, clipped_x)
    if check.size:
        flag[check] = (dy2[check].min(1) + dx2[check].min(1))/den.min() > _FALLBACK_EXPONENT
    any_flag = bool(flag.any())

    # slots: tile t owns pad[t] consecutive slots (its cells, then zero
    # padding).  Tiles are stored in order of pad, so tiles with equal pad are
    # contiguous and go through one batched matrix product.
    first = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    cnt = np.diff(np.r_[first, n])
    tiles = key[first]
    T = tiles.size
    pad = _padded_counts(cnt)
    perm = np.argsort(pad, kind='stable')
    pad_sorted = pad[perm]
    base_sorted = np.cumsum(pad_sorted) - pad_sorted
    base = np.empty(T, dtype=int)
    base[perm] = base_sorted
    slot = np.repeat(base - first, cnt) + np.arange(n)
    nslots = int(pad.sum())
    tile_row, tile_col = tiles//ntc, tiles % ntc
    # start of each cell's kernel inside the flattened (nslots, F) frames
    py = slot*F + row0 - np.repeat(tile_row*G, cnt)
    px = slot*F + col0 - np.repeat(tile_col*G, cnt)
    group_start = np.flatnonzero(np.r_[True, pad_sorted[1:] != pad_sorted[:-1]])
    group_end = np.r_[group_start[1:], T]
    tile_row, tile_col = tile_row[perm], tile_col[perm]  # in slot order from here on
    if Cw > 1:
        w_slot = np.zeros((nslots, Cw))
        w_slot[slot] = w

    def add_tile_products(KY, KX, c0, c1):
        ''' img[c0:c1] += the F x F block KY_t^T KX_t of every tile t (KY: (nslots, (c1-c0)*F)). '''
        nch = c1 - c0
        P = np.empty((T, nch*F, F))
        for g0, g1 in zip(group_start, group_end):
            k = int(pad_sorted[g0])
            s0 = int(base_sorted[g0])
            s1 = s0 + (g1 - g0)*k
            A = KY[s0:s1].reshape(g1 - g0, k, nch*F)
            Bk = KX[s0:s1].reshape(g1 - g0, k, F)
            np.matmul(A.transpose(0, 2, 1), Bk, out=P[g0:g1])
        P = P.reshape(T, nch, F, F)
        # overlap-add: the block covers its tile and the overhang into the
        # tiles below, to the right and diagonally (tiles are distinct)
        q = 2*r
        dst = img[c0:c1]
        dst[:, tile_row, :, tile_col, :] += P[:, :, :G, :G]
        if q:
            dst[:, tile_row + 1, :q, tile_col, :] += P[:, :, G:, :G]
            dst[:, tile_row, :, tile_col + 1, :q] += P[:, :, :G, G:]
            dst[:, tile_row + 1, :q, tile_col + 1, :q] += P[:, :, G:, G:]

    ones = np.ones(L)
    with np.errstate(divide='ignore', invalid='ignore'):
        for b in range(B):
            ky = np.exp(dy2/-den[b])
            kx = np.exp(dx2/-den[b])
            fy = 1.0/(ky @ ones)  # 1/sum over the window
            fx = 1.0/(kx @ ones)
            if Cw == 1:
                fy *= w[:, 0]
            if any_flag:
                fy[flag] = 0.0
                fx[flag] = 0.0
            ky *= fy[:, None]
            kx *= fx[:, None]
            KY = np.zeros(nslots*F)
            KX = np.zeros(nslots*F)
            _row_windows(KY, L)[py] = ky
            _row_windows(KX, L)[px] = kx
            KY = KY.reshape(nslots, F)
            KX = KX.reshape(nslots, F)
            if Cw == 1:
                # channel b: one per blur (or the only one)
                add_tile_products(KY, KX, b, b + 1)
            else:
                # one channel per signal column (B == 1), in groups
                for c0 in range(0, Cw, _CHAN_GROUP):
                    c1 = min(Cw, c0 + _CHAN_GROUP)
                    KYc = (w_slot[:, c0:c1, None]*KY[:, None, :]).reshape(nslots, (c1 - c0)*F)
                    add_tile_products(KYc, KX, c0, c1)
    return flag


def _accumulate_full(x, y, w, X_, Y_, den, C):
    ''' use_windowing=False: kernels over the whole image, one matrix product per chunk of cells.

    Returns raw (C, H, Wd) and the indices of the cells whose kernel underflows.
    '''
    H, Wd = Y_.size, X_.size
    N = x.size
    B = den.size
    _, bsel, wsel = _channels(B, w.shape[1])
    raw = np.zeros((C, H, Wd))
    tmp = np.empty((H, Wd))
    m = max(16, min(4096, _BATCH_BYTES//(24*(H + Wd))))  # cells per chunk
    underflow = []
    with np.errstate(divide='ignore', invalid='ignore'):
        for s in range(0, N, m):
            e = min(N, s + m)
            dy2 = (Y_[None, :] - y[s:e, None])**2
            dx2 = (X_[None, :] - x[s:e, None])**2
            flag = (dy2.min(1) + dx2.min(1))/den.min() > _FALLBACK_EXPONENT
            underflow.append(s + np.flatnonzero(flag))
            for b in range(B):
                ky = np.exp(dy2/-den[b])
                ky /= ky.sum(1, keepdims=True)
                kx = np.exp(dx2/-den[b])
                kx /= kx.sum(1, keepdims=True)
                ky[flag] = 0.0
                kx[flag] = 0.0
                for c in np.flatnonzero(bsel == b):
                    np.matmul((w[s:e, wsel[c], None]*ky).T, kx, out=tmp)
                    raw[c] += tmp
    return raw, np.concatenate(underflow)


def _exact_cells(cells, x, y, w, X_, Y_, dx, blur, r, signal, wavelet, use_windowing, W):
    ''' Add the given cells to W (H, Wd, C) with the original per-cell code. '''
    nb = blur.size
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        for n in cells:
            x_, y_ = x[n], y[n]
            if use_windowing:
                col = int(np.round((x_ - X_[0])/dx))
                row = int(np.round((y_ - Y_[0])/dx))
                row0 = min(max(row - r, 0), W.shape[0] - 1)
                row1 = min(max(row + r, 0), W.shape[0] - 1)
                col0 = min(max(col - r, 0), W.shape[1] - 1)
                col1 = min(max(col + r, 0), W.shape[1] - 1)
                rows, cols = slice(row0, row1 + 1), slice(col0, col1 + 1)
            else:
                rows, cols = slice(None), slice(None)
            k = np.exp(- ((X_[None, cols, None] - x_)**2 + (Y_[rows, None, None] - y_)**2)/(2.0*(dx*blur*2)**2))
            if signal:
                k /= np.sum(k, axis=(0, 1), keepdims=True)*dx**2
            else:
                k /= np.sum(k, axis=(0, 1), keepdims=True)
                k *= w[n, 0]
            if wavelet:
                for i in reversed(range(1, nb)):
                    k[..., i] = k[..., i] - k[..., i-1]
            if signal:
                k = k*w[n]
            W[rows, cols, :] += k


def _nan_windows(x, y, nan_cols, X_, Y_, dx, r, use_windowing, W):
    ''' Set the windows of cells with NaN weights to NaN (the original adds kernel * NaN there).

    nan_cols (n, Cw) marks the weight columns that are NaN for each cell.
    '''
    H, Wd, C = W.shape
    if use_windowing:
        row0, row1, col0, col1 = _windows(x, y, X_, Y_, dx, r)
    else:
        z = np.zeros(x.size, dtype=int)
        row0, row1, col0, col1 = z, z + H - 1, z, z + Wd - 1
    for cw in range(nan_cols.shape[1]):
        sel = nan_cols[:, cw]
        if not sel.any():
            continue
        # pixels covered by at least one window: 2D cumulative sum of corner marks
        D = np.zeros((H + 1, Wd + 1))
        np.add.at(D, (row0[sel], col0[sel]), 1.0)
        np.add.at(D, (row0[sel], col1[sel] + 1), -1.0)
        np.add.at(D, (row1[sel] + 1, col0[sel]), -1.0)
        np.add.at(D, (row1[sel] + 1, col1[sel] + 1), 1.0)
        covered = np.cumsum(np.cumsum(D, 0), 1)[:H, :Wd] > 0.5
        for c in (range(C) if nan_cols.shape[1] == 1 else [cw]):
            W[..., c][covered] = np.nan


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------
def _finish(X_, Y_, W, nb, draw, wavelet_magnitude):
    ''' Draw, take the magnitude for wavelets, and put channels first. '''
    fig = _draw(W, nb) if draw else None
    if wavelet_magnitude:
        np.abs(W, out=W)
    # channels first; a transposed (H, Wd, C) array, as in the original
    W = W.transpose((-1, 0, 1))
    if draw:
        return X_, Y_, W, fig
    return X_, Y_, W


def _draw(W, nb):
    ''' Figure of the image (each channel scaled to [0, 1]; up to 3 channels as RGB). '''
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    with np.errstate(divide='ignore', invalid='ignore'):
        toshow = W - np.min(W, axis=(0, 1), keepdims=True)
        toshow = toshow/np.max(toshow, axis=(0, 1), keepdims=True)
    if nb >= 3:
        toshow = toshow[..., :3]
    elif nb == 2:
        toshow = toshow[..., [0, 1, 0]]
    else:
        toshow = toshow[..., [0, 0, 0]]
    ax.imshow(np.abs(toshow))
    return fig
