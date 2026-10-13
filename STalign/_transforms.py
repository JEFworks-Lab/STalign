''' Sampling and coordinate transforms of fitted STalign mappings, in 2D and 3D.

Implementation of STalign.interp, interp3D, v_to_phii(_3D), build_transform(3D),
transform_image_*, transform_points_* and analyze3Dalign (see their docstrings for the
parameters).

A mapping fitted by LDDMM or LDDMM_3D_to_slice is an affine matrix A and a time dependent
velocity field v of shape (nt,*nv,d), components last, on the grid with 1D axes xv.

Backward map b (target -> source), the map LDDMM fits. The source image is sampled at b(y)
for every target pixel y:

    x = A^{-1} y,   then for t = nt-1, ..., 0:   x <- x - v[t](x) / nt

v[t](x) is bilinear (trilinear in 3D) interpolation, zero outside the velocity grid.

Forward map f (source -> target), by default the exact inverse of b. Inverting b one step
at a time is an implicit Euler scheme for the forward flow:

    for t = 0, ..., nt-1:   solve  x_{t+1} = x_t + v[t](x_{t+1}) / nt,   then  y = A x_nt

Each step is solved by the fixed-point iteration x <- x_t + v[t](x) / nt, starting from the
explicit Euler step x_t + v[t](x_t) / nt. The iteration contracts by max|Dv[t]| / nt per
sweep, which is below 1 whenever the backward step is invertible; on the tutorial fits it
needs 3 to 9 sweeps per step. Stopping at the starting point (exact_inverse=False) gives the
forward Euler map of earlier STalign versions, which inverts b only up to a time
discretization error: |b(f(x)) - x| is first order in 1/nt, up to ~100 um on the tutorials.

Evaluation. Points are processed in chunks of rows. Within a chunk they are kept in the
coordinates grid_sample uses (-1..1 across the velocity grid, components in x, y(, z)
order), in a planar (B,d,rows) layout: torch's CPU grid_sample parallelizes over the batch
axis only, and it reads this layout without a copy. The affine parts of the maps are folded
into the conversions from and to physical coordinates, and the velocity fields are
rescaled once, so a time step costs one grid_sample and one addition.
'''

import warnings

import numpy as np
import torch
from torch.nn.functional import grid_sample

# Exact inverse: an implicit Euler step stops when no point moves by more than INVERSE_TOL
# velocity grid cells / nt in one sweep; the round-trip error |b(f(x)) - x| is then below
# about INVERSE_TOL cells. Points still moving after INVERSE_MAX_ITER sweeps raise a warning.
INVERSE_TOL = 1e-6
INVERSE_MAX_ITER = 100
# Points are processed in chunks of this many rows. This bounds the temporary memory, and
# each chunk stops iterating as soon as its own points have converged.
CHUNK_SIZE = 1 << 18
# Minimum number of points per grid_sample batch entry (thread).
_MIN_BATCH_ROWS = 2048


def interp(x,I,phii,**kwargs):
    ''' Interpolate the 2D image I on grid x at the positions phii (see STalign.interp). '''
    # first we have to normalize phii to the range -1,1    
    I = torch.as_tensor(I)
    phii = torch.as_tensor(phii)
    phii = torch.clone(phii)
    for i in range(2):
        phii[i] -= x[i][0]
        phii[i] /= x[i][-1] - x[i][0]
    # note the above maps to 0,1
    phii *= 2.0
    # to 0 2
    phii -= 1.0
    # done

    # grid_sample wants a batch axis and the sample positions in xy order along the last axis
    out = grid_sample(I[None],phii.flip(0).permute((1,2,0))[None],align_corners=True,**kwargs)
    # note align corners true means square voxels with points at their centers
    # post processing, get rid of batch dimension

    return out[0]


def interp3D(x,I,phii,**kwargs):
    ''' Interpolate the 3D image I on grid x at the positions phii (see STalign.interp3D). '''
    # first we have to normalize phii to the range -1,1    
    I = torch.as_tensor(I)
    phii = torch.as_tensor(phii)
    phii = torch.clone(phii)
    for i in range(3):
        phii[i] -= x[i][0]
        phii[i] /= x[i][-1] - x[i][0]
    # note the above maps to 0,1
    phii *= 2.0
    # to 0 2
    phii -= 1.0
    # done

    # grid_sample wants a batch axis and the sample positions in xyz order along the last axis
    out = grid_sample(I[None],phii.flip(0).permute((1,2,3,0))[None],align_corners=True,**kwargs)
    # note align corners true means square voxels with points at their centers
    # post processing, get rid of batch dimension

    return out[0]


def _v_to_phii(xv, v, sample):
    ''' Semi-Lagrangian integration of v (see STalign.v_to_phii), with the interpolation
    function sample (interp or interp3D). '''
    xv = [torch.as_tensor(x) for x in xv]
    v = torch.as_tensor(v)
    d = len(xv)
    nv = tuple(len(x) for x in xv)
    if v.ndim == d + 2 and v.shape[-1] == d and tuple(v.shape[1:-1]) == nv:
        # (nt,*nv,d) as returned by LDDMM: put components first
        v = v.movedim(-1, 1)
    elif not (v.ndim == d + 2 and v.shape[1] == d and tuple(v.shape[2:]) == nv):
        raise ValueError(f'v has shape {tuple(v.shape)}, expected (nt,*nv,{d}) or (nt,{d},*nv) with nv={nv}')
    XV = torch.stack(torch.meshgrid(xv, indexing='ij'))
    phii = torch.clone(XV)
    dt = 1.0/v.shape[0]
    for t in range(v.shape[0]):
        Xs = XV - v[t]*dt
        phii = sample(xv,phii-XV,Xs)+Xs
    return phii


def v_to_phii(xv,v):
    ''' Integrate a 2D velocity field to a position field (see STalign.v_to_phii). '''
    return _v_to_phii(xv, v, interp)


def v_to_phii_3D(xv,v):
    ''' Integrate a 3D velocity field to a position field (see STalign.v_to_phii_3D). '''
    return _v_to_phii(xv, v, interp3D)


def _normalization(x, dtype, device):
    ''' Homogeneous matrix mapping positions (row, col[, slice] order) to the coordinates
    grid_sample uses for a grid with 1D axes x: -1 and 1 at the first and last samples of each
    axis (align_corners=True), with the components in reversed (x, y[, z]) order. '''
    d = len(x)
    x = [torch.as_tensor(xi, dtype=dtype, device=device) for xi in x]
    lo = torch.stack([xi[0] for xi in x])
    s = 2.0 / torch.stack([xi[-1] - xi[0] for xi in x])
    N = torch.zeros(d + 1, d + 1, dtype=dtype, device=device)
    N[:d, :d] = torch.diag(s).flip(0)
    N[:d, d] = (-s * lo - 1.0).flip(0)
    N[d, d] = 1.0
    return N


def _affine_planar(M, U):
    ''' Apply the homogeneous matrix M to the points U, stored as a (B,d,rows) tensor (one plane
    per component). Every output element goes through the same sequence of operations, so a
    point's result does not depend on where it sits in the array. '''
    d = U.shape[1]
    out = U.new_empty(U.shape)
    for i in range(d):
        acc = U[:, 0] * M[i, 0]
        for j in range(1, d):
            acc += U[:, j] * M[i, j]
        torch.add(acc, M[i, d], out=out[:, i])
    return out


def _to_planar(M, X):
    ''' Points X (m,d) mapped by the homogeneous matrix M, as a planar (B,d,rows) tensor.

    The points are split into B batch entries because torch's CPU grid_sample uses one thread per
    batch entry. The last point is repeated to fill the last entry.
    '''
    m, d = X.shape
    B = 1
    if X.device.type == 'cpu':
        B = max(1, min(torch.get_num_threads(), m // _MIN_BATCH_ROWS))
    rows = -(-m // B)
    if B * rows > m:
        X = torch.cat((X, X[-1:].expand(B * rows - m, d)))
    return _affine_planar(M, X.reshape(B, rows, d).transpose(1, 2))


def _from_planar(M, U, m):
    ''' Inverse of _to_planar: the planar points U mapped by M, as an (m,d) tensor. '''
    B, d, rows = U.shape
    return _affine_planar(M, U).transpose(1, 2).reshape(B * rows, d)[:m]


def _grid_sample_planar(I, U, **kwargs):
    ''' Sample the image I (C,*n) at the planar points U (B,d,rows), given in grid_sample
    coordinates; returns (B,C,rows). The planar layout is read by grid_sample without a copy,
    and the image is expanded (not copied) along the batch axis. '''
    B, d, rows = U.shape
    grid = U.transpose(1, 2).reshape((B, rows) + (1,) * (d - 1) + (d,))
    out = grid_sample(I[None].expand((B,) + I.shape), grid, align_corners=True, **kwargs)
    return out.view(B, I.shape[0], rows)


def _row_chunks(X):
    ''' Slices of CHUNK_SIZE rows of X. '''
    return [X[i:i + CHUNK_SIZE] for i in range(0, len(X), CHUNK_SIZE)]


def _float_dtype(*tensors):
    ''' Promoted floating point dtype of the given tensors (None entries and integer tensors are
    ignored); float64 when there is none. '''
    dtype = None
    for t in tensors:
        if t is not None and t.is_floating_point():
            dtype = t.dtype if dtype is None else torch.promote_types(dtype, t.dtype)
    return torch.float64 if dtype is None else dtype


class _Transform:
    ''' Affine matrix A and velocity field v (nt,*nv,d) on the grid xv, prepared for evaluating
    the backward and forward maps at the rows of an (m,d) tensor of points.

    All inputs may be numpy arrays or torch tensors. The computation uses the dtype of A and v
    (promoted; float64 if both are integer) and the device of v (of A if v is None).

    The flow is integrated in grid_sample coordinates u = N x (see _normalization), where one
    Euler step is u <- u +/- w[t](u) with w[t] = N v[t] / nt (linear part of N only). The
    affine parts of both maps are folded into the conversions to and from u.
    '''

    def __init__(self, xv, v, A):
        A = torch.as_tensor(A)
        v = None if v is None else torch.as_tensor(v)
        self.dtype = _float_dtype(A, v)
        self.device = A.device if v is None else v.device
        A = A.to(dtype=self.dtype, device=self.device)
        d = A.shape[0] - 1
        if v is not None and (v.ndim != d + 2 or v.shape[-1] != d or len(xv) != d):
            raise ValueError(f'A of shape {tuple(A.shape)} needs v of shape (nt,*nv,{d}) on {d} '
                             f'axes xv, got v of shape {tuple(v.shape)} and {len(xv)} axes')
        self.dim = d
        N = torch.eye(d + 1, dtype=self.dtype, device=self.device)
        self.w = []
        if v is not None:
            v = v.to(dtype=self.dtype, device=self.device)
            nt = v.shape[0]
            N = _normalization(xv, self.dtype, self.device)
            s = N[:d, :d].flip(0).diagonal()
            # one (d,*nv) field per time step: components first, in reversed order
            self.w = [(v[t] * s / nt).flip(-1).movedim(-1, 0).contiguous() for t in range(nt)]
            # stopping tolerance of each implicit Euler step: INVERSE_TOL of the smallest grid
            # cell (2/(n-1) in grid_sample coordinates), floored at a few ulps for low precision
            cell = 2.0 / (max(len(x) for x in xv) - 1)
            self.tol = max(INVERSE_TOL * cell, 64 * torch.finfo(self.dtype).eps) / nt
        Ni = torch.linalg.inv(N)
        # backward: target y -> N A^{-1} y -> flow -> N^{-1} u;  forward: x -> N x -> flow -> A N^{-1} u
        self.b_in, self.b_out = N @ torch.linalg.inv(A), Ni
        self.f_in, self.f_out = N, A @ Ni
        self.n_unconverged = 0

    def points(self, X):
        ''' X as a tensor of this transform's dtype and device. '''
        return torch.as_tensor(X, dtype=self.dtype, device=self.device)

    def backward(self, X):
        ''' b(X) for points X (m,d): target to source, the map LDDMM fits. '''
        U = _to_planar(self.b_in, X)
        for t in range(len(self.w)-1,-1,-1):
            U = U - _grid_sample_planar(self.w[t], U)
        return _from_planar(self.b_out, U, len(X))

    def forward(self, X, exact_inverse=True):
        ''' Source to target for points X (m,d): the exact inverse of b, or the forward Euler map
        if not exact_inverse. Points that do not converge are counted in n_unconverged. '''
        if exact_inverse and not bool(torch.isfinite(X).all()):
            # points with nan or inf coordinates have no inverse; solve for the others
            finite = torch.isfinite(X).all(-1)
            Y = torch.full_like(X, float('nan'))
            Y[finite] = self.forward(X[finite])
            return Y
        U = _to_planar(self.f_in, X)
        unconverged = None
        for t in range(len(self.w)):
            Y = U
            # explicit Euler step (the original forward map), the initial guess for the inverse
            U = Y + _grid_sample_planar(self.w[t], Y)
            if exact_inverse:
                U, bad = self._implicit_step(self.w[t], Y, U)
                if bad is not None:
                    unconverged = bad if unconverged is None else unconverged | bad
        if unconverged is not None:
            self.n_unconverged += int(unconverged.reshape(-1)[:len(X)].sum())
        return _from_planar(self.f_out, U, len(X))

    def _implicit_step(self, w, Y, U):
        ''' Fixed-point iteration U <- Y + w(U), which inverts the backward step U -> U - w(U).
        Returns the solution and None, or the last iterate and a (B,rows) mask of the points
        that did not converge. '''
        for _ in range(INVERSE_MAX_ITER):
            Un = _grid_sample_planar(w, U)
            Un += Y
            D = Un - U
            U = Un
            if torch.linalg.vector_norm(D, float('inf')) <= self.tol:
                return U, None
        return U, D.abs().amax(1) > self.tol

    def warn_unconverged(self, n_points, stacklevel):
        ''' Warn if some points of the forward map did not converge; stacklevel as for
        warnings.warn, counted from the caller of this method. '''
        if self.n_unconverged:
            warnings.warn(
                f'The exact inverse of the backward map did not converge for {self.n_unconverged} of '
                f'{n_points} points within {INVERSE_MAX_ITER} iterations per time step; their positions '
                'are approximate. The velocity field may be too large for its nt time steps to be '
                'invertible. exact_inverse=False gives the forward Euler map instead.',
                RuntimeWarning, stacklevel=stacklevel + 1)

    def map_points(self, X, direction, exact_inverse=True):
        ''' Backward ('b') or forward ('f') map of the points X (...,d), in chunks of rows. '''
        shape = X.shape
        if shape[-1] != self.dim:
            raise ValueError(f'points have {shape[-1]} components, expected {self.dim}')
        X = X.reshape(-1, shape[-1])
        if not len(X):
            return X.reshape(shape).clone()
        if direction == 'b':
            out = [self.backward(Xc) for Xc in _row_chunks(X)]
        else:
            out = [self.forward(Xc, exact_inverse) for Xc in _row_chunks(X)]
        return torch.cat(out).reshape(shape)


def _sample_points(XJ, xv, dim, tf):
    ''' Sample points for build_transform as a (...,dim) tensor: XJ given as a list of 1D axes
    (meshgrid with ij indexing), a list of meshgrid arrays, or one (...,dim) array; the
    velocity grid xv if XJ is None. '''
    if XJ is None:
        XJ = list(xv)
    if isinstance(XJ, (list, tuple)):
        XJ = [tf.points(x) for x in XJ]
        if XJ[0].ndim == 1: # need meshgrid
            XJ = torch.meshgrid(XJ, indexing='ij')
        elif XJ[0].ndim != dim: # otherwise assume already meshgrid
            raise Exception('Could not understand variable XJ type')
        return torch.stack(XJ, -1)
    # if it is already in meshgrid form we just need to make sure it is a tensor
    return tf.points(XJ)


def _build_transform(xv, v, A, direction, XJ, exact_inverse, dim):
    if direction not in ('b', 'f'):
        raise Exception(f'Direction must be "f" or "b" but you input {direction}')
    tf = _Transform(xv, v, A)
    XJ = _sample_points(XJ, xv, dim, tf)
    Xs = tf.map_points(XJ, direction, exact_inverse)
    # point the warning at the caller of STalign.build_transform(3D)
    tf.warn_unconverged(XJ[..., 0].numel(), stacklevel=4)
    return Xs



def build_transform(xv,v,A,direction='b',XJ=None,exact_inverse=True):
    ''' Backward or forward map on a grid of 2D points (see STalign.build_transform). '''
    return _build_transform(xv, v, A, direction, XJ, exact_inverse, 2)


def build_transform3D(xv,v,A,direction='b',XJ=None,exact_inverse=True):
    ''' Backward or forward map on a grid of 3D points (see STalign.build_transform3D). '''
    return _build_transform(xv, v, A, direction, XJ, exact_inverse, 3)


def _sample_image(x, I, X, **kwargs):
    ''' Sample the image I (C,*n) on the grid with 1D axes x at the positions X (...,d);
    returns (C,...). Same values as interp / interp3D with the same kwargs. '''
    I = torch.as_tensor(I, dtype=X.dtype, device=X.device)
    N = _normalization(x, X.dtype, X.device)
    shape = X.shape[:-1]
    X = X.reshape(-1, X.shape[-1])
    out = [I.new_zeros(I.shape[0], 0)]
    for Xc in _row_chunks(X):
        S = _grid_sample_planar(I, _to_planar(N, Xc), **kwargs)
        out.append(S.transpose(0, 1).reshape(I.shape[0], -1)[:, :len(Xc)])
    return torch.cat(out, 1).reshape(I.shape[:1] + shape)


def transform_image_source_with_A(A, XI, I, XJ):
    ''' Affine transform of an image (see STalign.transform_image_source_with_A). '''
    return transform_image_source_to_target(None, None, A, XI, I, XJ=XJ)


def transform_image_source_to_target(xv,v,A,xI,I,XJ=None):
    ''' Source image sampled at the backward map (see STalign.transform_image_source_to_target). '''
    phii = build_transform(xv,v,A,direction='b',XJ=XJ)
    return _sample_image(xI, I, phii, padding_mode='border')


def transform_image_target_to_source(xv,v,A,xJ,J,XI=None,exact_inverse=True):
    ''' Target image sampled at the forward map (see STalign.transform_image_target_to_source). '''
    phi = build_transform(xv,v,A,direction='f',XJ=XI,exact_inverse=exact_inverse)
    return _sample_image(xJ, J, phi, padding_mode='border')


def transform_points_source_to_target(xv,v,A,pointsI,exact_inverse=True):
    ''' Forward map of (N,d) points (see STalign.transform_points_source_to_target). '''
    tf = _Transform(xv, v, A)
    pointsIt = tf.map_points(tf.points(pointsI), 'f', exact_inverse)
    # point the warning at the caller of STalign.transform_points_source_to_target
    tf.warn_unconverged(len(pointsIt), stacklevel=3)
    return pointsIt


def transform_points_target_to_source(xv,v,A,pointsI):
    ''' Backward map of (N,d) points (see STalign.transform_points_target_to_source). '''
    tf = _Transform(xv, v, A)
    return tf.map_points(tf.points(pointsI), 'b')


def analyze3Dalign(labelfile, xv,v,mat, xJ, dx, scale_x, scale_y, x, y, X_, Y_, namesdict, device='cpu'):
    ''' Atlas coordinates and region labels of the cells of a slice (see STalign.analyze3Dalign).
    xJ, dx, scale_x, scale_y, X_, Y_ and device are not used. '''
    # used only here, so imported here rather than with this module
    import nrrd
    import pandas as pd

    vol,hdr = nrrd.read(labelfile)
    dxL = np.diag(hdr['space directions'])
    xL = [np.arange(n)*d - (n-1)*d/2 for n,d in zip(vol.shape,dxL)]

    # backward map (target -> atlas) at each cell, which lies on the target plane z = 0
    y_ = np.asarray(y, dtype=np.float64)
    x_ = np.asarray(x, dtype=np.float64)
    cells = np.stack([np.zeros_like(y_), y_, x_], -1)
    coords = build_transform3D(xv,v,mat,direction='b',XJ=cells)

    # label of the nearest atlas voxel (0 outside the atlas)
    labels = _sample_image(xL, vol[None].astype(np.float64), coords, mode='nearest')[0].int()

    coords = coords.detach().cpu().numpy()
    df_ = pd.DataFrame()
    df_['coord0'] = coords[:,0]
    df_['coord1'] = coords[:,1]
    df_['coord2'] = coords[:,2]
    df_['x'] = np.asarray(x)
    df_['y'] = np.asarray(y)
    df_['struct_id'] = labels.cpu().numpy()
    df_['acronym'] = [namesdict[i] for i in df_['struct_id']]
    return df_
