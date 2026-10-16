"""Tests of the point, grid and image transforms (STalign/_transforms.py).

The reference is a plain numpy implementation of the maps, with multilinear interpolation
that is zero outside the velocity grid (grid_sample with align_corners=True and zero padding):

    backward  b(y):  x = A^{-1} y;  for t = nt-1..0:  x <- x - v[t](x)/nt
    forward Euler:   for t = 0..nt-1:  x <- x + v[t](x)/nt;  then A x

The exact forward map (the default) is checked by the round trip b(f(x)) = x. The synthetic
velocity fields are smooth, vary in time and vanish at the edge of their grid; they deform
by up to ~7 grid cells, with max|Dv|/nt ~ 0.3 (the tutorial fits have 0.05 to 0.21).
When the frozen original implementation (version 1.0's STalign/STalign.py: the file named by
the environment variable STALIGN_ORIGINAL, for example from `git show b2068ed:STalign/STalign.py`
as the CI workflow does, or else dev/ref/stalign_orig.py) or the reference fits (dev/bench/ref,
dev/data/allen, not distributed) are present, the tests also compare with them.
"""
import importlib.util
import itertools
import os
import sys
import warnings
from collections import defaultdict

import numpy as np
import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from STalign import STalign  # noqa: E402

DEV = os.path.join(REPO, 'dev')


def _original():
    '''Version 1.0's STalign.py, if available (STALIGN_ORIGINAL or dev/ref/stalign_orig.py).'''
    path = os.environ.get('STALIGN_ORIGINAL') or os.path.join(DEV, 'ref', 'stalign_orig.py')
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
# numpy reference
# ---------------------------------------------------------------------------
def np_interp(x, F, X):
    """F (C,*n) on the grid with 1D axes x, at the points X (N,d); zero outside the grid."""
    d = len(x)
    u = [(X[:, i] - x[i][0]) / (x[i][-1] - x[i][0]) * (len(x[i]) - 1) for i in range(d)]
    i0 = [np.floor(ui).astype(int) for ui in u]
    f = [ui - ii for ui, ii in zip(u, i0)]
    out = np.zeros((len(X), F.shape[0]))
    for corner in itertools.product((0, 1), repeat=d):
        w = np.ones(len(X))
        inside = np.ones(len(X), dtype=bool)
        idx = []
        for i, c in enumerate(corner):
            w = w * (f[i] if c else 1 - f[i])
            ii = i0[i] + c
            inside &= (ii >= 0) & (ii < F.shape[1 + i])
            idx.append(np.clip(ii, 0, F.shape[1 + i] - 1))
        out += (w * inside)[:, None] * F[(slice(None),) + tuple(idx)].T
    return out


def np_backward(xv, v, A, Y):
    Ai = np.linalg.inv(A)
    X = Y @ Ai[:-1, :-1].T + Ai[:-1, -1]
    nt = v.shape[0]
    for t in range(nt - 1, -1, -1):
        X = X - np_interp(xv, np.moveaxis(v[t], -1, 0), X) / nt
    return X


def np_forward_euler(xv, v, A, X):
    nt = v.shape[0]
    for t in range(nt):
        X = X + np_interp(xv, np.moveaxis(v[t], -1, 0), X) / nt
    return X @ A[:-1, :-1].T + A[:-1, -1]


# ---------------------------------------------------------------------------
# synthetic transforms
# ---------------------------------------------------------------------------
def make_field(dim, nt, amp=60.0, seed=0):
    """Smooth time dependent velocity (nt,*nv,dim) on a grid with spacing 25, zero at its edge."""
    rng = np.random.default_rng(seed)
    L = [500.0, 600.0, 400.0][:dim]
    n = [41, 49, 33][:dim]
    xv = [np.linspace(-l, l, k) for l, k in zip(L, n)]
    X = np.stack(np.meshgrid(*xv, indexing='ij'), -1)
    window = np.prod([np.cos(np.pi * X[..., i] / (2 * L[i])) ** 2 for i in range(dim)], 0)
    v = np.zeros((nt,) + X.shape)
    for k in range(4):
        c = rng.uniform(-150, 150, dim)
        s = rng.uniform(100, 180)
        a = rng.normal(size=dim) * amp
        g = np.exp(-((X - c) ** 2).sum(-1) / (2 * s ** 2)) * window
        for t in range(nt):
            v[t] += g[..., None] * a * (1 + 0.4 * t * (-1) ** k)
    return xv, v


def make_affine(dim):
    c, s = np.cos(0.3), np.sin(0.3)
    A = np.eye(dim + 1)
    A[-3:-1, -3:-1] = 1.1 * np.array([[c, -s], [s, c]])
    A[:dim, dim] = [30.0, -40.0, 20.0][:dim]
    if dim == 3:
        A[0, 0] = 0.9
    return A


def make_points(dim, n=500, seed=1):
    """Points inside the velocity grid, outside it, and on its nodes."""
    rng = np.random.default_rng(seed)
    L = np.array([500.0, 600.0, 400.0][:dim])
    inside = rng.uniform(-0.9, 0.9, (n, dim)) * L
    outside = rng.uniform(1.05, 1.6, (20, dim)) * L * rng.choice([-1, 1], (20, dim))
    nodes = np.array(list(itertools.product(*[[-L[i], 0.0, 25.0, L[i]] for i in range(dim)])))
    return np.concatenate([inside, outside, nodes])


@pytest.fixture(scope='module')
def tf2():
    xv, v = make_field(2, 3)
    return xv, v, make_affine(2)


@pytest.fixture(scope='module')
def tf3():
    xv, v = make_field(3, 4)
    return xv, v, make_affine(3)


DV = 25.0  # velocity grid spacing of make_field


def T(a):
    return torch.as_tensor(a, dtype=torch.float64)


# ---------------------------------------------------------------------------
# T1: target_to_source is the backward map (build_transform 'b')
# ---------------------------------------------------------------------------
def test_target_to_source_is_backward_map(tf2):
    xv, v, A = tf2
    Y = make_points(2) @ A[:2, :2].T + A[:2, 2]
    out = STalign.transform_points_target_to_source(xv, v, A, Y)
    assert isinstance(out, torch.Tensor) and out.dtype == torch.float64
    # identical to build_transform 'b' at the same points, given as points or as a grid
    assert torch.equal(out, STalign.build_transform(xv, v, A, direction='b', XJ=Y[:, None, :])[:, 0, :])
    assert torch.equal(out, STalign.build_transform(xv, v, A, direction='b', XJ=Y))
    np.testing.assert_allclose(out.numpy(), np_backward(xv, v, A, Y), rtol=0, atol=1e-9)


def test_backward_grid(tf2):
    xv, v, A = tf2
    axes = [np.linspace(-700, 700, 37), np.linspace(-800, 750, 45)]
    Xs = STalign.build_transform(xv, v, A, direction='b', XJ=axes)
    assert Xs.shape == (37, 45, 2)
    grid = np.stack(np.meshgrid(*axes, indexing='ij'), -1)
    np.testing.assert_allclose(Xs.numpy(), np_backward(xv, v, A, grid.reshape(-1, 2)).reshape(grid.shape),
                               rtol=0, atol=1e-9)
    # meshgrid arrays give the same result; XJ=None samples on the velocity grid
    assert torch.equal(Xs, STalign.build_transform(xv, v, A, direction='b', XJ=[grid[..., 0], grid[..., 1]]))
    assert STalign.build_transform(xv, v, A, direction='b').shape == (41, 49, 2)


@pytest.mark.filterwarnings('ignore::UserWarning')  # the original's torch.tensor(tensor) copies
def test_backward_matches_original(tf2, tf3):
    so = _original()
    xv, v, A = tf2
    axes = [np.linspace(-700, 700, 37), np.linspace(-800, 750, 45)]
    args = ([T(x) for x in xv], T(v), T(A))
    np.testing.assert_allclose(STalign.build_transform(*args, direction='b', XJ=axes).numpy(),
                               so.build_transform(*args, direction='b', XJ=axes).numpy(), rtol=0, atol=1e-10)
    xv, v, A = tf3
    axes = [np.linspace(-450, 450, 7), np.linspace(-650, 650, 9), np.linspace(-420, 420, 8)]
    args = ([T(x) for x in xv], T(v), T(A))
    np.testing.assert_allclose(STalign.build_transform3D(*args, direction='b', XJ=axes).numpy(),
                               so.build_transform3D(*args, direction='b', XJ=axes).numpy(), rtol=0, atol=1e-10)


# ---------------------------------------------------------------------------
# T2: the forward map is the exact inverse of the backward map
# ---------------------------------------------------------------------------
def test_forward_exact_inverse_2d(tf2):
    xv, v, A = tf2
    P = make_points(2)
    Q = STalign.transform_points_source_to_target(xv, v, A, P)
    roundtrip = np.abs(np_backward(xv, v, A, Q.numpy()) - P).max()
    assert roundtrip < 1e-6 * DV
    # legacy map: forward Euler, which is not the inverse
    Ql = STalign.transform_points_source_to_target(xv, v, A, P, exact_inverse=False)
    np.testing.assert_allclose(Ql.numpy(), np_forward_euler(xv, v, A, P), rtol=0, atol=1e-9)
    assert np.abs(np_backward(xv, v, A, Ql.numpy()) - P).max() > 1e4 * roundtrip
    # the inverse also holds the other way: f(b(y)) = y
    Y = P @ A[:2, :2].T + A[:2, 2]
    Yr = STalign.transform_points_source_to_target(xv, v, A, STalign.transform_points_target_to_source(xv, v, A, Y))
    assert np.abs(Yr.numpy() - Y).max() < 1e-5 * DV


def test_forward_grid_and_images(tf2):
    xv, v, A = tf2
    axes = [np.linspace(-450, 450, 31), np.linspace(-550, 550, 29)]
    grid = np.stack(np.meshgrid(*axes, indexing='ij'), -1)
    for exact in (True, False):
        phi = STalign.build_transform(xv, v, A, direction='f', XJ=axes, exact_inverse=exact)
        assert phi.shape == (31, 29, 2)
        assert torch.equal(phi.reshape(-1, 2), STalign.transform_points_source_to_target(
            xv, v, A, grid.reshape(-1, 2), exact_inverse=exact))
        # target image sampled at the forward map, clamped at its border
        xJ = [np.linspace(-600, 600, 50), np.linspace(-700, 700, 60)]
        J = np.random.default_rng(2).random((2, 50, 60))
        out = STalign.transform_image_target_to_source(xv, v, A, xJ, J, XI=axes, exact_inverse=exact)
        ref = STalign.interp(xJ, T(J), phi.permute(2, 0, 1), padding_mode='border')
        assert out.shape == (2, 31, 29)
        np.testing.assert_allclose(out.numpy(), ref.numpy(), rtol=0, atol=1e-12)
    # the default is the exact inverse
    assert torch.equal(STalign.build_transform(xv, v, A, direction='f', XJ=axes),
                       STalign.build_transform(xv, v, A, direction='f', XJ=axes, exact_inverse=True))


def test_source_image_to_target(tf2):
    xv, v, A = tf2
    xI = [np.linspace(-600, 600, 50), np.linspace(-700, 700, 60)]
    I = np.random.default_rng(3).random((3, 50, 60))
    XJ = [np.linspace(-500, 500, 21), np.linspace(-600, 600, 23)]
    out = STalign.transform_image_source_to_target(xv, v, A, xI, I, XJ=XJ)
    phii = STalign.build_transform(xv, v, A, direction='b', XJ=XJ)
    ref = STalign.interp(xI, T(I), phii.permute(2, 0, 1), padding_mode='border')
    assert out.shape == (3, 21, 23)
    np.testing.assert_allclose(out.numpy(), ref.numpy(), rtol=0, atol=1e-12)
    # affine only
    out = STalign.transform_image_source_with_A(A, xI, I, XJ)
    phii = STalign.build_transform(None, None, A, direction='b', XJ=XJ)
    np.testing.assert_allclose(out.numpy(), STalign.interp(xI, T(I), phii.permute(2, 0, 1),
                                                           padding_mode='border').numpy(), rtol=0, atol=1e-12)


def test_forward_outside_grid_and_affine_only(tf2):
    xv, v, A = tf2
    # far outside the velocity grid v is zero, so both maps are affine there
    P = np.array([[2000.0, 0.0], [-1500.0, 3000.0]])
    for exact in (True, False):
        Q = STalign.transform_points_source_to_target(xv, v, A, P, exact_inverse=exact)
        np.testing.assert_allclose(Q.numpy(), P @ A[:2, :2].T + A[:2, 2], rtol=1e-15, atol=1e-12)
    P = make_points(2)
    Q = STalign.transform_points_source_to_target(None, None, A, P)
    np.testing.assert_allclose(Q.numpy(), P @ A[:2, :2].T + A[:2, 2], rtol=1e-15, atol=1e-12)
    Y = STalign.transform_points_target_to_source(None, None, A, Q)
    np.testing.assert_allclose(Y.numpy(), P, rtol=0, atol=1e-10)


def test_forward_nonfinite_and_empty(tf2):
    xv, v, A = tf2
    P = np.array([[10.0, 20.0], [np.nan, 0.0], [0.0, np.inf], [-100.0, 50.0]])
    Q = STalign.transform_points_source_to_target(xv, v, A, P).numpy()
    assert np.isnan(Q[1:3]).all() and np.isfinite(Q[[0, 3]]).all()
    np.testing.assert_allclose(Q[[0, 3]], STalign.transform_points_source_to_target(xv, v, A, P[[0, 3]]).numpy(),
                               rtol=0, atol=1e-6 * DV)
    for f in (STalign.transform_points_source_to_target, STalign.transform_points_target_to_source):
        out = f(xv, v, A, np.zeros((0, 2)))
        assert isinstance(out, torch.Tensor) and out.shape == (0, 2)


def test_forward_not_invertible_warns():
    # a velocity field whose Euler steps fold space (|Dv|/nt > 1) has no inverse
    rng = np.random.default_rng(4)
    xv = [np.linspace(0, 100, 11), np.linspace(0, 100, 11)]
    v = rng.normal(size=(1, 11, 11, 2)) * 40
    P = rng.uniform(10, 90, (200, 2))
    with pytest.warns(RuntimeWarning, match='did not converge'):
        Q = STalign.transform_points_source_to_target(xv, v, np.eye(3), P)
    assert torch.isfinite(Q).all()
    with pytest.warns(RuntimeWarning, match='did not converge'):
        STalign.build_transform(xv, v, np.eye(3), direction='f', XJ=P)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        STalign.transform_points_source_to_target(xv, v, np.eye(3), P, exact_inverse=False)


def test_forward_float32(tf2):
    xv, v, A = tf2
    P = make_points(2)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        Q = STalign.transform_points_source_to_target(xv, v.astype(np.float32), A.astype(np.float32), P)
    assert Q.dtype == torch.float32
    Q64 = STalign.transform_points_source_to_target(xv, v, A, P)
    assert (Q.double() - Q64).abs().max() < 1e-3 * DV


@pytest.mark.filterwarnings('ignore::UserWarning')  # the original's torch.tensor(tensor) copies
def test_forward_matches_original_legacy(tf2):
    so = _original()
    xv, v, A = tf2
    P = make_points(2)
    args = ([T(x) for x in xv], T(v), T(A))
    np.testing.assert_allclose(STalign.transform_points_source_to_target(*args, T(P), exact_inverse=False).numpy(),
                               so.transform_points_source_to_target(*args, T(P)).numpy(), rtol=0, atol=1e-10)
    axes = [np.linspace(-450, 450, 31), np.linspace(-550, 550, 29)]
    np.testing.assert_allclose(STalign.build_transform(*args, direction='f', XJ=axes, exact_inverse=False).numpy(),
                               so.build_transform(*args, direction='f', XJ=axes).numpy(), rtol=0, atol=1e-10)


# ---------------------------------------------------------------------------
# T3: 3D forward map
# ---------------------------------------------------------------------------
def test_forward_3d(tf3):
    xv, v, A = tf3
    P = make_points(3, n=300)
    Ql = STalign.build_transform3D(xv, v, A, direction='f', XJ=P, exact_inverse=False)
    np.testing.assert_allclose(Ql.numpy(), np_forward_euler(xv, v, A, P), rtol=0, atol=1e-9)
    Q = STalign.build_transform3D(xv, v, A, direction='f', XJ=P)
    assert Q.shape == P.shape
    roundtrip = np.abs(np_backward(xv, v, A, Q.numpy()) - P).max()
    assert roundtrip < 1e-6 * DV
    assert np.abs(np_backward(xv, v, A, Ql.numpy()) - P).max() > 1e4 * roundtrip
    # same points as (1,N,1,3) and through transform_points_source_to_target
    assert torch.equal(STalign.build_transform3D(xv, v, A, direction='f', XJ=P[None, :, None, :])[0, :, 0, :], Q)
    assert torch.equal(STalign.transform_points_source_to_target(xv, v, A, P), Q)
    # backward map of the target points
    Y = STalign.build_transform3D(xv, v, A, direction='b', XJ=Q)
    np.testing.assert_allclose(Y.numpy(), np_backward(xv, v, A, Q.numpy()), rtol=0, atol=1e-9)
    # on a grid given by its axes
    axes = [np.linspace(-350, 350, 5), np.linspace(-550, 550, 6), np.linspace(-450, 450, 7)]
    G = STalign.build_transform3D(xv, v, A, direction='f', XJ=axes)
    grid = np.stack(np.meshgrid(*axes, indexing='ij'), -1)
    assert G.shape == grid.shape
    assert np.abs(np_backward(xv, v, A, G.reshape(-1, 3).numpy()) - grid.reshape(-1, 3)).max() < 1e-6 * DV


def test_affine_only_3d():
    A = make_affine(3)
    P = make_points(3, n=50)
    for direction, ref in (('f', P @ A[:3, :3].T + A[:3, 3]), ('b', (P - A[:3, 3]) @ np.linalg.inv(A[:3, :3]).T)):
        out = STalign.build_transform3D(None, None, A, direction=direction, XJ=P)
        np.testing.assert_allclose(out.numpy(), ref, rtol=1e-14, atol=1e-11)


def test_bad_inputs(tf2):
    xv, v, A = tf2
    with pytest.raises(Exception, match='Direction'):
        STalign.build_transform(xv, v, A, direction='x')
    with pytest.raises(ValueError):
        STalign.transform_points_source_to_target(xv, v, A, np.zeros((4, 3)))
    with pytest.raises(ValueError):
        STalign.build_transform3D(xv, v, np.eye(4), direction='f', XJ=np.zeros((4, 3)))


# ---------------------------------------------------------------------------
# T4: analyze3Dalign evaluates the backward map at each cell
# ---------------------------------------------------------------------------
def _labels_at(vol, xL, coords):
    """Label of the nearest voxel (0 outside the volume)."""
    dxL = np.array([x[1] - x[0] for x in xL])
    idx = np.rint((coords - np.array([x[0] for x in xL])) / dxL).astype(int)
    inside = np.all((idx >= 0) & (idx < np.array(vol.shape)), 1)
    lab = np.zeros(len(coords), dtype=np.int64)
    lab[inside] = vol[tuple(idx[inside].T)]
    return lab


@pytest.mark.filterwarnings('ignore::DeprecationWarning')  # pynrrd's writer
def test_analyze3Dalign_synthetic(tmp_path, tf3):
    nrrd = pytest.importorskip('nrrd')
    xv, v, A = tf3
    # atlas of 50 unit voxels centered on 0, with a different label in every voxel
    shape = (16, 22, 18)
    vol = np.arange(1, np.prod(shape) + 1, dtype=np.uint32).reshape(shape)
    labelfile = str(tmp_path / 'labels.nrrd')
    nrrd.write(labelfile, vol, {'space directions': np.diag([50.0, 50.0, 50.0])})
    xL = [np.arange(n) * 50.0 - (n - 1) * 50.0 / 2 for n in shape]
    # cells of a slice at target z = 0, some mapped outside the atlas
    rng = np.random.default_rng(5)
    y = rng.uniform(-700, 700, 400)
    x = rng.uniform(-800, 800, 400)
    namesdict = defaultdict(lambda: 'unk', {0: 'bg', 1: 'first', 7: 'seventh'})
    # the unused arguments (xJ, dx, scale_x, scale_y, X_, Y_) can be anything
    df = STalign.analyze3Dalign(labelfile, xv, v, A, None, 50.0, None, None, x, y, None, None, namesdict)
    assert list(df.columns) == ['coord0', 'coord1', 'coord2', 'x', 'y', 'struct_id', 'acronym']
    coords = df[['coord0', 'coord1', 'coord2']].to_numpy()
    exact = np_backward(xv, v, A, np.stack([np.zeros_like(y), y, x], 1))
    np.testing.assert_allclose(coords, exact, rtol=0, atol=1e-9)
    lab = _labels_at(vol, xL, exact)
    assert np.array_equal(df['struct_id'].to_numpy(), lab)
    assert 0 < (lab == 0).mean() < 0.5
    assert np.array_equal(df['x'].to_numpy(), x) and np.array_equal(df['y'].to_numpy(), y)
    assert list(df['acronym']) == [namesdict[i] for i in lab]


def test_analyze3Dalign_reference_fit():
    nrrd = pytest.importorskip('nrrd')
    fit = os.path.join(DEV, 'bench', 'ref', 'merfish_allen3d.npz')
    labelfile = os.path.join(DEV, 'data', 'allen', 'aba_annotation.nrrd')
    if not (os.path.exists(fit) and os.path.exists(labelfile)):
        pytest.skip('reference fit or Allen annotation not available')
    z = np.load(fit)
    xv = [z[f'xv_{i}'] for i in range(3)]
    sub = np.random.default_rng(6).choice(len(z['aa__x']), 3000, replace=False)
    x, y = z['aa__x'][sub], z['aa__y'][sub]
    df = STalign.analyze3Dalign(labelfile, xv, z['v'], z['A'], None, 10, 0.9, 0.9, x, y, None, None,
                                defaultdict(lambda: 'unk'))
    coords = df[['coord0', 'coord1', 'coord2']].to_numpy()
    exact = np_backward(xv, z['v'], z['A'], np.stack([np.zeros_like(y), y, x], 1))
    np.testing.assert_allclose(coords, exact, rtol=0, atol=1e-8)
    vol, hdr = nrrd.read(labelfile)
    dxL = np.diag(hdr['space directions'])
    xL = [np.arange(n) * d - (n - 1) * d / 2 for n, d in zip(vol.shape, dxL)]
    assert np.array_equal(df['struct_id'].to_numpy(), _labels_at(vol, xL, exact))


# ---------------------------------------------------------------------------
# T5: numpy / torch, integer / float32 inputs
# ---------------------------------------------------------------------------
def test_input_types(tf2):
    xv, v, A = tf2
    P = np.round(make_points(2))
    ref_f = STalign.transform_points_source_to_target(xv, v, A, P)
    ref_b = STalign.transform_points_target_to_source(xv, v, A, P)
    inputs = [P.astype(np.int64), P.astype(np.float32), T(P), torch.as_tensor(P, dtype=torch.int32), P.tolist()]
    for Pin in inputs:
        for f, ref in ((STalign.transform_points_source_to_target, ref_f),
                       (STalign.transform_points_target_to_source, ref_b)):
            out = f([T(x) for x in xv], T(v), T(A), Pin)
            assert isinstance(out, torch.Tensor) and out.dtype == torch.float64
            assert torch.equal(out, ref)
    # integer sample grids
    axes = [np.arange(-400, 400, 50), np.arange(-500, 500, 40)]
    assert torch.equal(STalign.build_transform(xv, v, A, direction='f', XJ=axes),
                       STalign.build_transform(xv, v, A, direction='f', XJ=[a.astype(float) for a in axes]))


def test_inputs_not_modified(tf2):
    xv, v, A = tf2
    for P in (make_points(2), T(make_points(2))):
        P0 = P.copy() if isinstance(P, np.ndarray) else P.clone()
        for f in (STalign.transform_points_source_to_target, STalign.transform_points_target_to_source):
            for exact in ((True, False) if f is STalign.transform_points_source_to_target else (True,)):
                kwargs = {} if f is STalign.transform_points_target_to_source else {'exact_inverse': exact}
                f(xv, v, A, P, **kwargs)
                assert (P == P0).all()


def test_float32_image(tf2):
    xv, v, A = tf2
    xI = [np.linspace(-600, 600, 50), np.linspace(-700, 700, 60)]
    I = np.random.default_rng(7).random((3, 50, 60)).astype(np.float32)
    XJ = [np.arange(-500, 500, 50), np.arange(-600, 600, 50)]
    out = STalign.transform_image_source_to_target(xv, v, A, xI, I, XJ=XJ)
    ref = STalign.transform_image_source_to_target(xv, v, A, xI, I.astype(np.float64), XJ=XJ)
    assert out.dtype == torch.float64 and torch.equal(out, ref)
    out = STalign.transform_image_target_to_source(xv, v, A, xI, I, XI=XJ)
    assert torch.equal(out, STalign.transform_image_target_to_source(xv, v, A, xI, I.astype(np.float64), XI=XJ))


def test_calculate_tre_takes_transform_output(tf2):
    xv, v, A = tf2
    P = make_points(2)
    Q = STalign.transform_points_source_to_target(xv, v, A, P)
    mean, std = STalign.calculate_tre(Q, Q.numpy() + [3.0, 4.0])
    assert mean == pytest.approx(5.0) and std == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# T6: v_to_phii accepts the layout LDDMM returns
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('dim', [2, 3])
def test_v_to_phii_layouts(dim, tf2, tf3):
    xv, v, A = tf2 if dim == 2 else tf3
    f = STalign.v_to_phii if dim == 2 else STalign.v_to_phii_3D
    xv_t = [T(x) for x in xv]
    phii = f(xv_t, T(v))
    assert phii.shape == (dim,) + v.shape[1:-1]
    assert torch.equal(phii, f(xv_t, T(v).movedim(-1, 1)))
    # approximates the backward map with an identity affine (up to interpolation error)
    b = STalign.build_transform(xv, v, np.eye(3)) if dim == 2 else STalign.build_transform3D(xv, v, np.eye(4))
    assert (phii.movedim(0, -1) - b).abs().max() < 0.1 * DV
    with pytest.raises(ValueError):
        f(xv_t, T(v)[:, 1:])


@pytest.mark.filterwarnings('ignore::UserWarning')  # the original's torch.tensor(tensor) copies
def test_v_to_phii_matches_original(tf2):
    so = _original()
    xv, v, A = tf2
    xv_t = [T(x) for x in xv]
    assert torch.equal(STalign.v_to_phii(xv_t, T(v)), so.v_to_phii(xv_t, T(v).permute(0, 3, 1, 2)))


def test_interp_matches_original(tf2):
    so = _original()
    xv, v, A = tf2
    I = T(np.moveaxis(v[0], -1, 0))
    phii = T(make_points(2).T[:, :, None])
    for kwargs in ({}, {'padding_mode': 'border'}, {'mode': 'nearest'}):
        assert torch.equal(STalign.interp(xv, I, phii, **kwargs), so.interp(xv, I, phii, **kwargs))


def test_results_do_not_depend_on_thread_count(tf2, tf3):
    n = torch.get_num_threads()
    try:
        out = []
        for threads in (1, 3, 4):
            torch.set_num_threads(threads)
            res = []
            for (xv, v, A), dim in ((tf2, 2), (tf3, 3)):
                P = make_points(dim, n=20000)
                res.append(STalign.transform_points_source_to_target(xv, v, A, P))
                res.append(STalign.transform_points_target_to_source(xv, v, A, P))
            out.append(res)
    finally:
        torch.set_num_threads(n)
    for res in out[1:]:
        assert all(torch.equal(a, b) for a, b in zip(out[0], res))
