'''Tests of the C++ LDDMM engine, ``STalign._lddmm_cpp.CppEngine``.

The engine is checked against the autograd oracle in tests/reference.py: gradients at
non-trivial states (including sample points outside the source image and outside the
velocity grid) and 300-iteration trajectories through the public front-end, with and without
the affine step control, with the exact and the Euler landmark map and with the mixture EM.
Also tested: the engine interface (snapshots, segmented runs, niter=0), agreement with the
torch engine, results that are bitwise independent of the thread count, fork safety and
concurrent calls from Python threads.

Skipped when the extension is not built (``python setup.py build_ext --inplace``).
'''
import os
import select
import sys
import threading
import time
import warnings

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for path in (REPO, HERE):
    if path not in sys.path:
        sys.path.insert(0, path)

try:
    from STalign import _lddmm_cpp
except ImportError as e:  # extension not built, or a stale build
    pytest.skip(f'the C++ extension is not available ({e})', allow_module_level=True)
from STalign import _lddmm_common as C  # noqa: E402
from STalign import _lddmm_torch as TE  # noqa: E402
import reference as R  # noqa: E402

CppEngine = _lddmm_cpp.CppEngine
THREADS = 4  # the most threads any test uses

# Step sizes that give a stable, monotone descent on the synthetic problems (larger steps make
# these small fits chaotic, and then any two implementations drift apart at roundoff level).
FIT_2D = dict(a=12.0, sigmaM=0.3, sigmaR=5e3, epT=1e-2, epL=1e-5, epV=0.2)
FIT_3D = dict(nt=4, a=10.0, expand=1.25, sigmaM=0.5, sigmaR=5e2, epT=1e-2, epL=1e-5, epV=5.0,
              L=0.95 * R.rotation_3d(4.0, -3.0), T=[0.5, 0.3, -0.4])


def rel(a, b):
    '''max |a - b| relative to max |b|.'''
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return np.abs(a - b).max() / max(np.abs(b).max(), 1e-300)


def problem_2d(niter=300, CI=3, CJ=2, points=False, **kw):
    s = R.synthetic_2d(CI=CI, CJ=CJ)
    if points:
        kw.update(pointsI=s['pointsI'], pointsJ=s['pointsJ'])
    args = dict(FIT_2D, **kw)
    return C.prepare_problem(2, s['xI'], s['I'], s['xJ'], s['J'], niter=niter, **args), (s, args)


def problem_3d(niter=300, **kw):
    s = R.synthetic_3d()
    args = {**FIT_3D, 'affine_step_control': True, **kw}  # on, as in LDDMM_3D_to_slice
    return C.prepare_problem(3, s['xI'], s['I'], s['xJ'], s['J'], niter=niter, **args), (s, args)


def problem_step(dim, niter=300, **kw):
    '''A problem on which the affine step control engages (reference.step_control_problem),
    with the control on.'''
    d = R.step_control_problem(dim)
    s = {k: d.pop(k) for k in ('xI', 'I', 'xJ', 'J')}
    d.pop('niter')
    args = dict(d, affine_step_control=True, **kw)
    return C.prepare_problem(dim, s['xI'], s['I'], s['xJ'], s['J'], niter=niter, **args), (s, args)


def run_engine(problem, n, num_threads=THREADS):
    engine = CppEngine(problem, num_threads)
    engine.run(n)
    return engine.snapshot()


def assert_snapshots_equal(a, b):
    assert a.keys() == b.keys()
    for k in a:
        np.testing.assert_array_equal(a[k], b[k], err_msg=k)


# =========================================================================== gradients
def outside(X, xs):
    '''Fraction of the points X (..., d) outside the box of the grid xs.'''
    X = np.asarray(X).reshape(-1, len(xs))
    return np.mean(np.any([(X[:, d] < float(x[0])) | (X[:, d] > float(x[-1])) for d, x in enumerate(xs)], 0))


GRADIENT_CASES = {
    # (dim, scale/rotation, translation, velocity grid expansion, velocity amplitude, landmarks, CI, CJ)
    '2d_multichannel_points': (2, 1.05 * R.rotation_2d(-6.0), [1.5, -2.0], 1.4, 6.0, True, 3, 2),
    '2d_far_outside': (2, 1.3 * R.rotation_2d(20.0), [14.0, -11.0], 1.0, 8.0, True, 1, 3),
    '3d': (3, 0.95 * R.rotation_3d(10.0, -7.0), [1.0, 0.5, -1.2], 1.3, 4.0, False, 2, 1),
}


@pytest.mark.parametrize('case,landmark_map', [(case, 'exact') for case in GRADIENT_CASES]
                         + [('2d_far_outside', 'euler')])
def test_gradient_matches_oracle(case, landmark_map):
    dim, L, T, expand, amp, points, CI, CJ = GRADIENT_CASES[case]
    if dim == 2:
        s = R.synthetic_2d(CI=CI, CJ=CJ)
        a, sig = 12.0, dict(sigmaM=0.3, sigmaR=2e3, sigmaP=4.0)
        xv = R.velocity_grid(s['xI'], expand, a)
        v = R.smooth_random_v(xv, 3, amp, seed=1).numpy()
    else:
        s = R.synthetic_3d()
        a, sig = 10.0, dict(sigmaM=0.5, sigmaR=5e2)
        xv = R.velocity_grid(s['xI'], expand, a)
        v = R.smooth_random_v(xv, 4, amp, seed=2).numpy()
    pI, pJ = (s['pointsI'], s['pointsJ']) if points else (None, None)
    problem = C.prepare_problem(dim, s['xI'], s['I'], s['xJ'], s['J'], pointsI=pI, pointsJ=pJ, L=L, T=T, v=v,
                                xv=[x.numpy() for x in xv], a=a, landmark_map=landmark_map, **sig)
    problem.WM = np.random.default_rng(0).uniform(0.2, 1.0, problem.J.shape[1:])  # non-uniform weights
    got = CppEngine(problem, THREADS).gradient()
    ref = R.gradients(s['xI'], s['I'], s['xJ'], s['J'], L, T, v, xv, problem.WM, a=a, pointsI=pI, pointsJ=pJ,
                      landmark_map=landmark_map, **sig)

    for k in ('Lgrad', 'Tgrad', 'vgrad', 'direction'):
        assert rel(got[k], ref[k]) < 1e-9, k
    for k in ('EM', 'ER') + (('EP',) if points else ()):
        assert got[k] == pytest.approx(ref[k].item(), rel=1e-12), k
    assert (got['EP'] is None) == (ref['EP'] is None)
    # the state exercises border clamping of the image and zero padding of the velocity
    assert outside(ref['Xs'], s['xI']) > 0.03
    if case != '2d_multichannel_points':
        assert outside(ref['Xs'], xv) > 0.05


# =========================================================================== trajectories
TRAJECTORY_CASES = {
    '2d_estimated_mu': lambda: problem_2d(),
    '2d_points': lambda: problem_2d(points=True, sigmaP=2.0),
    '2d_points_euler': lambda: problem_2d(points=True, sigmaP=2.0, landmark_map='euler'),
    '2d_mixture_points': lambda: problem_2d(points=True, sigmaP=2.0, mixture='em'),
    '2d_mixture_artifact_user_muB': lambda: problem_2d(niter=120, mixture=dict(classes='MAB', start=20, every=3),
                                                        muB=[0.0, 0.05]),
    '2d_diffeo_start_user_mu': lambda: problem_2d(diffeo_start=30, muA=[0.7, 0.8], muB=0),
    '2d_one_source_channel_nt5': lambda: problem_2d(CI=1, CJ=3, nt=5),
    '3d': lambda: problem_3d(muA=[3.0], muB=0),
    '3d_no_step_control': lambda: problem_3d(affine_step_control=False),
    '2d_step_control_points_diffeo_start': lambda: problem_step(2),
    '3d_step_control_diffeo_start': lambda: problem_step(3, muA=[3.0], muB=0),
}


@pytest.mark.parametrize('case', TRAJECTORY_CASES)
def test_trajectory_matches_oracle(case):
    problem, (s, args) = TRAJECTORY_CASES[case]()
    out = C.run_lddmm(problem, draw=False, num_threads=THREADS, backend='cpp')
    f = R.LDDMM if problem.dim == 2 else R.LDDMM_3D_to_slice
    ref = f(s['xI'], s['I'], s['xJ'], s['J'], niter=problem.niter, **args)
    for k in ('A', 'v', 'WM', 'WA', 'WB') + (('Xs',) if problem.dim == 3 else ()):
        assert rel(out[k], ref[k]) < 1e-7, k
    assert float(ref['v'].abs().max()) > 0.5  # the fit really deforms
    snap = run_engine(problem, problem.niter)
    assert snap['Esave'].shape == ref['Esave'].shape
    assert rel(snap['Esave'], ref['Esave']) < 1e-7
    assert ('step_multipliers' in snap) == ('step_multipliers' in ref) == problem.affine_step_control
    if problem.affine_step_control:
        assert rel(snap['step_multipliers'], ref['step_multipliers']) < 1e-7


@pytest.mark.parametrize('dim', [2, 3])
def test_step_control_segmented_by_draws_matches_oracle(dim):
    # the front-end splits the run at the draws (iterations 0, 70, 140, ...); the multipliers
    # and previous gradients live in the engine and carry over between segments
    problem, (s, args) = problem_step(dim)
    ref = (R.LDDMM if dim == 2 else R.LDDMM_3D_to_slice)(s['xI'], s['I'], s['xJ'], s['J'], niter=problem.niter, **args)
    assert float(ref['step_multipliers'].min()) < 0.05  # the control is active
    whole = C.run_lddmm(problem, draw=False, num_threads=THREADS, backend='cpp')
    drawn = C.run_lddmm(problem, draw=70, num_threads=THREADS, backend='cpp')
    for k in ('A', 'v', 'WM', 'WA', 'WB') + (('Xs',) if dim == 3 else ()):
        assert torch.equal(drawn[k], whole[k]), k
        assert rel(drawn[k], ref[k]) < 1e-7, k


@pytest.mark.parametrize('dim', [2, 3])
def test_step_control_matches_torch_engine(dim):
    problem, _ = problem_step(dim)
    ours = run_engine(problem, problem.niter)
    torch_engine = TE.TorchEngine(problem, THREADS)
    torch_engine.run(problem.niter)
    theirs = {k: x.numpy() if isinstance(x, torch.Tensor) else x for k, x in torch_engine.snapshot().items()}
    assert ours.keys() == theirs.keys()
    assert ours.get('landmark_unconverged', 0) == theirs.get('landmark_unconverged', 0) == 0
    for k in ('A_eval', 'v_eval', 'L', 'T', 'v', 'WM', 'WA', 'WB', 'step_multipliers') + (('Xs',) if dim == 3 else ()):
        assert rel(ours[k], theirs[k]) < 1e-9, k


@pytest.mark.parametrize('mixture', [None, 'em', dict(classes='MAB', start=20, every=3, fix_muB=False)])
def test_landmark_maps_and_mixture_match_torch_engine(mixture):
    for landmark_map in ('exact', 'euler'):
        problem, _ = problem_2d(niter=150, points=True, sigmaP=2.0, landmark_map=landmark_map, mixture=mixture)
        ours = run_engine(problem, problem.niter)
        torch_engine = TE.TorchEngine(problem, THREADS)
        torch_engine.run(problem.niter)
        theirs = {k: x.numpy() if isinstance(x, torch.Tensor) else x for k, x in torch_engine.snapshot().items()}
        assert ours.keys() == theirs.keys()
        for k in ('A_eval', 'v_eval', 'L', 'T', 'v', 'WM', 'WA', 'WB', 'pointsIt', 'Esave'):
            assert rel(ours[k], theirs[k]) < 1e-9, (landmark_map, k)


def test_unconverged_landmark_map_is_counted_and_warns():
    s = R.synthetic_2d(CI=1, CJ=1)
    xv = [x.numpy() for x in R.velocity_grid(s['xI'], 1.0, 8.0)]
    # a velocity field too rough for its single time step to be invertible
    v = np.random.default_rng(0).uniform(-20.0, 20.0, size=(1,) + tuple(len(x) for x in xv) + (2,))
    problem = C.prepare_problem(2, s['xI'], s['I'], s['xJ'], s['J'], pointsI=s['pointsI'], pointsJ=s['pointsJ'],
                                v=v, xv=xv, a=8.0, niter=1)
    assert run_engine(problem, 1)['landmark_unconverged'] > 0
    with pytest.warns(RuntimeWarning, match='exact landmark map did not converge'):
        C.run_lddmm(problem, draw=False, num_threads=THREADS, backend='cpp')


def test_estimated_and_fixed_mu_match_oracle_estep():
    # the first E-step (iteration 50) uses muA/muB re-estimated at iteration 50 or fixed ones
    for kw in ({}, {'muA': [0.9, 1.1], 'muB': [0.0, 0.05]}):
        problem, (s, args) = problem_2d(niter=56, **kw)
        snap = run_engine(problem, problem.niter)
        ref = R.LDDMM(s['xI'], s['I'], s['xJ'], s['J'], niter=problem.niter, **args)
        for k in ('WM', 'WA', 'WB'):
            assert rel(snap[k], ref[k]) < 1e-10, (kw, k)


# =========================================================================== engine interface
def test_snapshot_contract_2d_with_points():
    problem, _ = problem_2d(niter=7, points=True)
    engine = CppEngine(problem, THREADS)
    engine.run(7)
    snap = engine.snapshot()
    nJ, nv = problem.J.shape[1:], problem.v.shape
    shapes = {'L': (2, 2), 'T': (2,), 'A_eval': (3, 3), 'v': nv, 'v_eval': nv, 'WM': nJ, 'WB': nJ, 'WA': nJ,
              'AI': problem.I.shape[:1] + nJ, 'fAI': problem.J.shape, 'Esave': (7, 4),
              'pointsIt': problem.pointsI.shape}
    assert snap.pop('landmark_unconverged') == 0  # the exact landmark map converged
    assert {k: snap[k].shape for k in snap} == shapes
    assert all(x.dtype == np.float64 for x in snap.values())
    # the caller owns the arrays
    snap['v'][...] = np.nan
    snap['WM'][...] = np.nan
    assert np.isfinite(engine.snapshot()['v']).all() and np.isfinite(engine.snapshot()['WM']).all()
    # A_eval / v_eval are the pair of the last forward pass, L/T/v the updated parameters
    assert not np.array_equal(snap['A_eval'][:2], np.c_[snap['L'], snap['T']])
    np.testing.assert_array_equal(engine.snapshot()['A_eval'], snap['A_eval'])


def test_snapshot_contract_3d():
    problem, _ = problem_3d(niter=3)
    snap = run_engine(problem, 3)
    nJ = problem.J.shape[1:]
    assert snap['Xs'].shape == nJ + (3,)
    assert snap['Esave'].shape == (3, 3)
    assert 'pointsIt' not in snap
    assert snap['A_eval'].shape == (4, 4)


def test_first_iteration_evaluates_initial_state():
    # F2: after one iteration, A_eval and v_eval are the initial parameters
    problem, _ = problem_2d(niter=1)
    snap = run_engine(problem, 1)
    np.testing.assert_array_equal(snap['A_eval'], C.affine(problem.L, problem.T))
    np.testing.assert_array_equal(snap['v_eval'], problem.v)
    assert np.abs(snap['v']).max() > 0  # but v itself was updated


def test_niter_zero_returns_initial_state():
    problem, _ = problem_3d(niter=0)
    out = C.run_lddmm(problem, draw=False, num_threads=THREADS, backend='cpp')
    np.testing.assert_array_equal(out['A'].numpy(), C.affine(problem.L, problem.T))
    np.testing.assert_array_equal(out['v'].numpy(), problem.v)
    assert out['Xs'].shape == problem.J.shape[1:] + (3,)


@pytest.mark.parametrize('make', [lambda: problem_2d(niter=0, points=True, diffeo_start=13)[0],
                                  lambda: problem_2d(niter=0, points=True, mixture=dict(start=13, every=7))[0],
                                  lambda: problem_step(2, niter=0, diffeo_start=13)[0],
                                  lambda: problem_step(3, niter=0, diffeo_start=13)[0]],
                         ids=['2d', '2d_mixture', '2d_step_control', '3d_step_control'])
def test_segmented_runs_are_identical(make):
    # the iteration counter persists across run() calls (it % 5, it >= 50 and diffeo_start
    # rules, the mixture EM's schedule), and so does the state of the affine step control and
    # of the mixture EM
    problem = make()
    whole = run_engine(problem, 120)
    engine = CppEngine(problem, THREADS)
    for n in (1, 12, 37, 50, 20):
        engine.run(n)
    assert_snapshots_equal(engine.snapshot(), whole)


def test_problem_arrays_are_not_modified():
    problem, _ = problem_2d(niter=0, points=True)
    before = {k: getattr(problem, k).copy() for k in ('I', 'J', 'L', 'T', 'v', 'WM', 'WA', 'WB', 'pointsI')}
    run_engine(problem, 60)
    for k, x in before.items():
        np.testing.assert_array_equal(getattr(problem, k), x, err_msg=k)


@pytest.mark.parametrize('change', ['device', 'dtype', 'source_channels', 'target_channels', 'nt', 'thin_source'])
def test_unsupported_problems(change):
    problem, _ = problem_2d(niter=0)
    ext = _lddmm_cpp._ext
    if change == 'device':
        problem.device = torch.device('meta')
    elif change == 'dtype':
        problem.dtype = torch.float32
    elif change == 'source_channels':
        problem.I = np.zeros((ext.MAX_CI + 1,) + problem.I.shape[1:])
    elif change == 'target_channels':
        problem.J = np.zeros((ext.MAX_CJ + 1,) + problem.J.shape[1:])
    elif change == 'nt':
        problem.v = np.zeros((ext.MAX_NT + 1,) + problem.v.shape[1:])
    else:
        problem.I = problem.I[:, :1]
    with pytest.raises(C.Unsupported):
        CppEngine(problem, THREADS)


def test_mu_forms_are_equivalent():
    # ints, float32, lists, numpy and torch, 1 or C_J entries: all the same (C_J,) float64 means
    results = []
    for muA in (3, np.float32(3.0), [3, 3], np.array([3.0]), torch.tensor([3, 3]), torch.tensor(3.0, dtype=torch.float32)):
        problem, _ = problem_2d(niter=0, muA=muA, muB=[0.0, 0.0])
        results.append(run_engine(problem, 60))
    for snap in results[1:]:
        assert_snapshots_equal(snap, results[0])


# =========================================================================== threads
@pytest.mark.parametrize('make', [lambda: problem_2d(niter=0, points=True)[0], lambda: problem_3d(niter=0)[0],
                                  lambda: problem_step(2, niter=0)[0], lambda: problem_step(3, niter=0)[0],
                                  lambda: problem_2d(niter=0, mixture=dict(start=10, every=3))[0]],
                         ids=['2d', '3d', '2d_step_control', '3d_step_control', '2d_mixture'])
def test_results_do_not_depend_on_thread_count(make):
    problem = make()
    reference = run_engine(problem, 60, num_threads=1)
    for n in (2, 3, THREADS):
        assert_snapshots_equal(run_engine(problem, 60, num_threads=n), reference)


def test_concurrent_calls_from_python_threads():
    problems = [problem_2d(niter=0, points=True)[0], problem_3d(niter=0)[0]]
    serial = [run_engine(p, 80, num_threads=n) for p, n in zip(problems, (THREADS, 2))]
    results = [None, None]

    def work(i, n):
        results[i] = run_engine(problems[i], 80, num_threads=n)

    threads = [threading.Thread(target=work, args=(i, n)) for i, n in enumerate((THREADS, 2))]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
    assert not any(t.is_alive() for t in threads)
    for got, want in zip(results, serial):
        assert_snapshots_equal(got, want)


@pytest.mark.skipif(not hasattr(os, 'fork'), reason='needs os.fork')
def test_fork_safety():
    # the parent's worker threads do not exist in a forked child; the child must re-create
    # its pool (any thread count) instead of hanging or failing
    problem, _ = problem_2d(niter=0)
    expected = run_engine(problem, 30, num_threads=THREADS)['v']  # leaves THREADS workers running
    r, w = os.pipe()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)  # "process is multi-threaded" warning
        pid = os.fork()
    if pid == 0:  # child: never return into pytest
        code = 1
        try:
            os.close(r)
            ok = all(np.array_equal(run_engine(problem, 30, num_threads=n)['v'], expected) for n in (THREADS, 3, 1))
            os.write(w, b'ok' if ok else b'mismatch')
            code = 0
        finally:
            os._exit(code)
    os.close(w)
    try:
        ready, _, _ = select.select([r], [], [], 120)
        message = os.read(r, 16) if ready else b'timeout'
    finally:
        os.close(r)
        if message == b'timeout':
            os.kill(pid, 9)
        os.waitpid(pid, 0)
    assert message == b'ok'
    # and the parent's pool still works
    np.testing.assert_array_equal(run_engine(problem, 30, num_threads=THREADS)['v'], expected)


@pytest.mark.parametrize('first', ['package', 'sys_path'])
def test_package_and_sys_path_imports_share_the_extension(first):
    # the tutorials put the STalign directory on sys.path; importing the module both ways in one
    # process must not initialize the extension twice (which fails)
    imports = {'package': f'sys.path.insert(0, {REPO!r}); import STalign._lddmm_cpp as a',
               'sys_path': f'sys.path.append({os.path.join(REPO, "STalign")!r}); import _lddmm_cpp as b'}
    second = 'sys_path' if first == 'package' else 'package'
    code = f'import sys; {imports[first]}; {imports[second]}; assert a._ext is b._ext'
    import subprocess
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr


def test_run_releases_the_gil():
    problem, _ = problem_3d(niter=0)
    engine = CppEngine(problem, 1)
    ticks = []
    done = threading.Event()

    def ticker():
        while not done.is_set():
            ticks.append(time.perf_counter())
            time.sleep(0.001)

    t = threading.Thread(target=ticker)
    t.start()
    start = time.perf_counter()
    engine.run(200)
    elapsed = time.perf_counter() - start
    done.set()
    t.join()
    during = [x for x in ticks if start < x < start + elapsed]
    assert elapsed < 0.05 or len(during) > 5  # the Python thread kept running
