'''LDDMM image registration: LDDMM (2D to 2D) and LDDMM_3D_to_slice (3D atlas to a 2D slice).'''
import warnings

import torch

try:
    from ._lddmm_common import DEFAULT_SIGMAR_3D, prepare_problem, run_lddmm
except ImportError:
    from _lddmm_common import DEFAULT_SIGMAR_3D, prepare_problem, run_lddmm


def LDDMM(xI, I, xJ, J, pointsI=None, pointsJ=None,
          L=None, T=None, A=None, v=None, xv=None,
          a=500.0, p=2.0, expand=2.0, nt=3,
          niter=5000, diffeo_start=0, epL=2e-8, epT=2e-1, epV=2e3,
          sigmaM=1.0, sigmaB=2.0, sigmaA=5.0, sigmaR=5e5, sigmaP=2e1,
          device='cpu', dtype=torch.float64, muB=None, muA=None,
          *, draw='end', num_threads=None, backend='auto', affine_step_control=False):
    ''' Run LDDMM between a pair of images.

    This jointly estimates an affine transform A, and a diffeomorphism phi.
    The map is of the form x -> A phi x


    Parameters
    ----------
    xI : list of torch tensor
        Location of voxels in source image I
    I : torch tensor
        source image I, with channels along first axis
    xJ : list of torch tensor
        Location of voxels in target image J
    J : torch tensor
        Target image J, with channels along first axis
    L : torch tensor
        Initial guess for linear transform (2x2 torch tensor). Defaults to None (identity).
    T : torch tensor
        Initial guess for translation (2 element torch tensor). Defaults to None (identity)
    A : torch tensor
        Initial guess for affine matrix.  Either L and T can be specified, or A, but not both.
        Defaults to None (identity).
    v : torch tensor
        Initial guess for velocity field, shape (nt, len(xv[0]), len(xv[1]), 2)
    xv : torch tensor
        pixel locations for velocity field
    a : float
        Smoothness scale of velocity field (default 500.0)
    p : float
        Power of Laplacian in velocity regularization (default 2.0)
    expand : float
        Factor to expand size of velocity field around image boundaries (default 2.0)
    nt : int
        Number of timesteps for integrating velocity field (default 3). Ignored if you input v.
    pointsI : torch tensor
        N x 2 set of corresponding points for matching in source image. Default None (no points).
    pointsJ : torch tensor
        N x 2 set of corresponding points for matching in target image. Default None (no points).
    niter : int
        Number of iterations of gradient descent optimization
    diffeo_start : int
        Number of iterations of gradient descent optimization for affine only, before nonlinear deformation.
    epL : float
        Gradient descent step size for linear part of affine.
    epT : float
        Gradient descent step size of translation part of affine.
    epV : float
        Gradient descent step size for velocity field.
    sigmaM : float
        Standard deviation of image matching term for Gaussian mixture modeling in cost function.
        This term generally controls matching accuracy with smaller corresponding to more accurate.
        As an common example (rule of thumb), you could chose this parameter to be the variance of the pixels
        in your target image.
    sigmaB : float
        Standard deviation of backtround term for Gaussian mixture modeling in cost function.
        If there is missing tissue in target, we may label some pixels in target as background,
        and not enforce matching here.
    sigmaA : float
        Standard deviation of artifact term for Gaussian mixture modeling in cost function.
        If there are artifacts in target or other lack of corresponding between template and target,
        we may label some pixels in target as artifact, and not enforce matching here.
    sigmaR: float
        Standard deviation for regularization. Smaller sigmaR means a smoother resulting transformation.
        Regularization is of the form: 0.5/sigmaR^2 sum_{t=0}^{nt-1} int_X |L v_t|^2 dx with
        L = (1 - a^2 Laplacian)^p, i.e. a sum over the nt time steps without a dt = 1/nt factor
        (nt times 0.5/sigmaR^2 int_0^1 int_X |Lv|^2 dx dt). Changing nt at a fixed sigmaR therefore
        changes the strength of the regularization.
    sigmaP: float
        Standard deviation for matching of points.
        Cost is of the form 0.5/sigmaP^2 sum_i (source_point_i - target_point_i)^2, where the source
        points are mapped to the target by the exact inverse of the fitted target-to-source map,
        i.e. as transform_points_source_to_target maps them after the fit. (Earlier versions used
        forward Euler steps, which invert that map only approximately.)
    device: str
        torch device. defaults to 'cpu'. Can also be 'cuda:0' for example.
    dtype: torch dtype
        torch data type. defaults to torch.float64
    muA: torch tensor whose dimension is the same as the target image
        Defaults to None, which means we estimate this. If you provide a value, we will not estimate it.
        Give one value per channel of the target (size 3 for a RGB image) or a single value for all channels.
        Ints, floats, lists, numpy arrays and torch tensors are accepted.
    muB: torch tensor whose dimension is the same as the target image
        Defaults to None, which means we estimate this. If you provide a value, we will not estimate it.
        Same format as muA.
    draw : 'end', bool or int, keyword only
        Progress figures (images, weights, velocity and energies). 'end' (default) draws them once
        after the last iteration, True every 10 iterations, an int N every N iterations, and
        False, None or 0 never. Drawing is slow; a drawing error only issues a warning.
    num_threads : int, keyword only
        Maximum number of CPU threads. Defaults to the environment variable STALIGN_NUM_THREADS,
        else torch.get_num_threads(). Fewer threads are used for small images.
    backend : {'auto', 'cpp', 'torch'}, keyword only
        'auto' (default) uses the compiled C++ engine when it is installed and the problem runs
        on the CPU in float64, and the pure torch engine otherwise. 'cpp' raises an error if the
        compiled engine cannot run the problem; 'torch' always uses torch. Both give the same
        results up to rounding.
    affine_step_control : bool, keyword only
        If True, the step of each affine parameter (each entry of L and T) is scaled by its own
        multiplier between 1e-3 and 1. The multiplier is halved whenever that parameter's
        gradient changes sign from one iteration to the next (the previous step overshot) and
        grows by 5% per iteration (up to 1) while the sign stays the same, so steps are never
        larger than epL and epT. This stops the affine part from oscillating when epL or epT is
        too large. Default False, which gives the plain gradient steps of earlier versions.
        (LDDMM_3D_to_slice uses it by default.)

    Returns a dictionary
    -------
    {
    'A': torch tensor
        Affine transform
    'v': torch tensor
        Velocity field
    'xv': list of torch tensor
        Pixel locations in v
    'WM': torch tensor
        Resulting weight 2D array (matching)
    'WB': torch tensor
        Resulting weight 2D array (background)
    'WA': torch tensor
        Resulting weight 2D array (artifact)
    }
    A and v are the pair used by the last iteration's forward pass, so build_transform(xv, v, A)
    reproduces the final alignment. With niter=0 the initial state is returned.

    '''
    problem = prepare_problem(2, xI, I, xJ, J, pointsI=pointsI, pointsJ=pointsJ, L=L, T=T, A=A, v=v, xv=xv,
                              a=a, p=p, expand=expand, nt=nt, niter=niter, diffeo_start=diffeo_start,
                              epL=epL, epT=epT, epV=epV, sigmaM=sigmaM, sigmaB=sigmaB, sigmaA=sigmaA,
                              sigmaR=sigmaR, sigmaP=sigmaP, device=device, dtype=dtype, muA=muA, muB=muB,
                              affine_step_control=affine_step_control)
    return run_lddmm(problem, draw=draw, num_threads=num_threads, backend=backend)


def LDDMM_3D_to_slice(xI, I, xJ, J, pointsI=None, pointsJ=None,
                      L=None, T=None, A=None, v=None, xv=None,
                      a=500.0, p=2.0, expand=1.25, nt=3,
                      niter=5000, diffeo_start=0, epL=1e-6, epT=1e1, epV=1e3,
                      sigmaM=1.0, sigmaB=2.0, sigmaA=5.0, sigmaR=DEFAULT_SIGMAR_3D, sigmaP=2e1,
                      device='cpu', dtype=torch.float64, muA=None, muB=None,
                      *, draw='end', num_threads=None, backend='auto', affine_step_control=True):
    ''' LDDMM for 3D to 2D slice mapping.

    Jointly estimates a 3D affine transform A and a 3D diffeomorphism phi that map the 3D source
    I (for example an atlas) onto the 2D target slice J. The map is of the form x -> A phi x.
    The parameters are those of LDDMM, with the differences below.

    Parameters
    ----------
    xI : list of 3 arrays
        Location of voxels in the source volume I, with channels along the first axis.
    xJ : list of 2 or 3 arrays
        Location of pixels in the target slice J (row, column). With two arrays the slice
        lies in the plane z = 0 of the target space.
    J : array
        Target slice with channels along the first axis, shape (C, H, W) or (C, 1, H, W).
    L, T, A : arrays
        Initial 3x3 linear part and 3 element translation, or 4x4 affine A (not both).
    v, xv : arrays
        Initial velocity field (nt, len(xv[0]), len(xv[1]), len(xv[2]), 3) and its grid.
    expand : float
        Factor to expand size of velocity field around image boundaries (default 1.25)
    sigmaR : float
        Standard deviation for regularization (default 2e6). Smaller sigmaR means a smoother
        transformation. Regularization is of the form 0.5/sigmaR^2 sum_{t=0}^{nt-1} int_X |L v_t|^2 dx
        with L = (1 - a^2 Laplacian)^p over all three axes, a sum over time steps without dt as
        in LDDMM. Before this version the 3D regularizer took the Fourier transform over only
        two of the three axes, which made it much stronger (and uneven); the default was 1e8.
        With the corrected regularizer, 1e8 regularizes much less than the old code did (in the
        3D tutorials the rms velocity became up to twice as large), and the new default 2e6
        gives deformations of about the old size. See CHANGES.md.
    pointsI, pointsJ, sigmaP :
        Not used: there is no landmark term in 3D, and passing points issues a warning.
        Use L_T_from_points to initialize L and T from landmarks instead.
    muA: torch tensor whose dimension is the same as the target image
        Defaults to None, which means we estimate this. If you provide a value, we will not estimate it.
        Give one value per channel of the target or a single value. For a grayscale target
        this should be a single value.
    muB: torch tensor whose dimension is the same as the target image
        Defaults to None, which means we estimate this. If you provide a value, we will not estimate it.
        Same format as muA.
    draw, num_threads, backend : keyword only
        As in LDDMM.
    affine_step_control : bool, keyword only
        As in LDDMM, but True by default: each affine parameter's step is scaled by a multiplier
        between 1e-3 and 1 that is halved when its gradient changes sign and grows by 5% per
        iteration (up to 1) otherwise. With the default epT, plain gradient steps make the
        translation jump back and forth by tens of micrometers per iteration until the end, so
        the result depends on rounding: in the tutorials, changing the initial translation in
        its last digit moved the cells by 24-84 um on average without the control, and by about
        1e-12 um with it. False gives the plain gradient steps of earlier versions.

    Returns a dictionary
    -------
    {
    'A': torch tensor
        4x4 affine transform
    'v': torch tensor
        Velocity field
    'xv': list of torch tensor
        Pixel locations in v
    'WM', 'WB', 'WA': torch tensor
        Resulting weights (matching, background, artifact), shape (1, H, W)
    'Xs': torch tensor
        Locations in the source space sampled for each target pixel, shape (1, H, W, 3)
    }
    A, v and Xs come from the last iteration's forward pass. With niter=0 the initial state is
    returned.

    '''
    if pointsI is not None or pointsJ is not None:
        warnings.warn('LDDMM_3D_to_slice has no landmark matching term: pointsI, pointsJ and sigmaP are '
                      'ignored. Use L_T_from_points to initialize L and T from landmarks instead.', stacklevel=2)
    problem = prepare_problem(3, xI, I, xJ, J, L=L, T=T, A=A, v=v, xv=xv,
                              a=a, p=p, expand=expand, nt=nt, niter=niter, diffeo_start=diffeo_start,
                              epL=epL, epT=epT, epV=epV, sigmaM=sigmaM, sigmaB=sigmaB, sigmaA=sigmaA,
                              sigmaR=sigmaR, sigmaP=sigmaP, device=device, dtype=dtype, muA=muA, muB=muB,
                              affine_step_control=affine_step_control)
    return run_lddmm(problem, draw=draw, num_threads=num_threads, backend=backend)
