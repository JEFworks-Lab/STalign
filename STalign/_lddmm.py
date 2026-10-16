'''LDDMM image registration: LDDMM (2D to 2D) and LDDMM_3D_to_slice (3D atlas to a 2D slice).'''
import warnings

import torch

try:
    from ._lddmm_common import SIGMA_DISP, prepare_problem, resolve_defaults, run_lddmm
except ImportError:
    from _lddmm_common import SIGMA_DISP, prepare_problem, resolve_defaults, run_lddmm


def LDDMM(xI, I, xJ, J, pointsI=None, pointsJ=None,
          L=None, T=None, A=None, v=None, xv=None,
          a=500.0, p=2.0, expand=2.0, nt=3,
          niter=None, diffeo_start=0, epL=None, epT=None, epV=None,
          sigmaM=1.0, sigmaB=2.0, sigmaA=5.0, sigmaR=None, sigmaP=2e1,
          device='cpu', dtype=torch.float64, muB=None, muA=None,
          *, draw='end', num_threads=None, backend='auto', affine_step_control=False, mixture=None,
          prefit='auto', sigma_disp=SIGMA_DISP, legacy_defaults=False):
    ''' Run LDDMM between a pair of images.

    This jointly estimates an affine transform A, and a diffeomorphism phi.
    The map is of the form x -> A phi x

    By default the fit starts with a coarse affine pre-alignment (prefit), and the
    regularization weight sigmaR and the step sizes epL, epT, epV are chosen from the images
    unless given (see their descriptions). Coordinates are assumed to be in micrometers.
    legacy_defaults=True turns these defaults off.

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
        Number of iterations of gradient descent optimization. Default None: 2000 (5000 with
        legacy_defaults=True).
    diffeo_start : int
        Number of iterations of gradient descent optimization for affine only, before nonlinear deformation.
    epL : float
        Gradient descent step size for linear part of affine.
    epT : float
        Gradient descent step size of translation part of affine.
    epV : float
        Gradient descent step size for velocity field.
        Default None for each of epL, epT, epV: the steps of earlier versions (2e-8, 2e-1, 2e3)
        times sigmaM^2 * 1.1 / G, where G estimates the curvature of the image matching term
        from the target's variance and the source's gradients under the initial affine (after
        the pre-fit). This adapts the steps to the image intensities and pixel size, where the
        old defaults could be far too large or too small. Each is then lowered where needed to
        keep gradient descent stable, from the curvature of the energy at the initial affine
        (the image term with the same assumptions as G, the landmarks and the regularizer):
        epV to at most 1 / (lambda_v + DV / sigmaR^2), where lambda_v is the largest eigenvalue
        of the kernel-smoothed curvature of the image and landmark terms in v (by subspace
        iteration) and DV the area of a velocity grid cell; epL and epT by a common factor, so
        that the largest eigenvalue of the affine curvature scaled by the steps (divided by 10,
        as during the deformation, unless diffeo_start > 0) is at most 0.5. This matters for
        small images (G grows with the tissue area, the stable epV does not), landmarks, a small
        sigma_disp and coordinates far from the origin. A given value is used as is.
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
        Regularization is of the form ``0.5/sigmaR^2 sum_{t=0}^{nt-1} int_X |L v_t|^2 dx`` with
        ``L = (1 - a^2 Laplacian)^p``, i.e. a sum over the nt time steps without a dt = 1/nt
        factor (nt times ``0.5/sigmaR^2 int_0^1 int_X |Lv|^2 dx dt``). Changing nt at a fixed
        sigmaR therefore changes the strength of the regularization.
        Default None: computed from sigma_disp as
        sigma_disp * sqrt(12 pi) * a * sqrt(nt) * (dx / 30) * sigmaM / s_J, where dx is the
        target pixel size in source units (under the affine after the pre-fit) and s_J the
        typical tissue intensity of the target. The prior then allows displacements of about
        sigma_disp micrometers whatever the pixel size, intensity scale, a and nt (at the
        defaults a = 500, nt = 3, a 30 um target with s_J = sigmaM gets 5e5, the previous
        default). A given value is used as is, as in earlier versions (5e5 with
        legacy_defaults=True).
    sigmaP: float
        Standard deviation for matching of points.
        Cost is of the form 0.5/sigmaP^2 sum_i (source_point_i - target_point_i)^2, where the source
        points are mapped to the target by the exact inverse of the fitted target-to-source map,
        i.e. as transform_points_source_to_target maps them after the fit. (Earlier versions used
        forward Euler steps, which invert that map only approximately; legacy_defaults=True
        uses them.)
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
    mixture : None, 'em' or dict, keyword only
        How the weights WM, WA, WB of the matching, artifact and background classes are found.
        None (default): as in earlier versions (WM starts at 0.5; from iteration 50, every 5
        iterations, an E-step with the fixed variances sigmaM^2, sigmaA^2, sigmaB^2).
        'em': a mixture EM whose matching variance vM is estimated from the residual. WM starts
        at 1. From iteration 50, every 5 iterations, each target pixel is weighted between the
        matching class (variance vM) and the background class (mean muB, the median of the
        target's border unless muB is given; variance (sigmaB/sigmaM)^2 vM), with the matching
        prior kept >= 0.5. Pixels that look like background while the deformed source does not
        match them, as where the target section is incomplete, then get a low matching weight.
        A dict selects 'em' and overrides some of its options: start (50), every (5), classes
        ('MB'; 'MAB' adds the artifact class with mean muA, the mean of the target unless muA is
        given, and variance (sigmaA/sigmaM)^2 vM; 'M' and 'MA' are also accepted), pi_min (0.5),
        pi_init ((0.9, 0.05, 0.05)), fix_muA (False), fix_muB (True: muB is not re-estimated).
        Only the weights change; the matching term keeps sigmaM. Worth trying where the target
        lacks tissue that the source has; it can make targets of uneven density worse.
        align_cells uses it by default.
    prefit : 'auto', True or False, keyword only
        Coarse affine pre-alignment before LDDMM. The initial affine (L, T or A; identity by
        default) is rotated about the target tissue's centroid in 12 steps of 30 degrees, each
        start is fitted as an affine map on images pooled to about 80 pixels across, and the
        best one is refined at twice that resolution; starts whose energies are within 1% of
        the best count as ties, and the smallest rotation among them wins. LDDMM then starts
        from the result. With landmarks the pre-fit includes the landmark term and keeps the
        initial orientation. 'auto' (default) runs it unless landmarks (pointsI, pointsJ) or a
        nonzero initial velocity field v are given; True always; False never. It is deterministic,
        and what it did is returned as 'prefit'. If the target has no tissue (a constant image)
        or the pre-fit fails, a warning is issued and the fit starts from the given affine.
    sigma_disp : float, keyword only
        Prior displacement scale in micrometers (default 94.03, which reproduces the previous
        default sigmaR=5e5 on 30 um images). Used only when sigmaR is None; larger values allow
        larger deformations. It assumes that coordinates are in micrometers.
    legacy_defaults : bool, keyword only
        If True, use the defaults of version 1.0 instead of those chosen from the data (see
        CHANGES.md): no pre-fit (unless prefit=True), niter=5000, epL=2e-8, epT=2e-1, epV=2e3,
        sigmaR=5e5, and the forward Euler map in the landmark term. The corrections listed in
        CHANGES.md still apply, so results can differ from those of version 1.0. Default False.

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
    'prefit': dict
        What the pre-fit did: 'ran' (bool) and 'A' (the affine LDDMM started from: the
        pre-fit's result, or the initial affine if it did not run); if it ran, 'A_start' (the
        initial affine), 'energy' (the final energy of every start), 'starts' (their rotations
        in degrees), 'best' (the index of the winner), 'pool' (the source and target pooling
        factors) and 'seconds'; if not, 'reason'.
    'settings': dict
        The niter, sigmaR, epL, epT and epV the fit used. Passing them with A=prefit['A'] (in
        place of L, T or A) and prefit=False (and legacy_defaults=True if it was given), the
        other arguments unchanged, repeats the fit exactly, without the data-driven choices.
    }
    A and v are the pair used by the last iteration's forward pass, so build_transform(xv, v, A)
    reproduces the final alignment. With niter=0 the initial state (after the pre-fit) is returned.

    '''
    problem = prepare_problem(2, xI, I, xJ, J, pointsI=pointsI, pointsJ=pointsJ, L=L, T=T, A=A, v=v, xv=xv,
                              a=a, p=p, expand=expand, nt=nt, niter=niter, diffeo_start=diffeo_start,
                              epL=epL, epT=epT, epV=epV, sigmaM=sigmaM, sigmaB=sigmaB, sigmaA=sigmaA,
                              sigmaR=sigmaR, sigmaP=sigmaP, device=device, dtype=dtype, muA=muA, muB=muB,
                              affine_step_control=affine_step_control, mixture=mixture)
    info = resolve_defaults(problem, prefit=prefit, sigma_disp=sigma_disp, legacy_defaults=legacy_defaults,
                            num_threads=num_threads)
    out = run_lddmm(problem, draw=draw, num_threads=num_threads, backend=backend)
    out.update(info)
    return out


def LDDMM_3D_to_slice(xI, I, xJ, J, pointsI=None, pointsJ=None,
                      L=None, T=None, A=None, v=None, xv=None,
                      a=500.0, p=2.0, expand=1.25, nt=3,
                      niter=5000, diffeo_start=0, epL=1e-6, epT=1e1, epV=None,
                      sigmaM=1.0, sigmaB=2.0, sigmaA=5.0, sigmaR=None, sigmaP=2e1,
                      device='cpu', dtype=torch.float64, muA=None, muB=None,
                      *, draw='end', num_threads=None, backend='auto', affine_step_control=True,
                      mixture=None, prefit='auto', sigma_disp=SIGMA_DISP, legacy_defaults=False):
    ''' LDDMM for 3D to 2D slice mapping.

    Jointly estimates a 3D affine transform A and a 3D diffeomorphism phi that map the 3D source
    I (for example an atlas) onto the 2D target slice J. The map is of the form x -> A phi x.
    The parameters are those of LDDMM, with the differences below. By default the fit starts
    with a coarse pre-alignment of the slice plane (prefit) and sigmaR is chosen from the
    images; coordinates are assumed to be in micrometers.

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
        Standard deviation for regularization. Smaller sigmaR means a smoother
        transformation. Regularization is of the form
        ``0.5/sigmaR^2 sum_{t=0}^{nt-1} int_X |L v_t|^2 dx`` with ``L = (1 - a^2 Laplacian)^p``
        over all three axes, a sum over time steps without dt as
        in LDDMM. Earlier versions took its Fourier transform over only two of the three axes,
        which made it much stronger (and uneven), with the default 1e8.
        Default None: computed from sigma_disp as
        sigma_disp * sqrt(64 pi) * a^1.5 * sqrt(nt) * (dx / 30) * sigmaM / s_J (dx: the
        target pixel size in atlas units under the affine after the pre-fit; s_J: the typical
        tissue intensity of the target), so that the prior allows displacements of about
        sigma_disp micrometers (1.1e7 and 1.85e6 in the MERFISH and STARmap tutorials). A given
        value is used as is (2e6 with legacy_defaults=True, which gives deformations of about
        the size of the old regularizer's). See CHANGES.md.
    niter, epL, epT, epV :
        Defaults 5000, 1e-6, 1e1 and None: 1e3 (no data-driven steps in 3D), lowered to
        sigmaR^2 / DV where needed to keep the velocity step stable for the regularizer
        (epV * DV / sigmaR^2 <= 1, DV the volume of a velocity grid cell). That happens only
        for a very small sigma_disp (a few um) with sigmaM well below the target's intensity
        scale. A given value is used as is.
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
    mixture : None, keyword only
        The mixture EM of LDDMM (mixture='em') is not available in 3D; anything but None raises
        NotImplementedError.
    prefit : 'auto', True or False, keyword only
        Coarse pre-alignment of the slice plane before LDDMM. Starting from the initial affine,
        the plane is tilted about its two in-plane axes by -6, 0 and 6 degrees each and shifted
        along its normal by -300, 0 and 300 um (27 starts); each start fits a rotation, an
        in-plane affine map and a shift on images pooled to about 80 pixels across, and the
        best one is refined at twice that resolution (ties within 1% go to the start closest to
        the initial plane). 'auto' (default) runs it unless a nonzero initial velocity field v
        is given; True always; False never. It is deterministic, and what it did is returned as
        'prefit'. If the target has no tissue (a constant image) or the pre-fit fails, a warning
        is issued and the fit starts from the given affine.
    sigma_disp : float, keyword only
        Prior displacement scale in micrometers (default 94.03), used when sigmaR is None.
    legacy_defaults : bool, keyword only
        If True, use fixed defaults instead of those chosen from the data (see CHANGES.md): no
        pre-fit (unless prefit=True), sigmaR=2e6 and epV=1e3. The corrections still apply (the
        regularizer over all three axes, the affine step control and the others listed in
        CHANGES.md), so results differ from those of version 1.0, whose default sigmaR=1e8
        regularizes much less with the corrected regularizer. Default False.

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
    'prefit', 'settings': dict
        As in LDDMM; the pre-fit's 'starts' are (tilt 1, tilt 2, offset) triples.
    }
    A, v and Xs come from the last iteration's forward pass. With niter=0 the initial state
    (after the pre-fit) is returned.

    '''
    if mixture is not None:
        raise NotImplementedError('LDDMM_3D_to_slice has no mixture EM (mixture=...); it exists in LDDMM only')
    if pointsI is not None or pointsJ is not None:
        warnings.warn('LDDMM_3D_to_slice has no landmark matching term: pointsI, pointsJ and sigmaP are '
                      'ignored. Use L_T_from_points to initialize L and T from landmarks instead.', stacklevel=2)
    problem = prepare_problem(3, xI, I, xJ, J, L=L, T=T, A=A, v=v, xv=xv,
                              a=a, p=p, expand=expand, nt=nt, niter=niter, diffeo_start=diffeo_start,
                              epL=epL, epT=epT, epV=epV, sigmaM=sigmaM, sigmaB=sigmaB, sigmaA=sigmaA,
                              sigmaR=sigmaR, sigmaP=sigmaP, device=device, dtype=dtype, muA=muA, muB=muB,
                              affine_step_control=affine_step_control)
    info = resolve_defaults(problem, prefit=prefit, sigma_disp=sigma_disp, legacy_defaults=legacy_defaults,
                            num_threads=num_threads)
    out = run_lddmm(problem, draw=draw, num_threads=num_threads, backend=backend)
    out.update(info)
    return out
