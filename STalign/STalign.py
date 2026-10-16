''' Tools for aligning spatial transcriptomics data

The LDDMM fits (LDDMM, LDDMM_3D_to_slice, and align_cells for two sets of cells),
rasterization and the transforms of fitted mappings are implemented in private modules next
to this file (_lddmm*.py, _cells.py, _raster.py, _transforms.py) and exposed here.  The LDDMM
fits use a compiled C++ engine when it is installed and pure torch otherwise; backend_info()
reports which.
'''

import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch

# The second import is for the tutorials, which put this folder on sys.path and import this
# file directly (the private modules are then top-level modules too).
try:
    from . import _raster, _transforms
    from ._cells import align_cells
    from ._lddmm import LDDMM, LDDMM_3D_to_slice
    from ._lddmm_common import backend_info
except ImportError:
    import _raster
    import _transforms
    from _cells import align_cells
    from _lddmm import LDDMM, LDDMM_3D_to_slice
    from _lddmm_common import backend_info

# These are part of this module's API: present them as defined here (like numpy's
# set_module), so that help() and the documentation, which lists the functions of this
# module, show them with it.
for _f in (LDDMM, LDDMM_3D_to_slice, align_cells, backend_info):
    _f.__module__ = __name__
del _f


def normalize(arr, t_min=0, t_max=1):
    """Linearly normalizes an array between two specifed values.

    Parameters
    ----------
    arr : numpy array
        array to be normalized
    t_min : int or float
        Lower bound of normalization range
    t_max : int or float
        Upper bound of normalization range

    Returns
    -------
    norm_arr : numpy array
        Array of the same shape as arr with normalized values.
        A constant arr (zero range) maps to t_min.

    """
    return _raster.normalize(arr, t_min=t_min, t_max=t_max)



def rasterize(x, y, g=np.ones(1), dx=30.0, blur=None, expand=1.1, draw=10000, wavelet_magnitude=False,use_windowing=True,
              *, blur_um=None):
    ''' Rasterize a spatial transcriptomics dataset into a density image

    Each cell adds a Gaussian kernel centered on the cell, normalized to unit
    sum over the pixels where it is evaluated and multiplied by the cell's
    weight g.
    
    Parameters
    ----------
    x : numpy array of length N
        x location of cells
    y : numpy array of length N
        y location of cells
    g : numpy array of length N
        RNA count of cells
        If not given, density image is created.  Unless g is all ones it is
        min-max normalized to [0, 1] (see normalize), so cells at the minimum
        of g contribute nothing, and a constant g gives an image of zeros.
        Use rasterize_with_signal for an image of unnormalized values.
    dx : float
        Pixel size to rasterize data (default 30.0, in same units as x and y)
    blur : float or list of floats
        Width of the Gaussian interpolation kernel, in pixels.  The kernel
        standard deviation is 2*blur pixels (2*blur*dx in the units of x and
        y).  Use a list to do multi scale (one channel per blur).  Defaults
        to 1.0, unless blur_um is given.
    expand : float
        Factor to expand sampled area beyond cells. Defaults to 1.1.
    draw : int
        If nonzero, draw the image in a new figure and return its handle.
        Defaults to 10000.  The figure is drawn once, after all cells are
        added (earlier versions redrew it every draw cells).
    wavelet_magnitude : bool
        If True, take the absolute value of difference between scales for raster images.
        When using this option blur should be sorted from greatest to least.
    use_windowing : bool
        If True (default), each cell's kernel is evaluated on a (2r+1) x (2r+1)
        pixel window around the cell's nearest pixel (clipped at the image
        border), r = ceil(4*max(blur)) pixels (about 2 standard deviations),
        and normalized to unit sum on that window.  The window holds about 95%
        of the Gaussian, so the kernel acts like one about 4% narrower.
        If False, the untruncated kernel is evaluated on the whole image.
    blur_um : float or list of floats, keyword only
        The kernel standard deviation in the units of x and y (for example
        micrometers), instead of blur: blur = blur_um/(2*dx).  The kernel
        then has the same physical width at any pixel size dx.  Give either
        blur or blur_um, not both.
    
        
    Returns
    -------
    X  : numpy array
        Locations of pixels along the x axis
    Y  : numpy array
        Locations of pixels along the y axis
    M : numpy array
        A rasterized image with len(blur) channels along the first axis
    fig : matplotlib figure handle
        Only returned if draw is nonzero: handle to the figure of the image.
        
    Raises
    ------    
    Exception 
        If wavelet_magnitude is set to true but blur is not sorted from greatest to least.
    ValueError
        If both blur and blur_um are given.
    IndexError
        If the image is empty (e.g. a single cell, or cells on a line).
        
    Notes
    -----
    Cells with NaN coordinates (possible with pandas inputs, whose min and
    max skip NaN) are ignored, with a warning.
    
    Examples
    --------
    Rasterize a dataset at 30 micron pixel size, with three kernels.
    
    >>> X,Y,M,fig = STalign.rasterize(x,y,dx=30.0,blur=[2.0,1.0,0.5],draw=10000)
    
    Rasterize a dataset at 30 micron pixel size, with three kernels, using difference between scales.
    
    >>> X,Y,M,fig = STalign.rasterize(x,y,dx=30.0,blur=[2.0,1.0,0.5],draw=10000, wavelet_magnitude=True)

    Rasterize a dataset at 15 micron pixel size with a kernel of standard deviation 30 microns
    (the same as blur=1.0).

    >>> X,Y,M = STalign.rasterize(x,y,dx=15.0,blur_um=30.0,draw=0)
        
        
    '''
    return _raster.rasterize(x, y, g=g, dx=dx, blur=blur, expand=expand, draw=draw,
                             wavelet_magnitude=wavelet_magnitude, use_windowing=use_windowing, blur_um=blur_um)
    

    
def rasterize_with_signal(x, y, s=None, dx=30.0, blur=None, expand=1.1, draw=0, wavelet_magnitude=False,use_windowing=True,
                          *, blur_um=None):
    ''' Rasterize a spatial transcriptomics dataset with a signal per cell

    Each cell adds a Gaussian kernel centered on the cell, normalized to unit
    sum over the pixels where it is evaluated, divided by dx**2 and
    multiplied by the cell's signal value(s), so the image is a signal
    density per unit area.  See rasterize for the kernel and the window.
    
    Parameters
    ----------
    x : numpy array of length N
        x location of cells
    y : numpy array of length N
        y location of cells
    s : numpy array of length N, or N by M for M signals
        Signal value of each cell, used as is (not normalized).  Signals may
        be signed, and the image keeps their sign.  Each of the M columns
        gives one channel.  If None, a density image is created.
    dx : float
        Pixel size to rasterize data (default 30.0, in same units as x and y)
    blur : float or list of floats
        Width of the Gaussian interpolation kernel, in pixels.  The kernel
        standard deviation is 2*blur pixels (2*blur*dx in the units of x and
        y).  A list does multi scale, which is only possible if s is None.
        Defaults to 1.0, unless blur_um is given.
    expand : float
        Factor to expand sampled area beyond cells. Defaults to 1.1.
    draw : int
        If nonzero, draw the image in a new figure and return its handle.
        Defaults to 0.  The figure is drawn once, after all cells are added.
    wavelet_magnitude : bool
        If True, take the absolute value of difference between scales for raster images.
        When using this option blur should be sorted from greatest to least.
        With a signal (a single blur) this only takes the absolute value.
    use_windowing : bool
        If True (default), evaluate each kernel on a window of about 2
        standard deviations; if False, on the whole image (see rasterize).
    blur_um : float or list of floats, keyword only
        The kernel standard deviation in the units of x and y, instead of
        blur: blur = blur_um/(2*dx).  Give either blur or blur_um, not both.
    
        
    Returns
    -------
    X  : numpy array
        Locations of pixels along the x axis
    Y  : numpy array
        Locations of pixels along the y axis
    M : numpy array
        A rasterized image with channels along the first axis: one per
        column of s, or len(blur) if s is None
    fig : matplotlib figure handle
        Only returned if draw is nonzero: handle to the figure of the image.
        
    Raises
    ------    
    Exception 
        If s is given with more than one blur, or if wavelet_magnitude is set
        to true but blur is not sorted from greatest to least.
    ValueError
        If both blur and blur_um are given.
    IndexError
        If s has fewer rows than there are cells, or if the image is empty.
        
        
    
    Examples
    --------
    Rasterize the total counts and the counts of two genes at 30 micron pixel size.
    
    >>> X,Y,M = STalign.rasterize_with_signal(x,y,s=np.stack((total,gene1,gene2),-1),dx=30.0,blur=1.0)
    
    Rasterize the density at three scales, using difference between scales.
    
    >>> X,Y,M = STalign.rasterize_with_signal(x,y,dx=30.0,blur=[2.0,1.0,0.5],wavelet_magnitude=True)
        
        
    '''
    return _raster.rasterize_with_signal(x, y, s=s, dx=dx, blur=blur, expand=expand, draw=draw,
                                         wavelet_magnitude=wavelet_magnitude, use_windowing=use_windowing,
                                         blur_um=blur_um)
    
        


def rasterizePCA(x, y, G):
    """Rasterize a spatial transcriptomics dataset into a density image and perform PCA on it.
    
    Parameters
    ----------
    x : numpy array of length N
        x location of cells
    y : numpy array of length N
        y location of cells
    G : pandas Dataframe of shape (N,M)
        gene expression level of cells for M genes
       
    Returns
    -------
    X : numpy array
        A rasterized image for each gene with the channel along the first axis
    Y : numpy array
        A rasterized image unraveled to 1-D for each gene; data is centered
    W : numpy array
        The eigenvalues for the principal components in descending order
    V : numpy array
        The normalized eigenvectors for the principal components
        V[:, i] corresponds to eigenvalue at W[i]
    Z : numpy array
        Y rotated to align with the principal component axes (Z = V.T @ Y)
    nrows : int
        Row dimension of rasterized image
    ncols : int
        Column dimension of rasterized image

    Notes
    -----
    Each value/row at the same index in x, y, and G should all correspond to the same cell.
    x[i] <-> y[i] <-> G[i,:]

    Each gene is rasterized with rasterize(x, y, g, dx=30.0, blur=1.0), which
    min-max normalizes its counts; a constant gene gives an image of zeros.

    """

    nrows=0
    ncols=0

    for i in range(G.shape[1]):
        g = np.array(G.iloc[:,i])

        XI,YI,I = rasterize(x,y,g,dx=30.0,blur=1.0,expand=1.1, draw=0, use_windowing=True)

        if(i==0):
            # dimensions
            nrows=YI.size
            ncols=XI.size
            X = np.empty([G.shape[1], nrows, ncols])
            Y = np.empty([G.shape[1], nrows*ncols])

        # centers data
        X[i] = I[0]
        I_ = I.ravel()
        meanI = np.mean(I_)
        I_ -= meanI
        Y[i] = I_
        
        if(i % 50 == 0):
            print(f"{i} out of {G.shape[1]} genes rasterized.")
        
    S = np.cov(Y) # computes covariance matrix
    W,V = np.linalg.eigh(S) # W = eigenvalues, V = eigenvectors
    
    # reverses order to make it descending by eigenvalue
    W = W[::-1]
    V = V[:,::-1]
    Z = V.T @ Y
    
    return X, Y, W, V, Z, nrows, ncols



def make_scree(W, name, p=6):
    """Create a scree plot for a given set of eigenvalues.
    
    Parameters
    ----------
    W : numpy array
        Eigenvalues in descending order
    name : str
        Name of the dataset the eigenvalue originate from
        Will be included in plot title
    p : int
        Number of eigenvalue from W to plot. Defaults to 6
        
    Returns
    -------
    fig : matplotlib Figure
        Figure handle for scree plot
        
    Raises
    ------
    ValueError
        If p is larger than the number to eigenvalues in W.
    
    """
    
    if(p > W.size):
        raise ValueError("Cannot plot more eigenvalues than what is given.")
    
    fig, ax = plt.subplots(figsize=(8,6))
    
    ax.bar(range(p),W[:p],width=.9,tick_label=range(1,p+1))
    ax.plot(W[:p],'ko-',linewidth=1)

    ax.set_title("Scree Plot: First %s PCs (%s)" %(p,name), fontsize=18,fontweight='bold')
    ax.set_xlabel('Principal Components')
    ax.set_ylabel("Eigenvalues")
        
    return fig



def saveRasters(X, Y, W, V, Z, scree_fig, nrows, ncols, name, path):
    """
    Create an RGB image principal components 1-3 and 4-6.
    Save numpy arrays necessary for rasterizing and performing PCA.
    Save scree plot and RGB image as jpeg, and their figure handles.
    
    Parameters
    ----------
    X : numpy array
        A rasterized image for each gene with the channel along the first axis
    Y : numpy array
        A rasterized image unraveled to 1-D for each gene; data is centered
    W : numpy array
        The eigenvalues for the principal components in descending order
    V : numpy array
        The normalized eigenvectors for the principal components
        V[:, i] corresponds to eigenvalue at W[i]
    Z : numpy array
        X rotated to align with the principal component axes
    scree_fig : matplotlib Figure
        Figure handle for scree plot
    nrows : int
        Row dimension of rasterized image
    ncols : int
        Column dimension of rasterized image
    name : str
        Name of the dataset the arrays originate from
    path : str
        Path to folder where files should be saved (created if needed).
        Files are named <name>_screefig.jpg, <name>_rgbfig.jpg and <name>_arrays.npz.

    Returns
    -------
    None

    """
    
    I_pca = Z.reshape((Z.shape[0], nrows, ncols))
    I_rgb = np.array(I_pca[:6].transpose(1,2,0))
    I_rgb[...,:] = normalize(I_rgb[...,:])
    
    fig,axs = plt.subplots(1,2)
    axs[0].imshow(I_rgb[:,:,:3])
    axs[1].imshow(I_rgb[:,:,3:])
    
    fig.suptitle("Principal Components in RGB (%s)" %name, fontsize=18,fontweight='bold')
    axs[0].set_title("PCs 1-3")
    axs[1].set_title("PCs 4-6")
    
    if path:
        os.makedirs(path, exist_ok=True)
    scree_fig.savefig(os.path.join(path, f"{name}_screefig.jpg"))
    fig.savefig(os.path.join(path, f"{name}_rgbfig.jpg"))
    np.savez(os.path.join(path, f"{name}_arrays"), X=X, Y=Y, W=W, V=V, Z=Z, I_pca=I_pca, I_rgb=I_rgb,
             screefig=scree_fig, rgbfig = fig)
    
    print(f"Saved all {name} files!\n")
    
    

    
def interp(x,I,phii,**kwargs):
    '''
    Interpolate the 2D image I, with regular grid positions stored in x (1d arrays),
    at the positions stored in phii (3D arrays with first channel storing component)    
    
    Parameters
    ----------
    x : list of arrays
        List of arrays storing the pixel locations along each image axis. convention is row column order not xy.
    I : array
        Image array. First axis should contain different channels.
    phii : array
        Sampling array. First axis should contain sample locations corresponding to each axis.
    **kwargs : dict
        Other arguments fed into the torch interpolation function torch.nn.grid_sample
        
    
    Returns
    -------
    out : torch tensor
            The image I resampled on the points defined in phii.
    
    Notes
    -----
    Convention is to use align_corners=True.
    
    This uses the torch library.
    '''
    return _transforms.interp(x,I,phii,**kwargs)

def interp3D(x,I,phii,**kwargs):
    '''
    Interpolate the 3D image I, with regular grid positions stored in x (1d arrays),
    at the positions stored in phii (4D arrays with first channel storing component)    
    
    Parameters
    ----------
    x : list of arrays
        List of arrays storing the pixel locations along each image axis. convention is row column order not xy.
    I : array
        Image array. First axis should contain different channels.
    phii : array
        Sampling array. First axis should contain sample locations corresponding to each axis.
    **kwargs : dict
        Other arguments fed into the torch interpolation function torch.nn.grid_sample
        
    
    Returns
    -------
    out : torch tensor
            The image I resampled on the points defined in phii.
    
    Notes
    -----
    Convention is to use align_corners=True.
    
    This uses the torch library.
    '''
    return _transforms.interp3D(x,I,phii,**kwargs)


def clip(I):
    ''' clip an arrays values between 0 and 1.  Useful for visualizing
    
    Parameters
    ----------
    I : torch tensor
        A torch tensor to clip.
    
    Returns
    -------
    Ic : torch tensor
        Clipped torch tensor.
    
    '''
    Ic = torch.clone(I)
    Ic[Ic<0]=0
    Ic[Ic>1]=1
    return Ic



# timesteps will be along the first axis
def v_to_phii(xv,v):
    ''' Integrate a velocity field over time to return a position field (diffeomorphism).
    
    Parameters
    ----------
    xv : list of torch tensor 
        List of 1D tensors describing locations of sample points in v
    v : torch tensor
        4D (nt,v0,v1,2) velocity field, the layout LDDMM returns. The component-first
        layout (nt,2,v0,v1) is also accepted (on a 2x2 grid, where the two are ambiguous,
        the LDDMM layout is assumed).
    
    Returns
    -------
    phii: torch tensor
        Inverse map (position field) computed by method of characteristics, (2,v0,v1)

    Notes
    -----
    This semi-Lagrangian scheme approximates the map build_transform(xv,v,A,'b') computes
    for an identity A, up to interpolation error. Use build_transform for exactly the map
    LDDMM fitted.

    '''
    return _transforms.v_to_phii(xv,v)

def v_to_phii_3D(xv,v):
    ''' Integrate a 3D velocity field over time to return a 3D position field (diffeomorphism).
    
    Parameters
    ----------
    xv : list of torch tensor 
        List of 1D tensors describing locations of sample points in v
    v : torch tensor
        5D (nt,v0,v1,v2,3) velocity field, the layout LDDMM_3D_to_slice returns. The
        component-first layout (nt,3,v0,v1,v2) is also accepted (on a 3x3x3 grid, where the
        two are ambiguous, the LDDMM layout is assumed).
    
    Returns
    -------
    phii: torch tensor
        Inverse map (position field) computed by method of characteristics, (3,v0,v1,v2)

    Notes
    -----
    This semi-Lagrangian scheme approximates the map build_transform3D(xv,v,A,'b') computes
    for an identity A, up to interpolation error. Use build_transform3D for exactly the map
    LDDMM_3D_to_slice fitted.

    '''
    return _transforms.v_to_phii_3D(xv,v)

def to_A(L,T):
    ''' Convert a linear transform matrix and a translation vector into an affine matrix.
    
    Parameters
    ----------
    L : torch tensor
        2x2 linear transform matrix
        
    T : torch tensor
        2 element translation vector (note NOT 2x1)
        
    Returns
    -------
    
    A : torch tensor
        Affine transform matrix
        
        
    '''
    O = torch.tensor([0.,0.,1.],device=L.device,dtype=L.dtype)
    A = torch.cat((torch.cat((L,T[:,None]),1),O[None]))
    return A
def to_A_3D(L,T):
    ''' Convert a linear transform matrix and a translation vector into an affine matrix.
    
    Parameters
    ----------
    L : torch tensor
        3x3 linear transform matrix
        
    T : torch tensor
        3 element translation vector (note NOT 2x1)
        
    Returns
    -------
    
    A : torch tensor
        Affine transform matrix
        
        
    '''
    O = torch.tensor([0.,0.,0.0,1.],device=L.device,dtype=L.dtype)
    A = torch.cat((torch.cat((L,T[:,None]),1),O[None]))
    return A

def extent_from_x(xJ):
    ''' Given a set of pixel locations, returns an extent 4-tuple for use with np.imshow.
    
    Note inputs are locations of pixels along each axis, i.e. row column not xy.
    
    Parameters
    ----------
    xJ : list of torch tensors
        Location of pixels along each axis
    
    Returns
    -------
    extent : tuple
        (xmin, xmax, ymin, ymax) tuple
    
    Examples
    --------
    
    >>> extent_from_x(xJ)
    >>> fig,ax = plt.subplots()
    >>> ax.imshow(J,extent=extentJ)
    
    '''
    dJ = [x[1]-x[0] for x in xJ]
    extentJ = ( (xJ[1][0] - dJ[1]/2.0).item(),
               (xJ[1][-1] + dJ[1]/2.0).item(),
               (xJ[0][-1] + dJ[0]/2.0).item(),
               (xJ[0][0] - dJ[0]/2.0).item())
    return extentJ

def L_T_from_points(pointsI,pointsJ):
    '''
    Compute an affine transformation from points.
    
    Note for an affine transformation (6dof) we need 3 points.
    
    Outputs, L,T should be rconstructed blockwize like [L,T;0,0,1]
    
    Parameters
    ----------
    pointsI : array
        An Nx2 array of floating point numbers describing source points in ROW COL order (not xy)
    pointsJ : array
        An Nx2 array of floating point numbers describing target points in ROW COL order (not xy)
    
    Returns
    -------
    L : array
        A 2x2 linear transform array.
    T : array
        A 2 element translation vector
    '''
    if pointsI is None or pointsJ is None:
        raise Exception('Points are set to None')
        
    nI = pointsI.shape[0]
    nJ = pointsJ.shape[0]
    if nI != nJ:
        raise Exception(f'Number of pointsI ({nI}) is not equal to number of pointsJ ({nJ})')
    if pointsI.shape[1] != 2:
        raise Exception(f'Number of components of pointsI ({pointsI.shape[1]}) should be 2')
    if pointsJ.shape[1] != 2:
        raise Exception(f'Number of components of pointsJ ({pointsJ.shape[1]}) should be 2')
    # transformation model
    if nI < 3:
        # translation only 
        L = np.eye(2)
        T = np.mean(pointsJ,0) - np.mean(pointsI,0)
    else:
        # we need an affine transform
        pointsI_ = np.concatenate((pointsI,np.ones((nI,1))),1)
        pointsJ_ = np.concatenate((pointsJ,np.ones((nI,1))),1)
        II = pointsI_.T@pointsI_
        IJ = pointsI_.T@pointsJ_
        A = (np.linalg.inv(II)@IJ).T        
        L = A[:2,:2]
        T = A[:2,-1]
    return L,T


def build_transform(xv,v,A,direction='b',XJ=None,exact_inverse=True):
    ''' Create sample points to transform source to target from affine and velocity.
    
    Parameters
    ----------
    xv : list of array
        Sample points for velocity
    v : array
        time dependent velocity field, (nt,v0,v1,2). None for an affine only transform.
    A : array
        Affine transformation matrix
    direction : char
        'f' for forward and 'b' for backward. 'b' is default and is used for transforming images.
        'f' is used for transforming points.
    XJ : array
        Sample points for target (meshgrid with ij index style).  Defaults to None 
        to keep sampling on the xv. Can also be a list of 1D arrays (one per axis), or an
        array of points with components along the last axis.
    exact_inverse : bool
        Only used for direction 'f'. If True (default), the forward map is the exact inverse
        of the backward map, to about 1e-6 of a velocity grid spacing. If False, it is the
        forward Euler integration used by earlier versions of STalign, which inverts the
        backward map only up to a time discretization error (up to tens of microns on the
        tutorial alignments).
    
    Returns
    -------
    Xs : torch tensor
        Sample points in mehsgrid format, (...,2).

    Notes
    -----
    The backward map ('b', target to source) is the map LDDMM fits: the inverse affine
    transform, followed by explicit Euler steps of -v[t]/nt for t = nt-1 down to 0. The
    exact forward map inverts each step by fixed-point iteration (an implicit Euler step of
    the forward flow), starting from the forward Euler step. A RuntimeWarning is raised for
    points that do not converge, which can only happen if v is too large for nt time steps.

    Inputs can be numpy arrays or torch tensors, with integer or floating point values. The
    computation uses the floating point dtype of A and v (float64 if both are integer) and
    the device of v.
    
    '''
    return _transforms.build_transform(xv,v,A,direction=direction,XJ=XJ,exact_inverse=exact_inverse)

def build_transform3D(xv,v,A,direction='b',XJ=None,exact_inverse=True):
    ''' Create sample points to transform source to target from affine and velocity.
    
    Parameters
    ----------
    xv : list of array
        Sample points for velocity
    v : array
        time dependent velocity field, (nt,v0,v1,v2,3). None for an affine only transform.
    A : array
        Affine transformation matrix
    direction : char
        'f' for forward and 'b' for backward. 'b' is default and is used for transforming images.
        'f' is used for transforming points.
    XJ : array
        Sample points for target (meshgrid with ij index style).  Defaults to None 
        to keep sampling on the xv. Can also be a list of 1D arrays (one per axis), or an
        array of points with components along the last axis, for example (N,3).
    exact_inverse : bool
        Only used for direction 'f'. If True (default), the forward map is the exact inverse
        of the backward map, to about 1e-6 of a velocity grid spacing. If False, it is the
        forward Euler integration, which inverts the backward map only up to a time
        discretization error.
    
    Returns
    -------
    Xs : torch tensor
        Sample points in mehsgrid format, (...,3).

    Notes
    -----
    After LDDMM_3D_to_slice the source is the atlas and the target is the slice: 'b' maps
    slice positions to the atlas and 'f' maps atlas positions to the slice. See
    build_transform for how the two maps are computed.
    
    '''
    return _transforms.build_transform3D(xv,v,A,direction=direction,XJ=XJ,exact_inverse=exact_inverse)

def transform_image_source_with_A(A, XI, I, XJ):
    '''
    Transform an image with an affine matrix
    
    Parameters
    ----------
    
    A  : torch tensor
         Affine transform matrix
        
    XI : list of numpy arrays
         List of arrays storing the pixel location in image I along each image axis. 
         convention is row column order not xy. i.e, 
         locations of pixels along the y axis (rows) followed by
         locations of pixels along the x axis (columns)  
    
    I  : numpy array
         A rasterized image with len(blur) channels along the first axis
        
    XJ : list of numpy arrays
         List of arrays storing the pixel location in image I along each image axis. 
         convention is row column order not xy. i.e, 
         locations of pixels along the y axis (rows) followed by
         locations of pixels along the x axis (columns)         
    
    Returns
    -------
    AI : torch tensor
        image I after affine transformation A, with channels along first axis
              
    '''
    return _transforms.transform_image_source_with_A(A, XI, I, XJ)

def transform_image_source_to_target(xv,v,A,xI,I,XJ=None):
    '''
    Transform an image from source to target space, by sampling it at the backward map.

    Parameters
    ----------
    xv : list of array
        Sample points for velocity
    v : array
        time dependent velocity field, (nt,v0,v1,2). None for an affine only transform.
    A : array
        Affine transformation matrix
    xI : list of arrays
        Locations of the pixels of I along each axis (row column order, not xy).
    I : array
        Source image, channels along the first axis.
    XJ : list of arrays
        Locations of the output pixels (see build_transform). Defaults to the velocity grid.

    Returns
    -------
    phiI : torch tensor
        I in target space, channels along the first axis.

    Notes
    -----
    I is sampled with bilinear interpolation, clamped at its border. Inputs can be numpy
    arrays or torch tensors of any numeric dtype.
    '''
    return _transforms.transform_image_source_to_target(xv,v,A,xI,I,XJ=XJ)

def transform_image_target_to_source(xv,v,A,xJ,J,XI=None,exact_inverse=True):
    '''
    Transform an image from target to source space, by sampling it at the forward map.

    Parameters
    ----------
    xv : list of array
        Sample points for velocity
    v : array
        time dependent velocity field, (nt,v0,v1,2). None for an affine only transform.
    A : array
        Affine transformation matrix
    xJ : list of arrays
        Locations of the pixels of J along each axis (row column order, not xy).
    J : array
        Target image, channels along the first axis.
    XI : list of arrays
        Locations of the output pixels (see build_transform). Defaults to the velocity grid.
    exact_inverse : bool
        If True (default), use the exact inverse of the backward map; if False, the forward
        Euler map of earlier versions (see build_transform).

    Returns
    -------
    phiiJ : torch tensor
        J in source space, channels along the first axis.

    Notes
    -----
    J is sampled with bilinear interpolation, clamped at its border. Inputs can be numpy
    arrays or torch tensors of any numeric dtype.
    '''
    return _transforms.transform_image_target_to_source(xv,v,A,xJ,J,XI=XI,exact_inverse=exact_inverse)

def transform_points_source_to_target(xv,v,A,pointsI,exact_inverse=True):
    '''
    Transform points from source to target space.  Note points are in row column order, not xy.

    Parameters
    ----------
    xv : list of array
        Sample points for velocity
    v : array
        time dependent velocity field, (nt,*nv,d). None for an affine only transform.
    A : array
        Affine transformation matrix, (d+1,d+1). With d=3 (after LDDMM_3D_to_slice) this maps
        atlas points to the slice.
    pointsI : array
        (N,d) points in source space. Numpy array or torch tensor, integer or floating point;
        it is not modified.
    exact_inverse : bool
        If True (default), use the exact inverse of the backward map LDDMM fits; if False, the
        forward Euler map of earlier versions (see build_transform).

    Returns
    -------
    pointsIt : torch tensor
        (N,d) points in target space.
    '''
    return _transforms.transform_points_source_to_target(xv,v,A,pointsI,exact_inverse=exact_inverse)

def transform_points_target_to_source(xv,v,A,pointsI):
    '''
    Transform points from target to source space.  Note points are in row column order, not xy.

    This is the backward map LDDMM fits, identical to build_transform(xv,v,A,direction='b')
    evaluated at the points.

    Parameters
    ----------
    xv : list of array
        Sample points for velocity
    v : array
        time dependent velocity field, (nt,*nv,d). None for an affine only transform.
    A : array
        Affine transformation matrix, (d+1,d+1).
    pointsI : array
        (N,d) points in target space. Numpy array or torch tensor, integer or floating point;
        it is not modified.

    Returns
    -------
    pointsIt : torch tensor
        (N,d) points in source space.
    '''
    return _transforms.transform_points_target_to_source(xv,v,A,pointsI)

def calculate_tre(pointsI, pointsJ):
    ''' Target registration error (TRE) between two sets of corresponding points

    Parameters
    ----------
    pointsI : array or torch tensor (N x 2)
        Points, e.g. the output of transform_points_source_to_target
    pointsJ : array or torch tensor (N x 2)
        Corresponding points

    Returns
    -------
    meanTRE : float
        Mean distance between corresponding points
    stdTRE : float
        Standard deviation of the distances
    '''
    pointsI = pointsI.detach().cpu().numpy() if isinstance(pointsI, torch.Tensor) else np.asarray(pointsI)
    pointsJ = pointsJ.detach().cpu().numpy() if isinstance(pointsJ, torch.Tensor) else np.asarray(pointsJ)
    TRE_i = np.sqrt(np.sum((pointsI - pointsJ)**2,axis=1))
    meanTRE = np.mean(TRE_i)
    stdTRE = np.std(TRE_i)
    return meanTRE, stdTRE


#Download and store ontology
def download_aba_ontology(url,file_name): 
    ''' Create 3D altas ontology
        
        Parameters
        ----------
        url : Link to url contain atlas ontology
        file_name : File name to save atlas ontology
        
        Returns
        -------
        ontology_name : File name to store ontologies
        namesdict : Dictionary of all brain rgeion names

        Raises
        ------
        requests.HTTPError
            If the download fails (nothing is written then).
    '''
    # used only by the atlas helpers, so imported here rather than with this module
    import pandas as pd
    import requests

    r = requests.get(url, timeout=(30, 300))
    print(r)
    r.raise_for_status()
    with open(file_name,'w',encoding='utf-8') as f:
        f.write(r.text)
    ontology_name = file_name

    O = pd.read_csv(ontology_name)

    # store the ontology in a dictionary
    namesdict = defaultdict(lambda: 'unk')
    namesdict[0] = 'bg'

    # we need to add the structure names from the structure_id
    for i,n in zip(O['id'],O['acronym']):
        namesdict[i] = n
    return ontology_name,namesdict

def download_aba_image_labels(imageurl, labelurl, imagefile, labelfile):
    ''' Create 3D altas image and region annotations
            
        Parameters
        ----------
        imageurl : array
            Link containing atlas cell stained image.
        labelurl : array
            Link containing atlas altas images for each voxel in imageurl.
        imagefile : array
            File location to save imageurl information.
        labelfile : array
            File location to save labelurl information.
        
        Returns
        -------
        imagefile : array
            File location to save imageurl information.
        labelfile : array
            File location to save labelurl information.

        Raises
        ------
        requests.HTTPError
            If a download fails (the file is not written then).
    '''
    import requests # used only by the atlas helpers, so imported here

    def download(url, path):
        # write to a temporary file and rename it when complete, so that a
        # failed download never leaves a broken file at path
        tmp = os.fspath(path) + '.part'
        with requests.get(url, stream=True, timeout=(30, 300)) as r:
            r.raise_for_status()
            with open(tmp, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    if chunk: # filter out keep-alive new chunks
                        f.write(chunk)
        os.replace(tmp, path)

    download(imageurl, imagefile)
    download(labelurl, labelfile) # e.g. 'http://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/annotation/ccf_2017/annotation_50.nrrd'
    return imagefile, labelfile

def analyze3Dalign(labelfile, xv,v,mat, xJ, dx, scale_x, scale_y, x, y, X_, Y_, namesdict, device='cpu'):
    ''' Create dataframe with region annotations and 3D coordinates of target brain slice

    Each cell is mapped into the atlas with the backward map fitted by LDDMM_3D_to_slice
    (build_transform3D, direction 'b'), evaluated at the cell's own position, and gets the
    label of the nearest atlas voxel at that position.

    Parameters
    ----------
    labelfile : str
        File name of 3D atlas with region annotations (nrrd).
    xv : list of array
        Sample points for velocity
    v : array
        time dependent velocity field
    mat : array
        Affine transformation matrix
    xJ : list of arrays
        Pixel locations of the rasterized target image. Not used (kept for compatibility).
    dx : float
        Step of rasterized image. Not used (kept for compatibility).
    scale_x : double
        Value that x positions of atlas were scaled by. Not used (kept for compatibility).
    scale_y : double
        Value that y positions of atlas were scaled by. Not used (kept for compatibility).
    x : array
        X positions of target brain slice
    y : array
        Y positions of target brain slice
    X_ : array
        Not used (kept for compatibility).
    Y_ : array
        Not used (kept for compatibility).
    namesdict : Dictionary
        Dictionary of brain regions corresponding to region numbers
    device : str
        Not used (kept for compatibility); the computation runs on the device of v.

    Returns
    -------
    df : DataFrame
        One row per cell of the target brain slice: its 3D coordinates in the atlas
        (coord0, coord1, coord2), x, y, the region annotation (struct_id, 0 outside the atlas)
        and its acronym.
    '''
    return _transforms.analyze3Dalign(labelfile, xv,v,mat, xJ, dx, scale_x, scale_y, x, y, X_, Y_, namesdict, device=device)

def plot_brain_regions(df):
    ''' Plot all brain regions in target brain slice with different color.
            
        Parameters
        ----------
        df : DataFrame
            Dataframe containing each cell in original target brain slice, region annotations and 3D coordinates in terms of the altas.
        
        Returns
        -------
        fig : matplotlib figure
        ax : matplotlib axes
        
    '''
    #plot brain regions
    brain_regions = np.unique(df['acronym'])
    n = len(brain_regions)
    # one color per region: qualitative colormaps for few regions, otherwise
    # hues spaced by the golden ratio with alternating saturation and value
    if n <= 10:
        colors = plt.cm.tab10.colors[:n]
    elif n <= 20:
        colors = plt.cm.tab20.colors[:n]
    else:
        import colorsys
        colors = [colorsys.hsv_to_rgb((k*0.618033988749895) % 1.0, 0.65 + 0.35*(k % 2), 0.95 - 0.35*((k//2) % 2))
                  for k in range(n)]
    fig,ax = plt.subplots()
    for region,color in zip(brain_regions,colors):
        region_df = df[df['acronym']==region]
        ax.scatter(region_df['x'], region_df['y'], label = region, s= 0.1, color=color)
    if n:
        ax.legend(markerscale=20, loc='upper left', bbox_to_anchor=(1.0, 1.0), ncol=max(1, int(np.ceil(n/40))),
                  fontsize='x-small')
    return fig,ax

def plot_subset_brain_regions(df, brain_regions):
    ''' Plot subset of brain regions in target brain slice with different color.
            
        Parameters
        ----------
        df : DataFrame
            Dataframe containing each cell in original target brain slice, region annotations and 3D coordinates in terms of the altas.
        brain_regions : array
            Subset of brain regions of interest (i.e. ['CA1', 'CP', 'DG-sg'])
        
        Returns
        -------
        fig : matplotlib figure
        ax : matplotlib axes
        
    '''
    #plot brain regions
    fig,ax = plt.subplots()
    ax.scatter(df['x'], df['y'],color = 'grey',s= 0.1)   
    for region in brain_regions:
        region_df = df[df['acronym']==region]
        ax.scatter(region_df['x'], region_df['y'], label = region,s= 0.1)   
    if len(brain_regions):
        ax.legend(markerscale=20)
    return fig,ax
