''' Tools for registration of spatial transcriptomics data.
'''

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.nn.functional import grid_sample



def rasterize(x, y, dx=30.0, blur=1.0, expand=1.1, draw=0, wavelet_magnitude=False):
    ''' Rasterize a spatial transcriptomics dataset into a density image
    
    Paramters
    ---------
    x : numpy array of length N
        x location of cells
    y : numpy array of length N
        y location of cells
    dx : float
        Pixel size to rasterize data (default 30.0, in same units as x and y)
    blur : float or list of floats
        Standard deviation of Gaussian interpolation kernel.  Units are in 
        number of pixels.  Can be aUse a list to do multi scale.
    expand : float
        Factor to expand sampled area beyond cells. Defaults to 1.1.
    draw : int
        If True, draw a figure every draw points return its handle. Defaults to False (0).
    wavelet_magnitude : bool
        If True, take the absolute value of difference between scales for raster images.
        When using this option blur should be sorted from greatest to least.
    
        
    Returns
    -------
    X  : numpy array
        Locations of pixels along the x axis
    Y  : numpy array
        Locations of pixels along the y axis
    M : numpy array
        A rasterized image with len(blur) channels along the last axis
    fig : matplotlib figure handle
        If draw=True, returns a figure handle to the drawn figure.
        
    Raises
    ------    
    Exception 
        If wavelet_magnitude is set to true but blur is not sorted from greatest to least.
        
        
    
    Examples
    --------
    Rasterize a dataset at 30 micron pixel size, with three kernels.
    
    >>> X,Y,M,fig = tools.rasterize(x,y,dx=30.0,blur=[2.0,1.0,0.5],draw=10000)
    
    Rasterize a dataset at 30 micron pixel size, with three kernels, using difference between scales.
    
    >>> X,Y,M,fig = tools.rasterize(x,y,dx=30.0,blur=[2.0,1.0,0.5],draw=10000, wavelet_magnitude=True)
        
    '''
    
    # set blur to a list
    if not isinstance(blur,list):
        blur = [blur]
    nb = len(blur)
    blur = np.array(blur)
    n = len(x)
    
    if wavelet_magnitude and np.any(blur != np.sort(blur)[::-1]):
        raise Exception('When using wavelet magnitude, blurs must be sorted from greatest to least')
    
    minx = np.min(x)
    maxx = np.max(x)
    miny = np.min(y)
    maxy = np.max(y)
    minx,maxx = (minx+maxx)/2.0 - (maxx-minx)/2.0*expand, (minx+maxx)/2.0 + (maxx-minx)/2.0*expand
    miny,maxy = (miny+maxy)/2.0 - (maxy-miny)/2.0*expand, (miny+maxy)/2.0 + (maxy-miny)/2.0*expand
    X_ = np.arange(minx,maxx,dx)
    Y_ = np.arange(miny,maxy,dx)
    
    X = np.stack(np.meshgrid(X_,Y_))

    W = np.zeros((X.shape[1],X.shape[2],nb))

    
    if draw: fig,ax = plt.subplots()
    count = 0
    
    for x_,y_ in zip(x,y):
        # to speed things up I shoul index
        # to do this I'd have to find row and column indices
        #col = np.round((x_ - X_[0])/dx).astype(int)
        #row = np.round((y_ - X_[1])/dx).astype(int)
        #row0 = np.floor(row-blur*3).astype(int)
        #row1 = np.ceil(row+blur*3).astype(int)        
        #rows = np.arange(row0,row1+1)
        

        # this is incrementing one pixel at a time, it is way way faster, 
        # but doesn't use a kernel
        # I[c_,row,col] += 1.0
        # W[row,col] += 1.0

        k = np.exp( - ( (X[0][...,None] - x_)**2 + (X[1][...,None] - y_)**2 )/(2.0*(dx*blur*2)**2)  )
        k /= np.sum(k,axis=(0,1),keepdims=True)     
        if wavelet_magnitude:
            for i in reversed(range(nb)):
                if i == 0:
                    continue
                k[...,i] = k[...,i] - k[...,i-1]
        
        W += k
        
            
        

        if draw:
            if not count%draw or count==(x.shape[0]-1):
                print(f'{count} of {x.shape[0]}')

                ax.cla()
                toshow = W-np.min(W,axis=(0,1),keepdims=True)
                toshow = toshow / np.max(toshow,axis=(0,1),keepdims=True)
                
                if nb >= 3:
                    toshow = toshow[...,:3]
                elif nb == 2:
                    toshow = toshow[...,[0,1,0]]
                elif nb == 1:
                    toshow = toshow[...,[0,0,0]]
                
                ax.imshow(np.abs(toshow))
                fig.canvas.draw()

        count += 1
    W = np.abs(W)
    # we will permute so channels are on first axis
    W = W.transpose((-1,0,1))
    extent = (X_[0],X_[-1],Y_[0],Y_[-1])
    
    # rename
    X = X_
    Y = Y_
    if draw:
        output = X,Y,W,fig
    else:
        output = X,Y,W
    return output
    
    
    
def interp(x,I,phii,**kwargs):
    '''
    Interpolate the image I, with regular grid positions stored in x (1d arrays),
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

    # NOTE I should check that I can reproduce identity
    # note that phii must now store x,y,z along last axis
    # is this the right order?
    # I need to put batch (none) along first axis
    # what order do the other 3 need to be in?    
    out = grid_sample(I[None],phii.flip(0).permute((1,2,0))[None],align_corners=True,**kwargs)
    # note align corners true means square voxels with points at their centers
    # post processing, get rid of batch dimension

    return out[0]



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
        5D (nt,3,v0,v1,v2) velocity field
    
    Returns
    -------
    phii: torch tensor
        Inverse map (position field) computed by method of characteristics

    '''
    
    XV = torch.stack(torch.meshgrid(xv))
    phii = torch.clone(XV)
    dt = 1.0/v.shape[0]
    for t in range(v.shape[0]):
        Xs = XV - v[t]*dt
        phii = interp(xv,phii-XV,Xs)+Xs
    return phii


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

def extent_from_x(xJ):
    ''' Given a set of pixel locations, returns an extent 4-tuple for use with np.imshow.
    
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


def LDDMM(xI,I,xJ,J,pointsI=None,pointsJ=None,
          a=500.0,p=3.0,expand=2.0,nt=3,
         niter=5000,diffeo_start=100, epL=5e-11, epT=5e-4, epV=5e1,
         sigmaM=0.2,sigmaB=0.19,sigmaA=0.3,sigmaR=5e5,sigmaP=2e-1):
    ''' Run LDDMM between a pair of images.
    
    This jointly estimates an affine transform A, and a diffeomorphism phi.
    The map is off the form x -> A phi x
    
    
    Parameters
    ----------
    xI : list of torch tensor
        Location of voxels in atlas image I
    I : torch tensor
        Atlas image I, with channels along first axis        
    xJ : list of torch tensor
        Location of voxels in target image J
    J : torch tensor
        Target image J, with channels along first axis
    a : float
        Smoothness scale of velocity field (default 200.0)
    p : float
        Power of Laplacian in velocity regularization (default 2.0)
    expand : float
        Factor to expand size of velocity field around image boundaries (default 2.0)
    nt : int
        Number of timesteps for integrating velocity field (defulat 3)
    pointsI : torch tensor
        N x 2 set of corresponding points for matching in atlas image. Default None (no points).
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
        Standard deviation of matching term for Gaussian mixture modeling in cost function. 
        This term generally controls matching accuracy with smaller corresponding to more accurate.
    sigmaB : float
        Standard deviation of backtround term for Gaussian mixture modeling in cost function. 
        If there is missing tissue in target, we may label some pixels in target as background,
        and not enforce matching here.
    sigmaA : float
        Standard deviation of artifact term for Gaussian mixture modeling in cost function. 
        If there are artifacts in target or other lack of corresponding between template and target, 
        we may label some pixels in target as artifact, and not enforce matching here.
        
    
    Returns
    -------
    A : torch tensor
        Affine transform
    v : torch tensor
        Velocity field
    
    '''
    
    
    
    
    # todo
    # implement local?
    # more iters

    #niter = 2000
    #diffeo_start = 100
    #epL = 5e-11
    #epT = 5e-4
    #epV = 5e1
    #niter = 5000

    L = torch.eye(2,device=device,dtype=dtype,requires_grad=True)
    #L.data[0,0] = -1.0
    #L.data[:2,:2] *= 0.2
    T = torch.zeros(2,device=device,dtype=dtype,requires_grad=True)
    #T.data[0] = +2000.0
    #T.data[1] = -xI[1][-1]/2.0*1.1

    # velocity
    #a = 500.0
    #p = 3.0
    #expand = 2.0
    minv = torch.as_tensor([x[0] for x in xI],device=device,dtype=dtype)
    maxv = torch.as_tensor([x[-1] for x in xI],device=device,dtype=dtype)
    minv,maxv = (minv+maxv)*0.5 + 0.5*torch.tensor([-1.0,1.0],device=device,dtype=dtype)[...,None]*(maxv-minv)*expand
    xv = [torch.arange(m,M,a*0.5,device=device,dtype=dtype) for m,M in zip(minv,maxv)]
    extentV = extent_from_x(xv)
    dv = torch.as_tensor([x[1]-x[0] for x in xv],device=device,dtype=dtype)
    XV = torch.stack(torch.meshgrid(xv),-1)
    fv = [torch.arange(n,device=device,dtype=dtype)/n/d for n,d in zip(XV.shape,dv)]
    extentF = extent_from_x(fv)
    FV = torch.stack(torch.meshgrid(fv),-1)
    LL = (1.0 + 2.0*a**2* torch.sum( (1.0 - torch.cos(2.0*np.pi*FV*dv))/dv**2 ,-1))**(p*2.0)

    K = 1.0/LL
    fig,ax = plt.subplots()
    ax.imshow(K,vmin=0.0,vmax=0.1,extent=extentF)
    #fig,ax = plt.subplots()
    #ax.imshow(K[0].cpu())
    DV = torch.prod(dv)
    Ki = torch.fft.ifftn(K).real
    fig,ax = plt.subplots()
    ax.imshow(Ki,vmin=0.0,extent=extentV)


    # nt = 3
    v = torch.zeros((nt,XV.shape[0],XV.shape[1],XV.shape[2]),device=device,dtype=dtype,requires_grad=True)


    WM = torch.ones(Jd[0].shape,dtype=J.dtype,device=J.device)*0.5
    WB = torch.ones(Jd[0].shape,dtype=J.dtype,device=J.device)*0.4
    WA = torch.ones(Jd[0].shape,dtype=J.dtype,device=J.device)*0.1
    pointsI = torch.tensor(pointsI,device=J.device,dtype=J.dtype)
    pointsJ = torch.tensor(pointsJ,device=J.device,dtype=J.dtype)
    
    #sigmaM = 0.2
    #sigmaB = 0.19
    #sigmaA = 0.3
    #sigmaR = 5e5
    #sigmaP = 2e-1
    
    fig,ax = plt.subplots(2,3)
    ax = ax.ravel()
    figE,axE = plt.subplots()
    Esave = []

    try:
        L.grad.zero_()
    except:
        pass
    try:
        T.grad.zero_()
    except:
        pass
    for it in range(niter):
        # make A
        A = to_A(L,T)
        # Ai
        Ai = torch.linalg.inv(A)
        # transform sample points
        Xs = (Ai[:2,:2]@XJd[...,None])[...,0] + Ai[:2,-1]    
        # now diffeo, not semilagrange here
        for t in range(nt-1,-1,-1):
            Xs = Xs + interp(xv,-v[t].permute(2,0,1),Xs.permute(2,0,1)).permute(1,2,0)/nt
        # and points
        pointsIt = torch.clone(pointsI)
        for t in range(nt):
            pointsIt += interp(xv,v[t].permute(2,0,1),pointsIt.T[...,None])[...,0].T/nt
        pointsIt = (A[:2,:2]@pointsIt.T + A[:2,-1][...,None]).T

        # transform image
        AI = interp(xI,I,Xs.permute(2,0,1))

        # transform the contrast
        B = torch.ones(1+AI.shape[0],AI.shape[1]*AI.shape[2],device=AI.device,dtype=AI.dtype)
        B[1:4] = AI.reshape(AI.shape[0],-1)
        #B = torch.ones(10,AI.shape[1]*AI.shape[2],device=AI.device,dtype=AI.dtype)
        #B[1:4] = AI.reshape(AI.shape[0],-1)
        #B[4] = (AI[0][None]**2).reshape(1,-1)
        #B[5] = (AI[1][None]**2).reshape(1,-1)
        #B[6] = (AI[2][None]**2).reshape(1,-1)
        #B[7] = (AI[0][None]*AI[1][None]).reshape(1,-1)
        #B[8] = (AI[0][None]*AI[2][None]).reshape(1,-1)
        #B[9] = (AI[1][None]*AI[2][None]).reshape(1,-1)
        with torch.no_grad():    
            BB = B@(B*WM.ravel()).T
            BJ = B@((Jd*WM).reshape(Jd.shape[0],Jd.shape[1]*Jd.shape[2])).T
            small = 0.1
            coeffs = torch.linalg.solve(BB + small*torch.eye(BB.shape[0],device=BB.device,dtype=BB.dtype),BJ)
        fAI = ((B.T@coeffs).T).reshape(Jd.shape)

        # objective function
        EM = torch.sum((fAI - Jd)**2*WM)/2.0/sigmaM**2
        ER = torch.sum(torch.sum(torch.abs(torch.fft.fftn(v,dim=(1,2)))**2,dim=(0,-1))*LL)*DV/2.0/v.shape[1]/v.shape[2]/sigmaR**2
        EP = torch.sum((pointsIt - pointsJ)**2)/2.0/sigmaP**2
        E = EM + ER + EP
        Esave.append( (E.item(), EM.item(), ER.item(), EP.item()) )
        # gradient update
        E.backward()
        with torch.no_grad():
            L -= (epL/(1.0 + (it>=diffeo_start)*9))*L.grad
            T -= (epT/(1.0 + (it>=diffeo_start)*9))*T.grad

            L.grad.zero_()
            T.grad.zero_()

            # v grad
            vgrad = v.grad
            # smooth it
            vgrad = torch.fft.ifftn(torch.fft.fftn(vgrad,dim=(1,2))*K[...,None],dim=(1,2)).real
            if it >= diffeo_start:
                v -= vgrad*epV
            v.grad.zero_()


        # update weights
        if not it%5:
            with torch.no_grad():
                # M step for these params
                muA = torch.sum(WA*Jd,dim=(-1,-2))/torch.sum(WA)
                muB = torch.sum(WB*Jd,dim=(-1,-2))/torch.sum(WB)
                if it <= 200:
                    muA = torch.tensor([0.75,0.77,0.79],device=J.device,dtype=J.dtype)
                    muB = torch.ones(Jd.shape[0],device=J.device,dtype=J.dtype)*0.96

                if it >= 50:

                    W = torch.stack((WM,WA,WB))
                    pi = torch.sum(W,dim=(1,2))
                    pi += torch.max(pi)*1e-6
                    pi /= torch.sum(pi)


                    # now the E step, update the weights
                    WM = pi[0]* torch.exp( -torch.sum((fAI - Jd)**2,0)/2.0/sigmaM**2 )/np.sqrt(2.0*np.pi*sigmaM**2)**J.shape[0]
                    WA = pi[1]* torch.exp( -torch.sum((muA[...,None,None] - Jd)**2,0)/2.0/sigmaA**2 )/np.sqrt(2.0*np.pi*sigmaA**2)**J.shape[0]
                    WB = pi[2]* torch.exp( -torch.sum((muB[...,None,None] - Jd)**2,0)/2.0/sigmaB**2 )/np.sqrt(2.0*np.pi*sigmaB**2)**J.shape[0]
                    WS = WM+WB+WA
                    WS += torch.max(WS)*1e-6
                    WM /= WS
                    WB /= WS
                    WA /= WS




        # draw
        if not it%10:
            ax[0].cla()
            ax[0].imshow(((AI-torch.amin(AI,(1,2))[...,None,None])/(torch.amax(AI,(1,2))-torch.amin(AI,(1,2)))[...,None,None]).permute(1,2,0).clone().detach().cpu(),extent=extentJ)
            ax[0].scatter(pointsIt[:,1].clone().detach().cpu(),pointsIt[:,0].clone().detach().cpu())
            ax[1].cla()    
            ax[1].imshow(clip(fAI.permute(1,2,0).clone().detach()).cpu(),extent=extentJ)
            ax[1].scatter(pointsIt[:,1].clone().detach().cpu(),pointsIt[:,0].clone().detach().cpu())

            ax[5].cla()
            ax[5].imshow(clip(fAI - Jd).permute(1,2,0).clone().detach().cpu()*0.5+0.5,extent=extentJ)
            ax[5].scatter(pointsIt[:,1].clone().detach().cpu(),pointsIt[:,0].clone().detach().cpu())
            ax[5].scatter(pointsJ[:,1].clone().detach().cpu(),pointsJ[:,0].clone().detach().cpu())

            ax[2].cla()
            ax[2].imshow(Jd.permute(1,2,0).cpu(),extent=extentJ)
            ax[2].scatter(pointsJ[:,1].clone().detach().cpu(),pointsJ[:,0].clone().detach().cpu())

            ax[4].cla()
            ax[4].imshow(clip(torch.stack((WM,WA,WB),-1).clone().detach()).cpu(),extent=extentJ)



            toshow = v[0].clone().detach().cpu()
            toshow /= torch.max(torch.abs(toshow))
            toshow = toshow*0.5+0.5
            toshow = torch.cat((toshow,torch.zeros_like(toshow[...,0][...,None])),-1)   
            ax[3].cla()
            ax[3].imshow(clip(toshow),extent=extentV)

            axE.cla()
            axE.plot(Esave)
            axE.legend(['E','EM','ER','EP'])
            axE.set_yscale('log')


            fig.canvas.draw()
            figE.canvas.draw()
            
    return A,V
