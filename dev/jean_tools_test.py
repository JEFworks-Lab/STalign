## test usng daniel's tools.py
## try to translate jean_test_merfish_merfish.py into using functions

#library(reticulate)
#use_condaenv("pytorch_m1")
#reticulate::repl_python()
## now you can use python commands

import tools 

files = glob.glob('/Users/jeanfan/OneDrive - Johns Hopkins/Data_Private/MERFISH_cortex_For_Daniel/raw/*metadata*.csv.gz')
files.sort()
files

## first merfish image
fname = files[-4]
df = pd.read_csv(fname.replace('by_gene','metadata'))
x = np.array(df['center_x'])
y = np.array(df['center_y'])
X1,Y1,image1,fig1 = tools.rasterize(x,y,dx=50,draw=10000)

## second merfish image
fname = files[-5]
df = pd.read_csv(fname.replace('by_gene','metadata'))
x = np.array(df['center_x'])
y = np.array(df['center_y'])
X2,Y2,image2,fig2 = tools.rasterize(x,y,dx=50,draw=10000)

## LDDMM (not working)
import importlib
importlib.reload(tools)
align, v = tools.LDDMM(image1=image1, image2=image2, diffeo_start = 300, epL = 5e-8, epT = 5e-4, epV = 5e1, niter = 1000, a = 5.0, p = 3.0,
expand = 2.0, sigmaM = 0.2, sigmaB = 0.1, sigmaA = 0.3, sigmaR = 200, sigmaP = 0.2, device = 'cpu', 
pointsI = np.array([[75,70]]),
pointsJ = np.array([[44,91]]))

## now apply
with torch.no_grad():
    # points are in row column
    pointsM = np.stack((y/dx-np.min(y)/dx,x/dx-np.min(x)/dx),-1)
    pointsM = torch.tensor(pointsM)
    pointsMt = torch.clone(pointsM)
    ## jean I believe this is applying the learned transform
    for t in range(nt):
        pointsMt += interp(xv,v[t].permute(2,0,1),pointsMt.T[...,None])[...,0].T/nt
    pointsMt = (A[:2,:2]@pointsMt.T + A[:2,-1][...,None]).T

f,ax = plt.subplots()
ax.scatter(pointsM[:,1],pointsM[:,0],s=0.1,ec='none')
ax.set_aspect('equal')
plt.show()
