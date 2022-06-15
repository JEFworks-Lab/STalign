# try daniel's Merfish to merfish registration with tools

#library(reticulate)
#use_condaenv("pytorch_m1")
#reticulate::repl_python()

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd # for csv.
from matplotlib import cm
from matplotlib.lines import Line2D
import os
import glob
import torch

from scipy.stats import rankdata
import nrrd
import time

import tools

files = glob.glob('/Users/jeanfan/OneDrive - Johns Hopkins/Data_Private/MERFISH_cortex_For_Daniel/raw/*metadata*.csv.gz')
files.sort()
files

## image 1
fname = files[-4]

names = os.path.splitext(fname)
if names[1] == '.gz':
    names = os.path.splitext(names[0])
outname = names[0] + '_with_structure_id_v02.csv'

df = pd.read_csv(fname.replace('by_gene','metadata'))
xI = np.array(df['center_x'])
yI = np.array(df['center_y'])
vI = np.array(df['volume'])

import imp
imp.reload(tools)
blur = [2.0,1.0,0.5]

draw = 10000
wavelet_magnitude = True
dx = 30.0
use_windowing = True
XI,YI,I,fig = tools.rasterize(xI,yI,dx=dx,blur=blur,draw=draw, wavelet_magnitude=wavelet_magnitude, use_windowing=use_windowing)

plt.show()

## image 2
fname = files[-5]

names = os.path.splitext(fname)
if names[1] == '.gz':
    names = os.path.splitext(names[0])
outname = names[0] + '_with_structure_id_v02.csv'

df = pd.read_csv(fname.replace('by_gene','metadata'))
xJ = np.array(df['center_x'])
yJ = np.array(df['center_y'])
vJ = np.array(df['volume'])

XJ,YJ,J,fig = tools.rasterize(xJ,yJ,dx=dx,blur=blur,draw=draw, wavelet_magnitude=wavelet_magnitude, use_windowing=use_windowing)

plt.show()

## run alignment
imp.reload(tools)
theta = 45*np.pi/180
## change device to cpu if not on deep learnign comp
out = tools.LDDMM([YI,XI],I,[YJ,XJ],J,
            L=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]]),
            T=np.array([5000,-1000.0]),
             device='cpu')
             
plt.show()             

## now need to figure out how to transfer diffeomorphism onto original points
out

type(out)
out[0]
out[1]

## affine and velocity field
A = out[0]
A.shape
v = out[1]
v.shape

## pull out from tools.py parameters
dtype = torch.float64
device = 'cpu'
a=500.0
p=2.0
expand=2.0
nt=3
dx = 30.0

minv = torch.as_tensor([x[0] for x in [YI,XI]],device=device,dtype=dtype)
maxv = torch.as_tensor([x[-1] for x in [YI,XI]],device=device,dtype=dtype)
minv,maxv = (minv+maxv)*0.5 + 0.5*torch.tensor([-1.0,1.0],device=device,dtype=dtype)[...,None]*(maxv-minv)*expand
xv = [torch.arange(m,M,a*0.5,device=device,dtype=dtype) for m,M in zip(minv,maxv)]

with torch.no_grad():
    # points are in row column
    #pointsM = np.stack((yI/dx-np.min(yI)/dx,xI/dx-np.min(xI)/dx),-1)
    pointsM = np.stack((yI,xI),-1)
    pointsM = torch.tensor(pointsM)
    pointsMt = torch.clone(pointsM)
    for t in range(nt):
        pointsMt += tools.interp(xv,v[t].permute(2,0,1),pointsMt.T[...,None])[...,0].T/nt
    pointsMt = (A[:2,:2]@pointsMt.T + A[:2,-1][...,None]).T
    
## show original points
f,ax = plt.subplots()
ax.scatter(pointsM[:,1],pointsM[:,0],s=0.1,ec='none')
ax.set_aspect('equal')
plt.show()

## show out results
f,ax = plt.subplots()
ax.scatter(pointsMt[:,1],pointsMt[:,0],s=0.1,ec='none')
ax.set_aspect('equal')
plt.show()

## show target points
pointsT = np.stack((yJ,xJ),-1)
f,ax = plt.subplots()
ax.scatter(pointsT[:,1],pointsT[:,0],s=0.1,ec='none')
ax.set_aspect('equal')
plt.show()


## output
fname = files[-4]
names = os.path.splitext(fname)
if names[1] == '.gz':
    names = os.path.splitext(names[0])
outname = names[0] + '_with_structure_id_v02.csv'
df = pd.read_csv(fname.replace('by_gene','metadata'))

dfout = pd.DataFrame(df,copy=True)
dfout['row_aligned'] = pointsMt[:,0]
dfout['col_aligned'] = pointsMt[:,1]

dfout

dfout.to_csv('jean_test_merfish_merfish_06132022')


#### recreate some figures
with torch.no_grad():
    # points are in row column
    #pointsM = np.stack((yI/dx-np.min(yI)/dx,xI/dx-np.min(xI)/dx),-1)
    pointsM = np.stack((yI,xI),-1)
    pointsM = torch.tensor(pointsM)
    pointsMt = torch.clone(pointsM)
    for t in range(nt):
        pointsMt += tools.interp(xv,v[t].permute(2,0,1),pointsMt.T[...,None])[...,0].T/nt
    pointsMt = (A[:2,:2]@pointsMt.T + A[:2,-1][...,None]).T
    
## show out results
f,ax = plt.subplots()
ax.scatter(pointsMt[:,1],pointsMt[:,0],s=0.1,ec='none')
ax.set_aspect('equal')
plt.show()

aI,aI,tI,fig = tools.rasterize(pointsMt[:,1].numpy(),pointsMt[:,0].numpy(),dx=dx,blur=blur,draw=draw, wavelet_magnitude=wavelet_magnitude, use_windowing=use_windowing)
plt.show()


## affine only
with torch.no_grad():
    # points are in row column
    #pointsM = np.stack((yI/dx-np.min(yI)/dx,xI/dx-np.min(xI)/dx),-1)
    pointsM = np.stack((yI,xI),-1)
    pointsM = torch.tensor(pointsM)
    pointsMt = torch.clone(pointsM)
    #for t in range(nt):
    #    pointsMt += tools.interp(xv,v[t].permute(2,0,1),pointsMt.T[...,None])[...,0].T/nt
    pointsMt = (A[:2,:2]@pointsMt.T + A[:2,-1][...,None]).T
    
## show out results
f,ax = plt.subplots()
ax.scatter(pointsMt[:,1],pointsMt[:,0],s=0.1,ec='none')
ax.set_aspect('equal')
plt.show()

aI,aI,tI,fig = tools.rasterize(pointsMt[:,1].numpy(),pointsMt[:,0].numpy(),dx=dx,blur=blur,draw=draw, wavelet_magnitude=wavelet_magnitude, use_windowing=use_windowing)
plt.show()
