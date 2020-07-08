#!/usr/bin/python

# Process the output of the jet images directly and make an input file for MCMC.

# Initial part based on lobedetect_loop_test.py

from __future__ import print_function
import sys
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from simulation_likefn import Likefn
from simulation_mcmc import run_mcmc,analyse_mcmc

simulation=sys.argv[1]
timestep=int(sys.argv[2])
view=int(sys.argv[3])

name="%s-%i-%i" % (simulation,timestep,view)
lkf=Likefn(name=name)

sourcedir='/beegfs/car/mayaahorton/PLUTO/problems/precessing/parameter_study/cubes/'+simulation+'/'

data=np.loadtxt(sourcedir+'lobelen-profile.txt')

# Do the rotation for this timestep
flt=data[:,0]==timestep
flt&=data[:,1]==view
slce=data[flt]
angle=np.mean(slce[:,6])
print('Rotation angle is', angle)

jet_raw=np.load(sourcedir+'mach-%i-%i.npy' % (timestep,view))

jet=ndimage.rotate(jet_raw,-angle)
imsize,_=jet.shape
halfsize=imsize/2

#detect edges of jet and lobe
for side in [0,1]:
    jyv = []
    jminv = []
    jmaxv = []
    # 0 is north side, 1 is south side
    if side==0:
        yrange=range(halfsize,imsize)
    else:
        yrange=range(halfsize-1,0,-1)
    for y in yrange:
        jetslice=jet[y,:]
        jpixels=np.where(jetslice>1e-1)[0]
        if jpixels.size>0:
            jyv.append(y)
            jminv.append(min(jpixels))
            jmaxv.append(max(jpixels))

    if len(jyv)==0:
        continue
    #convert lists to arrays and get average jet location
    jyv=np.array(jyv)
    jyvs=jyv-halfsize
    jminv=np.array(jminv)
    jmaxv=np.array(jmaxv)

    centre_jet = ((jminv+jmaxv)/2.0)-halfsize

    #print(side,jyvs,centre_jet)
    #errors=(jmaxv-jminv)/2.0
    lkf.add_side(jyvs,centre_jet,np.ones_like(jyvs),2*np.ones_like(jyvs))

prior_min=[0,0,0,-1.0,20,np.pi]
prior_max=[np.pi/2,np.pi/4,2*np.pi,1.0,200,2*np.pi]

lkf.set_range(prior_min,prior_max)

lkf.save(name+'.pickle')

chain=run_mcmc(lkf,outname=name+'.npy')
analyse_mcmc(lkf,chain,do_plot=True,do_print=True,do_show=False)
