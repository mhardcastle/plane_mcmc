# compute 'LF'

from __future__ import print_function
import numpy as np
from scipy.misc import logsumexp
from jet_fn import f,cf

def findmaxr(xd,yd,xderr,yderr):
    r=np.max(xd**2.0+yd**2.0)
    r+=3*np.max(xderr**2.0+yderr**2.0)
    return r

def findt(fparms,maxr,side,size=500):

    jetf=[f,cf][side]
    scale=1
    tmin=0
    found=False
    while not found:
        scale*=2
        t=np.linspace(tmin,scale,size)
        x,y=jetf(t,fparms)
        r=x**2.0+y**2.0
        index=np.argmin(r<maxr)
        if index==0:
            tmin=scale
            continue
        return t[index]


def lf(xd, yd, xderr, yderr, parms, side, size=1000,maxr=None):

    jetf=[f,cf][side]
    if maxr is None:
        maxr=findmaxr(xd,yd,xderr,yderr)
        
    tmax=findt(parms[:-1],maxr,side)
    t=np.linspace(0,tmax,size)
    #print parms[0],tmax
    x,y=jetf(t,parms[:-1])
    variance = parms[-1]

    # find the line integral element
    dx=x[0:-1]-x[1:]
    dy=y[0:-1]-y[1:]
    dl=np.sqrt(dx**2+dy**2)
    # find the midpoint -- lin interp to save re-evaluating the fn
    xc=(x[0:-1]+x[1:])*0.5
    yc=(y[0:-1]+y[1:])*0.5

    ll=0

    logsumdl = np.log(np.sum(dl))

    for i in range(len(xd)):
        scaledx = (xderr[i]**2 + variance)
        scaledy = (yderr[i]**2 + variance)
        prefix = 2 * np.pi * np.sqrt(scaledx) * np.sqrt(scaledy)
        exp = (xc-xd[i])**2 / scaledx
        exp += (yc-yd[i])**2 / scaledy
        ll += logsumexp(-exp / 2., b=dl/prefix) - logsumdl  # compute logp in log space
        # p=np.sum(np.exp(-((xc-xd[i])**2.0)/(2.0*scaledx)
        #                -((yc-yd[i])**2.0)/(2.0*scaledy))
        #         *dl/prefix)/np.sum(dl)
        #ll+=np.log(p)
    #print ll
    return ll

if __name__=='__main__':
    # example varying one parameter
    import matplotlib.pyplot as plt
    from parametric_generate import Data
    
    data = Data(*np.load('data.npy'))

    parms=data.true_params
    parms=np.append(parms,(1/3600.0)**2.0)
    print('Parameters are',parms)
                  
    # print(data.yerr)
    # print(data.xerr)
    # sigma = np.sqrt(data.xerr**2 + data.yerr**2)
    # print(lf(data.x, data.y, np.ones_like(data.x)*0.5, np.ones_like(data.y)*0.5, np.ones_like(data.x)*0.5, parms))  #

    parm=np.linspace(0,np.pi/2.0,100)
    ll=np.zeros_like(parm)
    for i in range(len(parm)):
        parms[0]=parm[i]
        ll[i]=lf(data.x, data.y, data.xerr, data.yerr, parms, 0)
        print(i,parm[i],ll[i])
    plt.plot(parm,ll)
    plt.show()

