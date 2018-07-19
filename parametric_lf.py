# compute 'LF'

import numpy as np
from scipy.misc import logsumexp
from jet_fn import f

def lf(xd, yd, xderr, yderr, parms, size=1000):
    
    t=np.linspace(0,1,size)

    x,y=f(t,parms)

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
        determinant = (xderr[i]**2) * (yderr[i]**2)
        prefix = 2 * np.pi * np.sqrt(determinant)
        exp = ((xc-xd[i]) / xderr[i])**2.
        exp += ((yc-yd[i]) / yderr[i])**2.
        ll += logsumexp(-exp / 2., b=dl/prefix) - logsumdl  # compute logp in log space
        # p=np.sum(np.exp(-((xc-xd[i])**2.0+(yc-yd[i])**2.0)/sigma[i])*dl)/np.sum(dl)
        # ll+=np.log(p)
    #print ll
    return ll

if __name__=='__main__':
    # example varying one parameter
    import matplotlib.pyplot as plt
    from parametric_generate import Data
    
    data = Data(*np.load('data.npy'))

    parms=data.true_params
    # print(data.yerr)
    # print(data.xerr)
    # sigma = np.sqrt(data.xerr**2 + data.yerr**2)
    # print(lf(data.x, data.y, np.ones_like(data.x)*0.5, np.ones_like(data.y)*0.5, np.ones_like(data.x)*0.5, parms))  #

    parm=np.linspace(0,3,100)
    ll=np.zeros_like(parm)
    for i in range(len(parm)):
        parms[0]=parm[i]
        ll[i]=lf(data.x, data.y, data.xerr, data.yerr, parms)
    plt.plot(parm,ll)
    plt.show()

