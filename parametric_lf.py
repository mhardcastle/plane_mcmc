# Example LF calculation
# This is just a stub test code at present

from __future__ import print_function
import numpy as np
from likefn import Likefn
import sys

if __name__=='__main__':

    if len(sys.argv)==1:
        name='data.pickle'
    else:
        name=sys.argv[1]
    # example varying one parameter
    import matplotlib.pyplot as plt
    from parametric_generate import Data
    
    lkf=Likefn.load(name)

    if lkf.truth is None:
        print("Can't do this without truth!")
        sys.exit(2)
    
    parms=lkf.truth
    parms=np.append(parms,(1/3600.0)**2.0)
    print('Parameters are',parms)
                  
    # print(data.yerr)
    # print(data.xerr)
    # sigma = np.sqrt(data.xerr**2 + data.yerr**2)
    # print(lf(data.x, data.y, np.ones_like(data.x)*0.5, np.ones_like(data.y)*0.5, np.ones_like(data.x)*0.5, parms))  #

    print('Loaded data has',lkf.sides,'sides')
    parm=np.linspace(0,np.pi/2.0,100)
    ll=np.zeros_like(parm)
    for i in range(len(parm)):
        parms[0]=parm[i]
        for side in range(lkf.sides):
            ll[i]+=lkf.lf(parms, side)
        print(i,parm[i],ll[i])
    plt.plot(parm,ll)
    plt.show()

