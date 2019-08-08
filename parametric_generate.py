# test parametric 2D line fitting
# generate some data points and save them in a data structure

from collections import namedtuple
import numpy as np
from likefn import Likefn
import sys

def generate(size=100,points=30,noise=1.0,age=1.0,truth=None,sides=1,outfile=None,plot=False):
    '''
    size: how many points will we generate along the line 
    points: how many will be selected at random 
    noise: simulated noise in arcsec 
    age: age of simulated jet in Myr 
    truth: the true parameters to use:
       inclination angle, cone angle, phase, age, beta, position angle
    outfile: file to write to
    '''
    
    scale=3600.0 # degrees to arcsec
    noise /= scale

    t=np.linspace(0,age,size)

    if plot:
        import matplotlib.pyplot as plt

    out=Likefn(z=0.1) # simulated data has this redshuft
    out.truth=truth
    
    for side in range(sides):

        # parameters for the simulation:
        # inclination angle, cone angle, phase, age, beta, position angle

        x,y=out.jetfn(side, t, truth)

        choice=np.random.choice(size,points,replace=False)
        xc=x[choice]
        yc=y[choice]

        xerr = np.random.normal(0, noise, size=points)
        yerr = np.random.normal(0, noise, size=points)

        noisyxc = xc + xerr
        noisyyc = yc + yerr

        xearray=yearray=np.array([noise]*points)

        out.add_side(noisyxc,noisyyc, xearray, yearray)
        
        if plot:
            plt.plot(x*scale,y*scale)
            plt.scatter(xc*scale,yc*scale,color='red')
            plt.errorbar(noisyxc*scale, noisyyc*scale, xerr=xearray*scale, yerr=yearray*scale, fmt='go')

    # put in a standard prior range
    out.set_range([0,0,0,-1.0,0.1,0],[np.pi/2,np.pi/4,2*np.pi,1.0,0.9999,2*np.pi])

    if outfile is not None:
        out.save(outfile)
            
    if plot:
        plt.show()

    return out

if __name__ == '__main__':

    try:
        name=sys.argv[1]
    except IndexError:
        name='data.pickle'
    try:
        sides=int(sys.argv[2])
    except IndexError:
        sides=1
        
    truth = [1.2,0.2,np.pi/2.0,-0.2,0.60,3*np.pi/4]
    generate(truth=truth, plot=True, sides=sides, outfile=name)
