# test parametric 2D line fitting
# 1: generate some 'seed' data points

from collections import namedtuple
import numpy as np
from jet_fn import f,cf

Data = namedtuple('Data', ['x', 'y', 'xerr', 'yerr', 'true_params'])

def generate(size=100,points=30,noise=1.0,age=1.0,truth=None,outfile='data.npy',plot=False):
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
        
    for side in range(2):

        jf=[f,cf][side]
        outname=[outfile,'cj_'+outfile][side]
        
        # parameters for the simulation:
        # inclination angle, cone angle, phase, age, beta, position angle

        x,y=jf(t, truth)

        choice=np.random.choice(size,points,replace=False)
        xc=x[choice]
        yc=y[choice]

        xerr = np.random.normal(0, noise, size=points)
        yerr = np.random.normal(0, noise, size=points)

        noisyxc = xc + xerr
        noisyyc = yc + yerr

        xearray=yearray=np.array([noise]*points)

        # use a npy array to save the data
        np.save(outname,[noisyxc,noisyyc, xearray, yearray, truth])

        if plot:
            plt.plot(x*scale,y*scale)
            plt.scatter(xc*scale,yc*scale,color='red')
            plt.errorbar(noisyxc*scale, noisyyc*scale, xerr=xearray*scale, yerr=yearray*scale, fmt='go')

    if plot:
        plt.show()

if __name__ == '__main__':
    
    truth = [0.8,0.1,np.pi,-0.5,0.60,3*np.pi/4]
    generate(truth=truth, plot=True)
