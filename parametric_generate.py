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
        from matplotlib import rc
        rc('font',**{'family':'serif','serif':['Times'],'size':14})
        rc('text', usetex=True)
        plt.figure(figsize=(8,8))

    out=Likefn(z=0.1,name='simulated') # simulated data has this redshift
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
            plt.plot(x*scale,y*scale,label='Jet' if side==0 else 'Counterjet',color='blue' if side==0 else 'green')
            plt.scatter(xc*scale,yc*scale,color='red',label='Sample points' if side==0 else None)
            plt.errorbar(noisyxc*scale, noisyyc*scale, xerr=xearray*scale, yerr=yearray*scale,fmt='+',color='orange',label='Points with added noise' if side==0 else None)


    # put in a standard prior range
    out.set_range([0,0,0,-1.0,0.1,0],[np.pi/2,np.pi/4,2*np.pi,1.0,0.9999,2*np.pi])

    if outfile is not None:
        out.save(outfile)
            
    if plot:
        plt.legend(loc=0)
        plt.axis('equal')
        plt.xlabel('Offset (arcsec)')
        plt.ylabel('Offset (arcsec)')
        plt.savefig('new_simulated_jet.pdf')
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
        
    truth = [1.2,0.25,np.pi/2.0,-0.5,0.60,3*np.pi/4]
    generate(truth=truth, points=15, noise=2.0, plot=True, sides=sides, outfile=name)
