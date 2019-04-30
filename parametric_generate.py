# test parametric 2D line fitting
# 1: generate some 'seed' data points

from collections import namedtuple

Data = namedtuple('Data', ['x', 'y', 'xerr', 'yerr', 'true_params'])

if __name__ == '__main__':
    
    import numpy as np
    import matplotlib.pyplot as plt

    from jet_fn import f

    scale=3600.0 # degrees to arcsec

    size = 100   # how many points will we generate along the line
    points = 30  # how many will be selected at random
    noise = 1.0  # noise in arcsec
    noise /= scale

    t=np.linspace(0,1,size)

    # parameters for the simulation:
    # inclination angle, cone angle, phase, age, beta, position angle
    
    truth = [0.8,0.1,np.pi,-0.5,0.60,3*np.pi/4]
    x,y=f(t, truth)

    plt.plot(x*scale,y*scale)

    choice=np.random.choice(size,points,replace=False)
    xc=x[choice]
    yc=y[choice]

    xerr = np.random.normal(0, noise, size=points)
    yerr = np.random.normal(0, noise, size=points)

    noisyxc = xc + xerr
    noisyyc = yc + yerr

    xearray=yearray=np.array([noise]*points)

    plt.scatter(xc*scale,yc*scale,color='red')
    plt.errorbar(noisyxc*scale, noisyyc*scale, xerr=xearray*scale, yerr=yearray*scale, fmt='go')

    # use any npy array to save the data
    np.save('data.npy',[noisyxc,noisyyc, xearray, yearray, truth])

    plt.show()
