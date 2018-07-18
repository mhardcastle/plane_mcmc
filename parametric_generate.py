# test parametric 2D line fitting
# 1: generate some 'seed' data points

from collections import namedtuple

Data = namedtuple('Data', ['x', 'y', 'xerr', 'yerr', 'true_params'])

if __name__ == '__main__':
    
    import numpy as np
    import matplotlib.pyplot as plt

    from jet_fn import f

    size=100
    points=30
    noise = 1.0/3600.0
    scale=3600.0 # degrees to arcsec

    t=np.linspace(0,1,size)

    truth = [1.2,0.2,np.pi,0.0,0.80,np.pi]
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

    np.save('data.npy',[noisyxc,noisyyc, xearray, yearray, truth])

    plt.show()
