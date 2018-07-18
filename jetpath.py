import numpy as np

def jetpath(t, i=1.6, psi=0.3, theta=0, pp=1, s_jet=1, beta=0.99, z=1,
            alpha=0, core=None):
    # version 3 also returns a Boolean parameter which is True, if the jet
    # starts out
 
    # plot solution of precessing jet aberration problem
    # this version implements a callable function
    # model of Gower et al. 1982 ApJ 262,478
    # Reproduced their Figure 2 (compare V01).
 
    # this version: origin = 0
    # output: list of jet (counterjet) position vectors in degrees
 
    # inputs:
    # dt: np array of times in Myr
    # i: inclination cone angle to line of sight / rad
    # psi: prec. cone 1/2-opening angle / rad
    # theta: phase of precession
    # pp: precession period / myr
    # s_jet: 1=jet (rightwards, x) -1=counterjet (leftwards, -x)
    # gamma: bulk Lorentz factor
    # beta = jet speed (max 0.99 c)
    # z: source redshift, if negative it is the distance in Mpc
    # alpha: position angle on sky from right
    # core: array, RA and DEC of jet origin in degree
 
    from scipy.constants import year, c, pi
    from astropy.cosmology import Planck15 as cosmo
    from astropy import units as u
 
    if z > 0:
        as_per_kpc = cosmo.arcsec_per_kpc_proper(z)
        as_per_kpc *= u.kpc
        as_per_kpc /= u.arcsec
        as_per_kpc = float(as_per_kpc)
    if z < 0:  # now z is the distance in Mpc
        as_per_kpc = 360. * 60. * 60. / (2. * pi * (-z) * 1000)
 
    # cgs units
    kpc = 3.0856e21
    myr = 1.e6 * year
    ccgs = c * 100.
 
    dt=t*myr
    # adjust i so 0 corresponds to "jet towards observer"
    # i = i-pi/2
 
    d = 1.  # distance to object, scale-free:1.
    omega = 2. * pi / (pp * myr)  # precession frequency
    # scale = 1. / ( (180./pi)*60 )  # arcmin
 
    # Lorentz factor of jet
    c = ccgs  # speed of light
    gamma = 1 / (np.sqrt(1 - (beta ** 2)))
    vj = beta * c
 
    # precession argument, Gower et al. eq (6)
    prec = theta - omega * dt
 
    # X-velocity, Gower et al. eq (1)
    vx = s_jet * beta * c
    vx *= (np.sin(psi) * np.sin(i) * np.cos(prec) + np.cos(psi) * np.cos(i))
 
    # Y-velocity, Gower et al. eq (2), remember prec from eq (6)
    vy = s_jet * beta * c
    vy *= np.sin(psi) * np.sin(prec)
 
    # Z-velocity, Gower et al. eq (3), remember prec from eq (6)
    vz = s_jet * beta * c
    vz *= (np.cos(psi) * np.sin(i) - np.sin(psi) * np.cos(i) * np.cos(prec))
 
    # calculate the sky pattern / units of cm
    y = vy * dt / (1 - vx / c)
    z = vz * dt / (1 - vx / c)
 
    # convert to kpc
    ykpc = y / kpc
    zkpc = z / kpc
 
    # convert to arcsec
    yas = ykpc * as_per_kpc
    zas = zkpc * as_per_kpc
 
    # rotate on the sky
    yrot = np.cos(alpha) * yas - np.sin(alpha) * zas
    zrot = np.sin(alpha) * yas + np.cos(alpha) * zas
 
    # convert to degree
    ydeg = yrot / 60. / 60.
    zdeg = zrot / 60. / 60.
 
    # adjust to core position
    if core is not None:
        ydeg += core[0]
        zdeg += core[1]
 
    return ydeg, zdeg

if __name__=='__main__':

    import matplotlib.pyplot as plt

    t=np.linspace(0,1.0,1000)
    x,y = jetpath(t,alpha=-0.5,psi=0.1,z=0.1)
    
    plt.plot(x,y)
    plt.show()
    
