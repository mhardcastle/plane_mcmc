import numpy as np

def jetpath(t, i=1.6, psi=0.3, theta=0, pp=0.3, s_jet=1, v=100, alpha=0, scale=1.0, core=None):
    '''Version of the jet path code that works for our simulation setup.
    This is like the Gower jet path code but has no light travel time
    -- i.e. set c to infinity and drop negligible terms. Then we have
    to provide a scaling factor.'''

    dt=t # scale for time later
    # adjust i so 0 corresponds to "jet towards observer"
    # i = i-pi/2
 
    d = 1.  # distance to object, scale-free:1.
    omega = 2. * np.pi / pp  # precession frequency
    # scale = 1. / ( (180./pi)*60 )  # arcmin
 
    # Lorentz factor of jet
    vj = v
 
    # precession argument, Gower et al. eq (6)
    prec = theta - omega * dt
 
    # X-velocity, Gower et al. eq (1)
    vx = s_jet * v
    vx *= (np.sin(psi) * np.sin(i) * np.cos(prec) + np.cos(psi) * np.cos(i))
 
    # Y-velocity, Gower et al. eq (2), remember prec from eq (6)
    vy = s_jet * v
    vy *= np.sin(psi) * np.sin(prec)
 
    # Z-velocity, Gower et al. eq (3), remember prec from eq (6)
    vz = s_jet * v
    vz *= (np.cos(psi) * np.sin(i) - np.sin(psi) * np.cos(i) * np.cos(prec))
 
    # calculate the sky pattern
    y = vy * dt 
    z = vz * dt 
  
    # rotate on the sky
    yrot = scale * (np.cos(alpha) * y - np.sin(alpha) * z)
    zrot = scale * (np.sin(alpha) * y + np.cos(alpha) * z)

    if core is not None:
        yrot+=core[0]
        zrot+=core[1]
    
    return np.array([yrot, zrot])

if __name__=='__main__':

    import matplotlib.pyplot as plt

    t=np.linspace(0,0.09,1000) # time in simulation units
    i=0.79
    alpha=-0.16+-np.pi/2
    psi=15*np.pi/180.0
    pp=1.0
    scale=512.0/5.0 # projected pixels to sim domain
    
    x,y = jetpath(t,i=i,alpha=alpha,psi=psi,pp=pp,scale=scale)
    plt.plot(x,y,label='jet')
    o=np.load('offsets-300-2-0.npy')
    # x,y = jetpath(t,i=i,alpha=alpha,psi=psi,pp=pp,scale=scale,s_jet=-1)
    # plt.plot(x,y,label='counterjet')
    # plt.legend(loc=0)
    plt.scatter(o[0],o[1])
    plt.show()
    
