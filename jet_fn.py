from jetpath import jetpath

def f(t,pars):
    # cumbersome approach for demonstration purposes!
    i=pars[0]
    psi=pars[1]
    theta=pars[2]
    pp=10**pars[3]
    beta=pars[4]
    alpha=pars[5]

    return jetpath(t,i=i,psi=psi,theta=theta,pp=pp,s_jet=1,beta=beta,z=0.1,alpha=alpha)

def cf(t,pars):
    # cumbersome approach for demonstration purposes!
    i=pars[0]
    psi=pars[1]
    theta=pars[2]
    pp=10**pars[3]
    beta=pars[4]
    alpha=pars[5]

    return jetpath(t,i=i,psi=psi,theta=theta,pp=pp,s_jet=-1,beta=beta,z=0.1,alpha=alpha)
