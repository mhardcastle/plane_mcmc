import numpy as np

def f(t,pars):
    # cumbersome approach for demonstration purposes!
    a=pars[0]
    b=pars[1]
    c=pars[2]
    d=pars[3]
    e=pars[4]
    f=pars[5]
    x=a*t+b*np.sin(c*t)
    y=d*t+e*np.cos(f*t)
    return x,y
