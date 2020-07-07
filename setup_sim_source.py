from __future__ import print_function
import sys
import numpy as np
from simulation_likefn import Likefn
import matplotlib.pyplot as plt

lkf=Likefn(name='simulation')

for i in range(2):
    o=np.load('offsets-300-2-%i.npy' % i)
    errors=np.ones_like(o[0])*2
    o[0]*=-1 if i else 1
    plt.plot(o[0],o[1])
    lkf.add_side(o[0],o[1],errors,errors)

plt.show()
    
prior_min=[0,0,0,-1.0,20,4.00]
prior_max=[np.pi/2,np.pi/4,2*np.pi,1.0,200,5.0]

lkf.set_range(prior_min,prior_max)

lkf.save('simulation.pickle')
