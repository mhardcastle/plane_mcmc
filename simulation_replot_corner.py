import matplotlib
matplotlib.use('agg')
from simulation_mcmc import analyse_mcmc
from simulation_likefn import Likefn
import numpy as np
import sys

name=sys.argv[1]
lkf=Likefn.load(name+'.pickle')
chain=np.load(name+'.npy')

analyse_mcmc(lkf, chain, do_plot=True, do_print=True, do_show=False)
