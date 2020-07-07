from simulation_mcmc import analyse_mcmc
from simulation_likefn import Likefn
import numpy as np

lkf=Likefn.load('simulation.pickle')
chain=np.load('simulation.npy')

analyse_mcmc(lkf, chain, do_plot=True, do_print=True, do_show=False)
