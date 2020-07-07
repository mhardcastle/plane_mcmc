from parametric_mcmc import analyse_mcmc
from likefn import Likefn
import numpy as np

lkf=Likefn.load('cygnusa_as_t.pickle')
chain=np.load('cygnusa_as_t.npy')

analyse_mcmc(lkf, chain, do_plot=True, do_print=False, do_show=False)
