# example -- run 3 times with a particular set of truth values and
# collect credible interval differences

from __future__ import print_function

import numpy as np
from parametric_generate import generate
from parametric_mcmc import run_mcmc, analyse_mcmc

truth = [0.8,0.1,np.pi,0.0,0.60,3*np.pi/4]

widths=[]
meds=[]
runs=10
parameter=3 # prec period

for i in range(runs):
    lkf=generate(points=30, sides=1,truth=truth, plot=False)
    print('Running MCMC for iteration',i+1)
    chain = run_mcmc(lkf)
    results=analyse_mcmc(lkf, chain, do_plot=False, do_print=False)
    # Now results is a dictionary with some useful stuff in it
    width=results['Upper'][parameter]-results['Lower'][parameter]
    print('Width of credible interval for run',i+1,'is',width)
    print('Median value is',results['Median'][parameter])
    widths.append(width)
    meds.append(results['Median'][parameter])

print('Mean credible interval over all runs is',np.mean(widths))
print('Error on the mean is',np.std(widths)/np.sqrt(runs-1))
print('Mean of the medians is',np.mean(meds))
print('Truth is',truth[parameter])
