# example -- run 3 times with a particular set of truth values and
# collect credible interval differences

from __future__ import print_function

import numpy as np
from parametric_generate import generate
from parametric_mcmc import Likefn, run_mcmc, analyse_mcmc

truth = [1.571,0.262,np.pi,-0.5,0.60,3*np.pi/4]

widths=[]
runs=40

outfile=open('inclination-out.txt','w')
outfile2=open('widths-out.txt','w')

for j in range(10,100,10):
    truth[0]=np.pi*j/180.0
    for i in range(runs):
        generate(points=30, truth=truth, plot=False)
        print('Running MCMC for iteration',i+1)
        lkf,chain = run_mcmc()
        #lkf=Likefn('data.npy')
        #chain=np.load('chain.npy')
        results=analyse_mcmc(lkf, chain, do_plot=False, do_print=False)
        # Now results is a dictionary with some useful stuff in it
        width=results['Upper'][3]-results['Lower'][3]
        widths.append(width)
        del(lkf)
        del(chain)
        del(results)

    print('----------------- Inclination angle %f ---------------' % j)
    print('Mean credible interval over all runs is',np.mean(widths))
    print('Error on the mean is',np.std(widths)/np.sqrt(runs-1))
    outfile.write('%f %f %f\n' % (j,np.mean(widths),np.std(widths)/np.sqrt(runs-1)))
    outfile.flush()
    outfile2.write('%f %s\n' % (j,str(widths)))
    outfile2.flush()
    
