from __future__ import print_function
import os
import sys
import emcee
import corner
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Times'],'size':12})
rc('text', usetex=True)
from getcpus import getcpus

import numpy as np
from tqdm import tqdm

from estimation import find_mode
from multiprocessing import Pool
from likefn import Likefn

def run_mcmc(lkf,iterations=5000,outname='chain.npy'):
    os.system("taskset -p 0xFFFFFFFF %d" % os.getpid())

    # set the priors
    # set_range sets the minimum and maximum values for a uniform prior
    # default values are:
    # inclination angle: 0 -> pi/2 (aligned->plane of sky)
    # opening angle:     0 -> pi/4
    # phase angle:       0 -> 2pi
    # log10(age/Myr):    -1 -> 1 (0.1 -- 10 Myr)
    # beta (v/c):        0.1 -> 0.9999
    # position angle:    0 -> 2pi
    
    # run the MCMC -- nothing below here needs to change for a particular run

    nwalkers=96
    pos=lkf.initpos(nwalkers)

    pool=Pool()
    sampler = emcee.EnsembleSampler(nwalkers, lkf.ndim, lkf,pool=pool)

    for _ in tqdm(sampler.sample(pos, iterations=iterations), total=iterations):
            pass
    pool.terminate()
    del(pool)
    
    np.save(outname,sampler.chain)
    return sampler.chain

def analyse_mcmc(lkf,chain,burnin=400,do_plot=False,do_print=False,omit_width=True,plot_chain=True,do_show=True):
    try:
        name=lkf.name
    except:
        name=None
    if name is not None:
        prefix=name+'_'
    else:
        prefix=''
    labels=('Inclination angle $i$','Cone angle $\\psi$','Phase $\\theta$','$\log_{10}(p/{\\rm Myr})$','$\\beta$','Pos angle $\\alpha$', 'line width $V$')
    if do_plot and plot_chain:
        # Chain plots
        plt.figure(figsize=(6,1.5*lkf.ndim))
        for i in range(lkf.ndim):
            plt.subplot(lkf.ndim,1,i+1)
            plt.plot(chain[:,:,i].transpose())
            plt.ylabel(labels[i])

    samples=chain[:, burnin:, :].reshape((-1, lkf.ndim))
    if lkf.truth is not None:
        truths=list(lkf.truth)
    else:
        truths=None
    if omit_width:
        dims=lkf.ndim-1
        samples=samples[:,:dims]
        labels=labels[:-1]
    else:
        dims=lkf.ndim
        if truths is not None:
            truths.append(None)
    if do_plot:
        # corner plot
        fig = corner.corner(samples,plot_contours=False,plot_density=True,plot_datapoints=False, truths=truths, labels=labels)
        plt.savefig(prefix+'corner.pdf')
    
    me=[]
    be=[]
    for i in range(dims):
        estimate=np.mean(samples[:,i])
        be.append(estimate)
        me.append(np.median(samples[:,i]))
    be=np.array(be)
    me=np.array(me)
    mode, credible = find_mode(samples)

    results={}
    if do_print:
        np.set_printoptions(precision=6,linewidth=120)
        print('\nResults!\n')
        print('Parameters:',' '.join(labels))
        if lkf.truth is not None:
            print('Truth:    ',np.array(lkf.truth))
        print('Estimate: ',be)
        print('Median:   ',me)
        print('Mode:     ',mode)
        print('Random:   ',samples[0])

        print('\nCredible intervals:')
        print('Lower:    ',credible[0])
        print('Upper:    ',credible[1])
    results['Bayesian']=be
    results['Median']=me
    results['Mode']=mode
    results['Lower']=credible[0]
    results['Upper']=credible[1]
        
    if do_plot:
        # Plot over truth
        fig, ax = plt.subplots()
        scale=3600.0
        for side in range(lkf.sides):
            if lkf.truth is not None:
                t=np.linspace(0, lkf.findt(lkf.truth,lkf.maxr[side],side), 1000)
                true_x, true_y = scale*lkf.jetfn(side, t, lkf.truth)
                
            t=np.linspace(0, lkf.findt(be,lkf.maxr[side],side), 1000)
            est_x, est_y = scale*lkf.jetfn(side, t, be)
            t=np.linspace(0, lkf.findt(me,lkf.maxr[side],side), 1000)
            mest_x, mest_y = scale*lkf.jetfn(side, t, me)
            t=np.linspace(0, lkf.findt(mode,lkf.maxr[side],side), 1000)
            modest_x, modest_y = scale*lkf.jetfn(side, t, mode)

            ax.errorbar(scale*lkf.xd[side], scale*lkf.yd[side], xerr=scale*lkf.xderr[side], yerr=scale*lkf.yderr[side], fmt='r+',label='Data' if side==lkf.sides-1 else None)
            ax.plot(est_x, est_y, '-', color='magenta', label='Bayesian estimator' if side==lkf.sides-1 else None)
            ax.plot(mest_x, mest_y, '-', color='blue', label='Median estimator' if side==lkf.sides-1 else None)
            ax.plot(modest_x, modest_y, '-', color='orange', label='Mode estimator' if side==lkf.sides-1 else None)
            if lkf.truth is not None:
                ax.plot(true_x, true_y, 'g--', label='truth' if 1-side else None, color='lime' if side else 'green')
            
        for i in np.random.choice(samples.shape[0], size=100):
            for side in range(2):
                t=np.linspace(0, lkf.findt(samples[i],lkf.maxr[side],side), 1000)
                x,y=scale*lkf.jetfn(side, t, samples[i])
                ax.plot(x, y, 'k-', alpha=0.1,zorder=-100)

        if lkf.truth is not None:
            plt.xlim(np.min(true_x),np.max(true_x))
            plt.ylim(np.min(true_y),np.max(true_y))
        plt.legend()
        plt.axis('equal')
        plt.xlabel('Offset (arcsec)')
        plt.ylabel('Offset (arcsec)')
        #fig, ax = plt.subplots()
        #ax.plot(sampler.lnprobability)
        plt.savefig(prefix+'compare.pdf')
        if do_show:
            plt.show()
    return results
    
if __name__=='__main__':
    try:
        name=sys.argv[1]
    except IndexError:
        name='data.pickle'
    lkf=Likefn.load(name)
    chain=run_mcmc(lkf)
    analyse_mcmc(lkf,chain,do_plot=True,do_print=True)
