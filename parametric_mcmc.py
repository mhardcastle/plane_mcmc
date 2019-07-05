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
from scipy.stats import halfcauchy

from jet_fn import f
from parametric_lf import lf,findt,findmaxr
from parametric_generate import Data
from estimation import find_mode
from multiprocessing import Pool

def width_prior(width, prior_scale):
    """
    Sets a halfcauchy prior with scale `prior_scale` by returnin the logprob for a given input width.
    Will return -np.inf for negative widths
    """
    return halfcauchy.logpdf(width, scale=prior_scale)

class Likefn(object):
    def __init__(self, filename, width_prior_scale=10.):
        self.data = Data(*np.load(filename))
        self.xd = self.data.x
        self.yd = self.data.y
        self.xderr = self.data.xerr
        self.yderr = self.data.yerr
        self.maxr=findmaxr(self.xd,self.yd,self.xderr,self.yderr)
        self.variance_prior_scale = width_prior_scale
        self.ndim = 7

    def set_range(self,rmin,rmax):
        self.rmin=rmin
        self.rmax=rmax
        
    def lnlike(self,X):
        return lf(self.xd,self.yd,self.xderr, self.yderr, X, maxr=self.maxr)

    def lnprior(self,X):
        # use global rmin, rmax for range
        for i,v in enumerate(X[:-1]):
            if v<self.rmin[i] or v>self.rmax[i]:
                return -np.inf
        return width_prior(X[-1], self.variance_prior_scale)
        # return 0

    def lnpost(self,X):
        result=self.lnprior(X)
        if result>-np.inf:
            result+=self.lnlike(X)
        return result

    def initpos(self,nwalkers):
        # pick nwalkers random positions in the range
        pos=[]
        for i in range(len(self.rmin)):
            pos.append(np.random.uniform(self.rmin[i],self.rmax[i],size=nwalkers))

        pos.append(halfcauchy(scale=self.variance_prior_scale).rvs(size=nwalkers))

        return np.asarray(pos).T

    def __call__(self,X):
        # make the instance callable so we can use multiprocessing
        return self.lnpost(X)

def run_mcmc(filename='data.npy',iterations=5000,outname='chain.npy'):
    os.system("taskset -p 0xFFFFFFFF %d" % os.getpid())
    lkf=Likefn(filename)

    # set the priors
    # set_range sets the minimum and maximum values for a uniform prior
    # default values are:
    # inclination angle: 0 -> pi/2 (aligned->plane of sky)
    # opening angle:     0 -> pi/4
    # phase angle:       0 -> 2pi
    # log10(age/Myr):    -1 -> 1 (0.1 -- 10 Myr)
    # beta (v/c):        0.1 -> 0.9999
    # position angle:    0 -> 2pi
    
    lkf.set_range([0,0,0,-1.0,0.1,0],[np.pi/2,np.pi/4,2*np.pi,1.0,0.9999,2*np.pi])

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
    return lkf,sampler.chain

def analyse_mcmc(lkf,chain,burnin=500,do_plot=False,do_print=False):
    labels=('Inclination angle $i$','Cone angle $\\psi$','Phase $\\theta$','$\log_{10}(p/{\\rm Myr})$','$\\beta$','Pos angle $\\alpha$', 'line width $V$')
    if do_plot:
        # Chain plots
        plt.figure(figsize=(6,1.5*lkf.ndim))
        for i in range(lkf.ndim):
            plt.subplot(lkf.ndim,1,i+1)
            plt.plot(chain[:,:,i].transpose())
            plt.ylabel(labels[i])

    samples=chain[:, burnin:, :].reshape((-1, lkf.ndim))
    if do_plot:
        # corner plot
        fig = corner.corner(samples,plot_contours=False,plot_density=True,plot_datapoints=False, truths=lkf.data.true_params+[None], labels=labels)
        plt.savefig('corner.pdf')
    
    me=[]
    be=[]
    for i in range(lkf.ndim):
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
        print('Truth:    ',np.array(lkf.data.true_params))
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

        t=np.linspace(0, findt(lkf.data.true_params,maxr=lkf.maxr), 1000)
        true_x, true_y = f(t, lkf.data.true_params)
        t=np.linspace(0, findt(be,maxr=lkf.maxr), 1000)
        est_x, est_y = f(t, be)
        t=np.linspace(0, findt(me,maxr=lkf.maxr), 1000)
        mest_x, mest_y = f(t, me)
        t=np.linspace(0, findt(mode,maxr=lkf.maxr), 1000)
        modest_x, modest_y = f(t, mode)

        ax.errorbar(lkf.xd, lkf.yd, xerr=lkf.xderr, yerr=lkf.yderr, fmt='r+')
        ax.plot(est_x, est_y, '-', color='magenta', label='Bayesian estimator')
        ax.plot(mest_x, mest_y, '-', color='blue', label='Median estimator')
        ax.plot(modest_x, modest_y, '-', color='orange', label='Mode estimator')
        ax.plot(true_x, true_y, 'g--', label='truth')
        for i in np.random.choice(samples.shape[0], size=100):
            t=np.linspace(0, findt(samples[i],maxr=lkf.maxr), 1000)
            x,y=f(t, samples[i])

            ax.plot(x, y, 'k-', alpha=0.1,zorder=-100)

        plt.xlim(np.min(true_x),np.max(true_x))
        plt.ylim(np.min(true_y),np.max(true_y))
        plt.legend()
        plt.axis('equal')
        #fig, ax = plt.subplots()
        #ax.plot(sampler.lnprobability)
        plt.savefig('compare.pdf')
        plt.show()
    return results
    
if __name__=='__main__':

    lkf,chain=run_mcmc()
    analyse_mcmc(lkf,chain,do_plot=True,do_print=True)
