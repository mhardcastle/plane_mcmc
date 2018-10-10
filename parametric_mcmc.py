from __future__ import print_function
import emcee
import corner
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Times'],'size':12})
rc('text', usetex=True)

import numpy as np
from tqdm import tqdm
from scipy.stats import halfcauchy
import sys

from jet_fn import f
from parametric_lf import lf
from parametric_generate import Data
from estimation import find_mode

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
        self.variance_prior_scale = width_prior_scale
        self.ndim = 7

    def set_range(self,rmin,rmax):
        self.rmin=rmin
        self.rmax=rmax
        
    def lnlike(self,X):
        return lf(self.xd,self.yd,self.xderr, self.yderr, X)

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

if __name__=='__main__':

    iterations = 2000
    burnin = 500
    lkf=Likefn('data.npy')
    lkf.set_range([0,0,0,-0.5,0.1,0],[np.pi/2,np.pi/4,2*np.pi,0.5,0.9999,2*np.pi])
    if len(sys.argv)==1:
        
        nwalkers=96
        pos=lkf.initpos(nwalkers)
        sampler = emcee.EnsembleSampler(nwalkers, lkf.ndim, lkf,threads=8)

        for _ in tqdm(sampler.sample(pos, iterations=iterations), total=iterations):
            pass

        np.save('chain.npy',sampler.chain)
        chain=sampler.chain
    else:
        chain=np.load('chain.npy')
    labels=('Inclination angle $i$','Cone angle $\\psi$','Phase $\\theta$','$\log_{10}(p/{\\rm Myr})$','$\\beta$','Pos angle $\\alpha$', 'line width $V$')
    plt.figure(figsize=(6,1.5*lkf.ndim))
    for i in range(lkf.ndim):
        plt.subplot(lkf.ndim,1,i+1)
        plt.plot(chain[:,:,i].transpose())
        plt.ylabel(labels[i])

    samples=chain[:, burnin:, :].reshape((-1, lkf.ndim))
    fig = corner.corner(samples,plot_contours=False,plot_density=True,plot_datapoints=False, truths=lkf.data.true_params+[None], labels=labels)
    plt.savefig('corner.pdf')
    
    print(samples.shape)
    me=[]
    be=[]
    for i in range(lkf.ndim):
        estimate=np.mean(samples[:,i])
        print(i,estimate)
        be.append(estimate)
        me.append(np.median(samples[:,i]))
    be=np.array(be)
    me=np.array(me)
    mode, _ = find_mode(samples)
        
    fig, ax = plt.subplots()

    t=np.linspace(0, 1, 1000)
    true_x, true_y = f(t, lkf.data.true_params)
    est_x, est_y = f(t, be)
    mest_x, mest_y = f(t, me)
    modest_x, modest_y = f(t, mode)
    print('Truth:    ',lkf.data.true_params)
    print('Estimate: ',be)
    print('Median: ',me)
    print('Mode: ',mode)
    print('Random:   ',samples[0])
    

    ax.errorbar(lkf.xd, lkf.yd, xerr=lkf.xderr, yerr=lkf.yderr, fmt='r+')
    ax.plot(est_x, est_y, '-', color='magenta', label='Bayesian estimator')
    ax.plot(mest_x, mest_y, '-', color='blue', label='Median estimator')
    ax.plot(modest_x, modest_y, '-', color='orange', label='Mode estimator')
    ax.plot(true_x, true_y, 'g--', label='truth')
    for i in np.random.choice(samples.shape[0], size=100):
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
