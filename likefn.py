# The likelihood function and data storage object

import numpy as np
import pickle
from jetpath import jetpath
from scipy.misc import logsumexp
from scipy.stats import halfcauchy

def width_prior(width, prior_scale):
    """
    Sets a halfcauchy prior with scale `prior_scale` by returning the logprob for a given input width.
    Will return -np.inf for negative widths
    """
    return halfcauchy.logpdf(width, scale=prior_scale)

class Likefn(object):
    def __init__(self, width_prior_scale=10., z=0.1,name=None):
        self.z=z
        self.sides=0
        self.xd=[]
        self.yd=[]
        self.xderr=[]
        self.yderr=[]
        self.maxr=[]
        self.ndim = 7
        self.variance_prior_scale = width_prior_scale
        self.truth=None
        self.core=None
        self.name=name

    def add_side(self, xd, yd, xderr, yderr):
        '''
        Add a list of data
        '''
        self.xd.append(np.array(xd))
        self.yd.append(np.array(yd))
        self.xderr.append(np.array(xderr))
        self.yderr.append(np.array(yderr))
        self.maxr.append(self.findmaxr(self.xd[-1],self.yd[-1],self.xderr[-1],self.yderr[-1]))
        self.sides=len(self.xd)
        
    def set_range(self,rmin,rmax):
        self.rmin=rmin
        self.rmax=rmax

    def f(self,t,pars):
        return jetpath(t,i=pars[0],psi=pars[1],theta=pars[2],pp=10**pars[3],s_jet=1,beta=pars[4],z=self.z,alpha=pars[5])

    def cf(self,t,pars):
        return jetpath(t,i=pars[0],psi=pars[1],theta=pars[2],pp=10**pars[3],s_jet=-1,beta=pars[4],z=self.z,alpha=pars[5])

    def jetfn(self,side,t,pars):
        return jetpath(t,i=pars[0],psi=pars[1],theta=pars[2],pp=10**pars[3],s_jet=-1 if side==1 else 1,beta=pars[4],z=self.z,alpha=pars[5])
    
    def findmaxr(self,xd,yd,xderr,yderr):
        r=np.max(xd**2.0+yd**2.0)
        r+=3*np.max(xderr**2.0+yderr**2.0)
        return r

    def findt(self,fparms,maxr,side,size=500):

        scale=1
        tmin=0
        found=False
        while not found:
            scale*=2
            t=np.linspace(tmin,scale,size)
            x,y=self.jetfn(side,t,fparms)
            r=x**2.0+y**2.0
            index=np.argmin(r<maxr)
            if index==0:
                tmin=scale
                continue
            return t[index]


    def lf(self, parms, side, size=1000):

        xd=self.xd[side]
        yd=self.yd[side]
        xderr=self.xderr[side]
        yderr=self.yderr[side]
        maxr=self.maxr[side]
        tmax=self.findt(parms[:-1],maxr,side)
        t=np.linspace(0,tmax,size)
        #print parms[0],tmax
        x,y=self.jetfn(side,t,parms[:-1])
        variance = parms[-1]

        # find the line integral element
        dx=x[0:-1]-x[1:]
        dy=y[0:-1]-y[1:]
        dl=np.sqrt(dx**2+dy**2)
        # find the midpoint -- lin interp to save re-evaluating the fn
        xc=(x[0:-1]+x[1:])*0.5
        yc=(y[0:-1]+y[1:])*0.5

        ll=0

        logsumdl = np.log(np.sum(dl))

        for i in range(len(xd)):
            scaledx = (xderr[i]**2 + variance)
            scaledy = (yderr[i]**2 + variance)
            prefix = 2 * np.pi * np.sqrt(scaledx) * np.sqrt(scaledy)
            exp = (xc-xd[i])**2 / scaledx
            exp += (yc-yd[i])**2 / scaledy
            ll += logsumexp(-exp / 2., b=dl/prefix) - logsumdl  # compute logp in log space
            # p=np.sum(np.exp(-((xc-xd[i])**2.0)/(2.0*scaledx)
            #                -((yc-yd[i])**2.0)/(2.0*scaledy))
            #         *dl/prefix)/np.sum(dl)
            #ll+=np.log(p)
        #print ll
        return ll

    
        
    def lnlike(self,X):
        rv=0
        for side in range(self.sides):
            rv+=self.lf(X, side)
        return rv

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

    def save(self,filename):
        '''
        Save the current state of the object.
        Parameters:
        filename -- a filename to save to
        '''
        f = file(filename, 'wb')
        pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        f.close()

    @staticmethod
    def load(filename):
        '''
        Load a previously saved object.
        Parameters:
        filename -- the name of a file to load
        '''
        with file(filename, 'rb') as f:
            return pickle.load(f)
