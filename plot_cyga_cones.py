from matplotlib import rc
import matplotlib.pyplot as plt
import aplpy
import numpy as np
import pyregion
from astropy.coordinates import SkyCoord
import astropy.units as u
import os

rc('font',**{'family':'serif','serif':['Times'],'size':12})
rc('text', usetex=True)

def plot_region(f,regname,color):
    r=pyregion.open(regname)
    ra=[]
    dec=[]
    for j,c in enumerate(r):
        sc=SkyCoord(c.coord_list[0],c.coord_list[1],unit=(u.deg,u.deg))
        ra.append(sc.ra.value)
        dec.append(sc.dec.value)
    f.show_markers(ra,dec,marker='+',facecolor=color,edgecolor=color,linewidth=1,s=100,zorder=100)
    

chains=np.load('/home/mjh/git/plane_mcmc/cygnusa_as.npy')
burnin=400                                         
samples=chains[:, burnin:, :].reshape((-1, 7))     

x0=514.5
y0=306.5

for peakno in range(2):
    f = aplpy.FITSFigure('/home/mjh/fits/3C405.J2000.FITS')
    f.show_colorscale(vmin=1e-3, vmax=1,stretch='log', cmap='Blues')
    plot_region(f,'/home/mjh/regionfiles/CygnusJetMulti.reg','red')
    plot_region(f,'/home/mjh/regionfiles/CygnusCounterMulti.reg','red')
    if peakno==0:
        peak=samples[samples[:,5]>4.98]
    else:
        peak=samples[samples[:,5]<4.98]
        
    for i in np.random.choice(peak.shape[0], size=500):
        sample=peak[i]
        coneangle=sample[1]
        radius=350.0
        for offset in [0,np.pi]:
            posangle=sample[5]+offset
            lowerx=x0-radius*np.sin(posangle+coneangle)
            lowery=y0+radius*np.cos(posangle+coneangle)
            upperx=x0-radius*np.sin(posangle-coneangle)
            uppery=y0+radius*np.cos(posangle-coneangle)

            plt.plot((x0,lowerx),(y0,lowery),'k-',alpha=0.01)
            plt.plot((x0,upperx),(y0,uppery),'k-',alpha=0.01)


    plt.tight_layout()
    plt.savefig('peak-%i.pdf' % (peakno+1))
    plt.show()


#plt.savefig('cyga_regions.pdf')
# make a cropped version for use in the paper
#os.system('/soft/bin/pdfcrop cyga_regions.pdf')


