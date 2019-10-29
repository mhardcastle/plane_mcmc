from matplotlib import rc
import matplotlib.pyplot as plt
import aplpy
import numpy as np
import pyregion
from astropy.coordinates import SkyCoord
import astropy.units as u
import os
from likefn import Likefn

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
    
lkf=Likefn.load('cygnusa_as.pickle')
chains=np.load('cygnusa_as.npy')
burnin=400                                         
samples=chains[:, burnin:, :].reshape((-1, 7))     

x0=514.5
y0=306.5
scale=3600.0/0.125

f = aplpy.FITSFigure('/home/mjh/fits/3C405.J2000.FITS')
f.show_colorscale(vmin=1e-3, vmax=1,stretch='log', cmap='Blues')
plot_region(f,'/home/mjh/regionfiles/CygnusJetMulti.reg','red')
plot_region(f,'/home/mjh/regionfiles/CygnusCounterMulti.reg','red')
ax=plt.gca()

for i in np.random.choice(samples.shape[0], size=500):
    sample=samples[i]
    for side in range(lkf.sides):
        t=np.linspace(0, lkf.findt(samples[i],lkf.maxr[side],side), 1000)
        x,y=scale*lkf.jetfn(side, t, samples[i])
        x+=x0
        y+=y0
        plt.plot(x, y, 'k-', alpha=0.01)



plt.tight_layout()
plt.savefig(lkf.name+'_jetoverlay.pdf')
os.system('pdfcrop '+lkf.name+'_jetoverlay.pdf')
plt.show()


#plt.savefig('cyga_regions.pdf')
# make a cropped version for use in the paper
#


