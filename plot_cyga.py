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
    

f = aplpy.FITSFigure('/home/mjh/fits/3C405.J2000.FITS')
f.show_colorscale(vmin=1e-3, vmax=1,stretch='log', cmap='Blues')
plot_region(f,'/home/mayaahorton/regionfiles/CygnusJetMulti.reg','red')
plot_region(f,'/home/mayaahorton/regionfiles/CygnusCounterMulti.reg','red')



plt.tight_layout()
plt.savefig('cyga_regions.pdf')
# make a cropped version for use in the paper
os.system('/soft/bin/pdfcrop cyga_regions.pdf')


plt.show()
