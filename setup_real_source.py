from __future__ import print_function
from parset import option_list
from options import options, print_options
import pyregion
from astropy.coordinates import SkyCoord
import astropy.units as u
import sys
import numpy as np
from likefn import Likefn

def die(s):
    print(s)
    sys.exit(-1)

if len(sys.argv)==1:
    print_options(option_list)
    sys.exit(1)
    
o=options(sys.argv[1:],option_list)

if o['z'] is None:
    die('Redshift must be defined')

if o['jet'] is None:
    die('Jet region file must be defined')

lkf=Likefn(z=o['z'],name=o['name'])

if o['counterjet'] is not None:
    regions=[o['jet'],o['counterjet']]
else:
    regions=o['jet']

for i, f in enumerate(regions):
    dxlist=[]
    dylist=[]
    r=pyregion.open(f)
    for j,c in enumerate(r):
        #print c.coord_list
        sc=SkyCoord(c.coord_list[0],c.coord_list[1],unit=(u.deg,u.deg))
        #print j,sc
        if i==0 and j==0:
            core=sc
            lkf.core=core
        else:
            dra,ddec=core.spherical_offsets_to(sc)
            dxlist.append(-dra.value)
            dylist.append(ddec.value)

    errors=np.ones_like(dxlist)/3600.0
    lkf.add_side(dxlist,dylist,errors,errors)

prior_min=[0,0,0,-1.0,0.1,0]
prior_max=[np.pi/2,np.pi/4,2*np.pi,1.0,0.9999,2*np.pi]

# fill out priors

prior_min[0]=o['incangle'][0]*np.pi/180.0
prior_max[0]=o['incangle'][1]*np.pi/180.0
prior_min[1]=o['openangle'][0]*np.pi/180.0
prior_max[1]=o['openangle'][1]*np.pi/180.0
# don't bother with phase
prior_min[3]=o['period'][0]
prior_max[3]=o['period'][1]
prior_min[4]=o['speed'][0]
prior_max[4]=o['speed'][1]
prior_min[5]=o['posangle'][0]*np.pi/180.0
prior_max[5]=o['posangle'][1]*np.pi/180.0

lkf.set_range(prior_min,prior_max)

lkf.save(o['name']+'.pickle')
