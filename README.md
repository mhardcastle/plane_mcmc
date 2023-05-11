# plane_mcmc
MCMC fitting to samples of 2D curves applied to precessing jets

Code by Maya Horton, Martin Krause, Shaun Read, Martin Hardcastle (2018 onwards)

Code in the directory:

* `jet_fn.py` -- the precessing jet function to fit. Wrapper around `jetpath`
* `jetpath.py` -- MK's jet path function
* `parametric_generate.py` -- make some data to fit to
* `parametric_mcmc.py` -- run the MCMC model.

OVERVIEW
The purpose of this code is to use a Markov Chain Monte Carlo approach to project a best-fitting precessing jet path onto a 2D image of an AGN jet. This path can then be used to constrain the separation distances of a hypothetical close supermassive black hole binary that could be responsible for such precession. The current code has been developed to proof-of-concept stage (Horton et al, 2020). It works best on well-resolved sources with clear jet paths in both lobes, and clear terminal hotspots.   


INSTRUCTIONS
The plane_fitting code can be run on a single-sided jet or a Fanaroff-Riley II style lobe with both jet and counterjet. It can be used with simulated or real data. It is easiest to start on a single-sided jet with simulated data and build up to real-world sources. Cygnus A can be used as a test source; region files and fits images can be provided. 

1. Generated data


2. Counterjet


3. Real sources

[instructions TBD]

