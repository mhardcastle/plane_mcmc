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

The plane_fitting code can be run on a single-sided jet or a Fanaroff-Riley II style lobe with both jet and counterjet. It can be used with simulated or real data. It is easiest to start on a single-sided jet with simulated data and build up to real-world sources. Cygnus A can be used as a test source; region files and fits images can be provided. 

It was designed to be used with 3C radio sources, particularly the `3CRR' survey. A list of 3C sources showing precession indicators can be found in Krause (2019). 

FITS images for most of the named 3C sources are available at https://3crr.extragalactic.info/ and 3C 405 (Cygnus A) can be downloaded directly from https://www.extragalactic.info/~mjh/3C405.FITS 

BIBLIOGRAPHY
Gower et al (1982) -- Relativistic precessing jets in quasars and radio galaxies
DOI: 10.1086/160442
Description of precessing jet model used by the MCMC code

Krause et al (2019) -- How frequent are close supermassive binary black holes in powerful jet sources?
DOI: 10.1093/mnras/sty2558
Overview of science case and list of potentially precessing sources

Horton et al (2020a) -- A Markov chain Monte Carlo approach for measurement of jet precession in radio-loud active galactic nuclei
DOI: 10.1093/mnras/staa429
Description of MCMC model and proof-of-concept test on 3C 405 (Cygnus A)


