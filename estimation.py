from __future__ import division
import numpy as np
from fastkde import fastKDE
from warnings import warn

__all__ = ['find_mode']


def make_indices(dimensions):
    # Generates complete set of indices for given dimensions
    level = len(dimensions)
    if level == 1:
        return list(range(dimensions[0]))
    indices = [[]]
    while level:
        _indices = []
        for j in range(dimensions[level - 1]):
            _indices += [[j] + i for i in indices]
        indices = _indices
        level -= 1
    try:
        return [tuple(i) for i in indices]
    except TypeError:
        return indices


def calc_min_interval(x, cred_mass):
    """Internal method to determine the minimum interval of
    a given width
    Assumes that x is sorted numpy array.
    credit: pymc3
    """
    n = len(x)

    interval_idx_inc = int(np.floor(cred_mass * n))
    n_intervals = n - interval_idx_inc
    interval_width = x[interval_idx_inc:] - x[:n_intervals]

    if len(interval_width) == 0:
        raise ValueError('Too few elements for interval calculation')

    min_idx = np.argmin(interval_width)
    hdi_min = x[min_idx]
    hdi_max = x[min_idx + interval_idx_inc]
    return hdi_min, hdi_max


def calc_hpd(x, cred_mass=0.68, transform=lambda x: x):
    """Calculate highest posterior density (HPD) of array for given credible interval mass. The HPD is the
    minimum width Bayesian credible interval (BCI).
    :Arguments:
      x : Numpy array
          An array containing MCMC samples
      cred_mass : float
          Desired credible interval probability mass
      transform : callable
          Function to transform data (defaults to identity)
    """
    # Make a copy of trace
    x = transform(x.copy())

    # For multivariate node
    if x.ndim > 1:

        # Transpose first, then sort
        tx = np.transpose(x, list(range(x.ndim))[1:] + [0])
        dims = np.shape(tx)

        # Container list for intervals
        intervals = np.resize(0.0, dims[:-1] + (2,))

        for index in make_indices(dims[:-1]):

            try:
                index = tuple(index)
            except TypeError:
                pass

            # Sort trace
            sx = np.sort(tx[index])

            # Append to list
            intervals[index] = calc_min_interval(sx, cred_mass)

        # Transpose back before returning
        return np.array(intervals)

    else:
        # Sort univariate node
        sx = np.sort(x)

        return np.array(calc_min_interval(sx, cred_mass))



def find_mode(trace, credible_interval_mass=0.68, restrict=True, **fastkde_kwargs):
    """
    Returns the estimated mode of your mcmc sample assuming it is generated from a continuous distribution
    , along with the Highest Posterior Density credible interval containing `cred_mass` fraction of the total 
    probability. Bandwidth is calculated independently.
    HPD is calculated using simple sorted arrays and the mode is calculated by the highest value in a KDE curve.
    
    trace: nd trace or chain from analysis of shape (nsamples, ndims)
    credible_interval_mass: The probability mass that the credible interval contains (0.68 == 1sigma)
    restrict: If true, estimate the mode only using samples within the credible interval. Slight speed bonus with reduced accuracy.
    		  The accuracy of the mode should not matter if its credible interval is larger than its accuracy.
    numPointsPerSigma: how many points per sigma interval to draw the kde with
    
	returns: mode, hpd_interval

    credit: https://stats.stackexchange.com/questions/259319/reliability-of-mode-from-an-mcmc-sample
    cite:
    """
    original_shape = trace.shape[1:]
    if trace.ndim == 1:
        trace = trace.reshape(-1, 1)
    else:
        trace = trace.reshape(trace.shape[0], -1)  # ravel last axis (first axis is steps)

    if 'numPointsPerSigma' not in fastkde_kwargs:
        fastkde_kwargs['numPointsPerSigma'] = 30

    hpd_estimates = np.atleast_2d(calc_hpd(trace, credible_interval_mass))  # (dims, interval)

    modes = np.zeros(trace.shape[-1])
    for i, (param, hpd) in enumerate(zip(trace.T, hpd_estimates)):
        if restrict:
            x = param[(param < hpd[1]) & (param > hpd[0])]
        else:
            x = param
        if hpd[0] == hpd[1]:
            modes[i] = hpd[0]
        else:
            kde = fastKDE.fastKDE(x, **fastkde_kwargs)
            modes[i] = kde.axes[0][np.argmax(kde.pdf)]

    modes = modes.reshape(original_shape)
    hpd_estimates = hpd_estimates.T.reshape((2,) + original_shape)
    if not np.all((modes <= hpd_estimates[1]) & (modes >= hpd_estimates[0])):
        warn("Mode estimation has resulted in a mode outside of the HPD region.\n"
             "HPD and mode are not reliable!")
    return modes, hpd_estimates
        
    
def add_hpd_lines_corner_plot(corner_axes, param_label_list, mode, hpd, param_name):
    i = param_label_list.index(param_name)
    ax = corner_axes[i, i]
    ax.axvline(mode, color='k', linestyle='-')
    ax.axvline(hpd[0], color='k', linestyle='--')
    ax.axvline(hpd[1], color='k', linestyle='--')
    ax.set_title('{}$ = {:.2f}^{{+{:.2f}}}_{{-{:.2f}}}$'.format(param_name, mode, mode-min(hpd), max(hpd)-mode))


def corner_plot_hpd(data, labels, cred_mass=0.68, corner_kwargs=None, fastkde_kwargs=None):
    if corner_kwargs is None:
        corner_kwargs = {}
    if fastkde_kwargs is None:
        fastkde_kwargs = {}
    figure = corner(data, labels=labels, **corner_kwargs)
    axes = np.asarray(figure.axes).reshape(len(labels), len(labels))
    modes, hpds = find_mode(data, cred_mass, **fastkde_kwargs)
    for param, mode, hpd in zip(labels, modes, hpds):
        add_hpd_lines_corner_plot(axes, labels, mode, hpd, param)
    return figure, modes, hpds


if __name__ == '__main__':
    from corner import corner
    import matplotlib.pyplot as plt

    # simulate a model with 3 parameters
    x = np.random.normal(0, 1, size=(100, 1))
    x = np.concatenate([x, np.random.normal(3, 0.2, size=(100, 1))], axis=1)
    x = np.concatenate([x, np.random.normal(2, 0.01, size=(100, 1))], axis=1)

    fig, mode, hpd = corner_plot_hpd(x, list('abc'))
    print(mode)
    print(hpd)
    plt.show()