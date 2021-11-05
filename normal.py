import numpy as np

def normal(X, mu, sigma):
    """Return likelihood of data given parameters"

    Computes the likelihood that the data X have been generated
    from the given parameters (mu, sigma) of the one-dimensional
    normal distribution.

    Args:
        X: vector of point samples
        mu: mean
        sigma: standard deviation
    Returns:
        a scalar likelihood
    """
    # mu = 1     # mean
    # sigma = 2  # standard deviation = sqrt(variance)
    N = len(X)     # number of datapoints

    #X is the generated data from our groundtruth distribution.
    # X = mu + sigma*np.random.randn(N,1)

    errors = []
    for i in range(N):
        error = ((X[i] - mu)**2)/(sigma**2)
        errors.append(error)

    lik = ((2*np.pi*(sigma**2))**(-N/2))*np.exp(-(1/2)*sum(errors))

    return lik
