import numpy as np

def log_normal(X, mu, sigma):
    """Return log-likelihood of data given parameters"

    Computes the log-likelihood that the data X have been generated
    from the given parameters (mu, sigma) of the one-dimensional
    normal distribution.

    Args:
        X: vector of point samples
        mu: mean
        sigma: standard deviation
    Returns:
        a scalar log-likelihood
    """
    # mu = 1     # mean
    # sigma = 2  # standard deviation = sqrt(variance)
    N = 20     # number of datapoints

    #X is the generated data from our groundtruth distribution.
    # X = mu + sigma*np.random.randn(N,1)

    errors = []
    for i in range(N):
        error = ((X[i] - mu)**2)/(sigma**2)
        errors.append(error)


    loglik = (-N/2)*np.log(2*np.pi)-(N/2)*np.log(sigma**2)-(1/2)*sum(errors)

    return loglik