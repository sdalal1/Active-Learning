import numpy as np
from scipy.stats import multivariate_normal

def initialize_parameters(k, d):
    # Initialize weights randomly
    weights = np.random.rand(k)
    weights /= np.sum(weights)

    # Initialize means randomly
    means = np.random.rand(k, d)

    # Initialize covariance matrices randomly
    covariances = np.zeros((k, d, d))
    for i in range(k):
        covariances[i] = np.eye(d) * np.random.rand(d)

    return weights, means, covariances

def e_step(X, weights, means, covariances):
    k = len(weights)
    n = len(X)
    d = X.shape[1]
    gamma = np.zeros((n, k))

    for i in range(n):
        for j in range(k):
            gamma[i, j] = weights[j] * multivariate_normal.pdf(X[i], mean=means[j], cov=covariances[j])

        gamma[i] /= np.sum(gamma[i])

    return gamma

def m_step(X, gamma):
    n, d = X.shape
    k = gamma.shape[1]

    weights = np.sum(gamma, axis=0) / n

    means = np.dot(gamma.T, X) / np.sum(gamma, axis=0)[:, np.newaxis]

    covariances = np.zeros((k, d, d))
    for j in range(k):
        diff = X - means[j]
        covariances[j] = np.dot(gamma[:, j] * diff.T, diff) / np.sum(gamma[:, j])

    return weights, means, covariances

def gmm_em(X, k, max_iter=100, tol=1e-4):
    n, d = X.shape
    weights, means, covariances = initialize_parameters(k, d)

    prev_likelihood = None
    for iter in range(max_iter):
        # E-step
        gamma = e_step(X, weights, means, covariances)

        # M-step
        weights, means, covariances = m_step(X, gamma)

        # Calculate log-likelihood
        likelihood = np.sum(np.log(np.sum(weights[j] * multivariate_normal.pdf(X, mean=means[j], cov=covariances[j]), axis=1)))
        
        # Check convergence
        if prev_likelihood is not None and np.abs(likelihood - prev_likelihood) < tol:
            break
        prev_likelihood = likelihood

    return weights, means, covariances

# Example usage:
np.random.seed(42)
# Generate synthetic data
samples = 100
d = 2  # dimensionality
k = 3  # number of clusters
X = np.zeros((samples, d))
for i in range(k):
    X[i * samples // k:(i + 1) * samples // k] = np.random.multivariate_normal(
        mean=np.random.rand(d) * 3,
        cov=np.eye(d) * (0.5 + np.random.rand(d)),
        size=samples // k
    )

# Run EM algorithm
weights, means, covariances = gmm_em(X, k)
print("Weights:", weights)
print("Means:", means)
print("Covariances:", covariances)
