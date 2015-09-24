from scipy.stats import norm
import numpy as np

def gaussian_corrections(t):
    """Returns the additive and multiplicative corrections for the mean
    and variance of a trunctated Gaussian random variable.

    In Trueskill/AdPredictor papers, denoted
    - V(t)
    - W(t) = V(t) * (V(t) + t)

    Returns (v(t), w(t))
    """
    # Clipping avoids numerical issues from ~0/~0.
    t = np.clip(t, -5, 5)
    v = norm.pdf(t) / norm.cdf(t)
    w = v * (v + t)
    return (v, w)

def kl_divergence(p, q):
    """Computes the Kullback-Liebler divergence between two Bernoulli
    random variables with probability p and q.
    Algebraically, KL(p || q)
    """
    q = np.clip(q, 0.001, 0.999)
    return p * np.log(p / q) + (1.0 - p) * np.log((1.0 - p) / (1.0 - q))

if __name__ == "__main__":
    print kl_divergence(0.1, 0.0)
    print gaussian_corrections(-6.0)

