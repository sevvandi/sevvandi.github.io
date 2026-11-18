"""
Caron_Fox_BNPGraph.py

Python port of the generative construction from:

    F. Caron and E.B. Fox. "Sparse graphs using exchangeable random
    measures." arXiv:1401.1137, 2014.

Original MATLAB package (BNPgraph) by Francois Caron, University of Oxford
(caron@stats.ox.ac.uk), April 2015.
http://www.stats.ox.ac.uk/~caron/code/BNPgraph

This module ports the "GGP" folder (the generalized gamma process
construction itself: sampling the Levy process, and building unipartite /
bipartite graphs from it) plus the plotting utilities in "utils". It does
NOT include the MATLAB package's separate MCMC posterior-inference layer
(GGPgraphmcmc.m, GGPbipgraphmcmc.m, @graphmcmc, @graphmodel) or the
baseline Erdos-Renyi / Barabasi-Albert / Lloyd simulators used there for
comparison -- those are a different (much larger) piece of the package,
not part of "the construction" itself.

Random number generation:
Every function takes an optional `rng` argument, a numpy
`Generator` (e.g. `np.random.default_rng(0)`). If omitted, a fresh
default_rng() is created. Pass the same rng across calls for
reproducibility.
"""

import math
import warnings

import numpy as np
import scipy.sparse as sp
from scipy.special import gammaln

try:
    import matplotlib.pyplot as plt
except ImportError:  # plotting is optional
    plt = None


# =============================================================================
# GGPcheckparams
# =============================================================================

def ggp_checkparams(alpha, sigma, tau):
    """
    Check the parameters of a GGP.

    Valid range is
        alpha>0, tau>0, sigma in (-infinity, 1)
            or
        alpha>0, tau=0, sigma in (0,1)

    Raises ValueError if the parameters are outside of this range.
    """
    if tau < 0:
        raise ValueError("tau must be nonnegative")
    if alpha <= 0:
        raise ValueError("alpha must be strictly positive")
    if sigma >= 1:
        raise ValueError("sigma must be in (-infinity, 1)")
    if tau == 0 and sigma <= 0:
        raise ValueError("If tau==0, sigma must be in (0,1)")


# =============================================================================
# GGPpsi / GGPkappa
# =============================================================================

def ggp_psi(t, alpha, sigma, tau):
    """
    Laplace exponent of a GGP.

    psi(t) = -log( E[exp(-t * sum_i w_i)] ) = alpha/sigma * ((t+tau)^sigma - tau^sigma)

    where (w_i) are the points of a Poisson process on R_+ with mean measure
    rho(dw) = alpha/Gamma(1-sigma) * w^(-1-sigma) * exp(-tau*w) dw

    Parameters:
        t : array-like of positive scalars
        alpha, sigma, tau : GGP parameters

    Returns:
        Laplace exponent evaluated at t (same shape as t)
    """
    t = np.asarray(t, dtype=float)
    if sigma == 0:
        return alpha * np.log(1 + t / tau)
    return alpha / sigma * ((t + tau) ** sigma - tau ** sigma)


def ggp_kappa(n, z, alpha, sigma, tau):
    """
    nth moment of a tilted GGP.

    kappa(n,z) = int_0^inf w^n * exp(-zw) * rho(w) dw
               = alpha * (z+tau)^(n-sigma) * gamma(n-sigma) / gamma(1-sigma)

    Returns:
        (out, logout)
    """
    logout = np.log(alpha) - (n - sigma) * np.log(z + tau) + gammaln(n - sigma) - gammaln(1 - sigma)
    return np.exp(logout), logout


# =============================================================================
# igaussrnd -- inverse Gaussian sampler
# =============================================================================

def igauss_rnd(mu, lam, size=1, rng=None):
    """
    Sample from an inverse Gaussian distribution with pdf

        f(x) = sqrt(lam / (2*pi*x^3)) * exp[-lam*(x-mu)^2 / (2*mu^2*x)]

    Reference: Chhikara & Folks, "The inverse gaussian distribution:
    theory, methodology, and applications", 1989.

    Parameters:
        mu, lam : positive scalars (or broadcastable arrays)
        size    : output shape (int or tuple)
        rng     : numpy Generator (optional)

    Returns:
        numpy array of shape `size`
    """
    rng = rng or np.random.default_rng()
    shape = size if isinstance(size, tuple) else (size,)

    mu = np.asarray(mu, dtype=float)
    lam = np.asarray(lam, dtype=float)

    Y = rng.chisquare(1, size=shape)
    X1 = mu / (2 * lam) * (2 * lam + mu * Y - np.sqrt(4 * lam * mu * Y + mu ** 2 * Y ** 2))
    X2 = mu ** 2 / X1

    U = rng.uniform(size=shape)
    P = mu / (mu + X1)
    X = np.where(U < P, X1, X2)
    return X


# =============================================================================
# etstablernd -- exponentially tilted stable sampler
# =============================================================================

def _sinc(x):
    """Unnormalized sinc: sin(x)/x, with sinc(0) = 1."""
    return 1.0 if x == 0.0 else math.sin(x) / x


def _ratio_B(x, sigma):
    return _sinc(x) / (_sinc(sigma * x) ** sigma) / (_sinc((1 - sigma) * x) ** (1 - sigma))


def _zolotarev(u, sigma):
    return ((math.sin(sigma * u)) ** sigma * (math.sin((1 - sigma) * u)) ** (1 - sigma) / math.sin(u)) ** (
        1 / (1 - sigma)
    )


def _etstable_rnd_one(V0, alpha, tau, rng):
    """Draw a single sample; algorithm of Devroye (2009) with Hofert (2011) corrections."""
    # Use numpy exp/log locally: unlike math.exp/log, they saturate to
    # inf/-inf on overflow/underflow instead of raising OverflowError,
    # which the rejection loop below relies on (huge/negative values are
    # simply rejected on the next comparison, matching MATLAB's IEEE
    # Inf/-Inf semantics).
    exp = np.exp
    log = np.log

    lambda_alpha = tau ** alpha * V0
    gamma = lambda_alpha * alpha * (1 - alpha)

    xi = 1 / math.pi * ((2 + math.sqrt(math.pi / 2)) * math.sqrt(2 * gamma) + 1)
    psi = 1 / math.pi * math.exp(-gamma * math.pi ** 2 / 8) * (2 + math.sqrt(math.pi / 2)) * math.sqrt(gamma * math.pi)
    w1 = xi * math.sqrt(math.pi / 2 / gamma)
    w2 = 2 * psi * math.sqrt(math.pi)
    w3 = xi * math.pi
    b = (1 - alpha) / alpha

    def gen_U():
        V = rng.uniform()
        W_p = rng.uniform()
        if gamma >= 1:
            if V < w1 / (w1 + w2):
                return abs(rng.standard_normal()) / math.sqrt(gamma)
            return math.pi * (1 - W_p ** 2)
        else:
            if V < w3 / (w3 + w2):
                return math.pi * W_p
            return math.pi * (1 - W_p ** 2)

    while True:
        # --- generate U with density g*/G* ---
        while True:
            U = gen_U()
            W = rng.uniform()
            zeta = math.sqrt(_ratio_B(U, alpha))
            z = 1 / (1 - (1 + alpha * zeta / math.sqrt(gamma)) ** (-1 / alpha))

            term1 = xi * math.exp(-gamma * U ** 2 / 2) if (U >= 0 and gamma >= 1) else 0.0
            term2 = psi / math.sqrt(math.pi - U) if (0 < U < math.pi) else 0.0
            term3 = xi if (gamma < 1 and 0 <= U <= math.pi) else 0.0

            rho = (
                math.pi
                * exp(-lambda_alpha * (1 - zeta ** (-2)))
                * (term1 + term2 + term3)
                / ((1 + math.sqrt(math.pi / 2)) * math.sqrt(gamma) / zeta + z)
            )
            if U < math.pi and W * rho <= 1:
                break

        # --- generate X with density proportional to g(x, U) ---
        a = _zolotarev(U, alpha)
        m = (b / a) ** alpha * lambda_alpha
        delta = math.sqrt(m * alpha / a)
        a1 = delta * math.sqrt(math.pi / 2)
        a2 = a1 + delta
        a3 = z / a
        s = a1 + delta + a3

        V_p = rng.uniform()
        N_p = rng.standard_normal()
        E_p = -math.log(rng.uniform())

        if V_p < a1 / s:
            X = m - delta * abs(N_p)
        elif V_p < a2 / s:
            X = delta * rng.uniform() + m
        else:
            X = m + delta + a3 * E_p

        E = -math.log(rng.uniform())
        cond = (
            a * (X - m)
            + exp(1 / alpha * log(lambda_alpha) - b * math.log(m)) * ((m / X) ** b - 1)
            - (N_p ** 2 / 2 if X < m else 0.0)
            - (E_p if X > m + delta else 0.0)
        )
        if X >= 0 and cond <= E:
            break

    return math.exp(1 / alpha * math.log(V0) - b * math.log(X))


def etstable_rnd(V0, alpha, tau, n=1, rng=None):
    """
    Sample from an exponentially tilted stable distribution with Laplace
    transform (in z): exp(-V0 * ((z+tau)^alpha - tau^alpha))

    Parameters:
        V0    : positive scalar
        alpha : real in (0,1)
        tau   : nonnegative scalar
        n     : number of samples
        rng   : numpy Generator (optional)

    Returns:
        numpy array of length n
    """
    if not (0 < alpha < 1):
        raise ValueError("alpha must be in (0,1)")
    if tau < 0:
        raise ValueError("tau must be >= 0")
    if V0 <= 0:
        raise ValueError("V0 must be > 0")

    rng = rng or np.random.default_rng()
    return np.array([_etstable_rnd_one(V0, alpha, tau, rng) for _ in range(n)])


# =============================================================================
# GGPsumrnd -- total mass of a GGP
# =============================================================================

def ggp_sumrnd(alpha, sigma, tau, rng=None):
    """
    Sample the total mass S of a GGP, with Laplace transform

        E[e^(-t*S)] = exp(-alpha/sigma * [(t+tau)^sigma - tau^sigma])

    Returns:
        S : positive scalar
    """
    rng = rng or np.random.default_rng()

    if sigma < -1e-8:
        # Compound Poisson case: S is a Poisson mixture of gammas
        rate = -alpha / sigma / tau ** (-sigma)
        K = rng.poisson(rate)
        S = rng.gamma(-sigma * K, 1 / tau) if K > 0 else 0.0
    elif sigma < 1e-8:
        # Gamma process case
        S = rng.gamma(alpha, 1 / tau)
    elif sigma == 0.5 and tau == 0:
        # Inverse Gaussian process case, in the tau -> 0 limit.
        # The direct MATLAB formula (mu = alpha/sqrt(tau) = Inf when
        # tau==0) is numerically unstable in floating point (Inf-Inf ->
        # NaN). As mu -> infinity, IG(mu, lambda) converges to the
        # Levy(scale=lambda) distribution, which we sample directly and
        # exactly instead: S = lambda / Z^2, Z ~ N(0,1). (Verified by
        # matching Laplace transforms: both give E[e^{-tS}] = exp(-sqrt(2*lambda*t)).)
        lam = 2 * alpha ** 2
        Z = rng.standard_normal()
        S = lam / (Z ** 2)
    else:
        # General case: exponentially tilted stable
        S = float(etstable_rnd(alpha / sigma, sigma, tau, 1, rng=rng)[0])
    return S


# =============================================================================
# GGPrnd -- sample the points (jumps) of a GGP
# =============================================================================

def ggp_rnd(alpha, sigma, tau, T=None, rng=None):
    """
    Sample points of a generalized gamma process with Levy measure

        alpha/Gamma(1-sigma) * w^(-1-sigma) * exp(-tau*w)

    For sigma>=0, samples points above threshold T>0 using the adaptive
    thinning strategy of Favaro & Teh (2013).

    Parameters:
        alpha : positive scalar
        sigma : real in (-inf, 1)
        tau   : nonnegative scalar
        T     : truncation threshold (positive scalar); chosen automatically
                if omitted
        rng   : numpy Generator (optional)

    Returns:
        (N, T) : N = numpy array of GGP points, T = threshold used

    Reference:
        S. Favaro and Y.W. Teh. MCMC for normalized random measure mixture
        models. Statistical Science, vol.28(3), pp.335-359, 2013.
    """
    ggp_checkparams(alpha, sigma, tau)
    rng = rng or np.random.default_rng()

    # --- Finite activity GGP ---
    if sigma < 0:
        rate = math.exp(math.log(alpha) - math.log(-sigma) + sigma * math.log(tau))
        K = rng.poisson(rate)
        N = rng.gamma(-sigma, 1 / tau, size=K)
        N = N[N > 0]
        return N, 0.0

    # --- Infinite activity GGP: choose T automatically if not given ---
    if T is None:
        if sigma > 0.1:
            Njumps = 20000
            T = math.exp(
                1 / sigma * (math.log(alpha) - math.log(sigma) - math.lgamma(1 - sigma) - math.log(Njumps))
            )
        else:
            T = 1e-10
            if sigma > 0:
                Njumps = math.floor(alpha / sigma / math.gamma(1 - sigma) * T ** (-sigma))
            else:
                Njumps = math.floor(-alpha * math.log(T))
    else:
        if T <= 0:
            raise ValueError("Threshold T must be strictly positive")
        if sigma > 1e-3:
            Njumps = math.floor(alpha / sigma / math.gamma(1 - sigma) * T ** (-sigma))
        else:
            Njumps = math.floor(-alpha * math.log(T))
        if Njumps > 1e7:
            warnings.warn(f"Expected number of jumps = {Njumps} - this may be slow")

    def W_fn(t, x, alpha, sigma, tau):
        if tau > 0:
            logout = (
                math.log(alpha)
                + math.log(1 - math.exp(-tau * (x - t)))
                + (-1 - sigma) * math.log(t)
                + (-t * tau)
                - math.log(tau)
                - math.lgamma(1 - sigma)
            )
        else:
            logout = math.log(alpha) - math.lgamma(1 - sigma) - math.log(sigma) + math.log(t ** (-sigma) - x ** (-sigma))
        return math.exp(logout)

    def inv_W(t, x, alpha, sigma, tau):
        if tau > 0:
            return t - 1 / tau * math.log(
                1 - math.gamma(1 - sigma) * x * tau / (alpha * t ** (-1 - sigma) * math.exp(-t * tau))
            )
        logout = -1 / sigma * math.log(t ** (-sigma) - sigma * math.gamma(1 - sigma) / alpha * x)
        return math.exp(logout)

    # --- Adaptive thinning strategy ---
    N = []
    t = T
    count = 0
    while True:
        e = -math.log(rng.uniform())  # exponential(1) rv
        if e > W_fn(t, math.inf, alpha, sigma, tau):
            return np.array(N), T
        t_new = inv_W(t, e, alpha, sigma, tau)

        if tau == 0 or (math.log(rng.uniform()) < ((-1 - sigma) * math.log(t_new / t))):
            # if tau>0, adaptive thinning -- otherwise accept always
            N.append(t_new)
        t = t_new
        count += 1
        if count > 10 ** 8:
            warnings.warn(f"T too small - lowering its value to {T / 10}")
            return ggp_rnd(alpha, sigma, tau, T / 10, rng=rng)


# =============================================================================
# GGPgraphrnd -- sample an (undirected, unipartite) GGP graph
# =============================================================================

def _sample_hyper(param, rng, is_sigma=False):
    """If `param` is a length-2 (shape, rate) pair, sample from Gamma(shape, 1/rate)."""
    if np.size(param) == 2:
        a0, b0 = param
        val = rng.gamma(a0, 1.0 / b0)
        return (1 - val) if is_sigma else val
    return param


def ggp_graphrnd(alpha, sigma, tau, T=None, rng=None):
    """
    Sample a GGP graph.

    Parameters:
        alpha, sigma, tau : GGP parameters. Each may instead be a
                             length-2 (shape, rate) pair, in which case
                             it is itself drawn from a Gamma hyperprior
                             (sigma is drawn as 1 - Gamma(...)), mirroring
                             the MATLAB version's hierarchical-prior option.
        T   : truncation threshold (optional)
        rng : numpy Generator (optional)

    Returns:
        G      : scipy.sparse csr_matrix (boolean), symmetric adjacency
                 matrix of nodes with at least one connection
        w      : sociability parameters of nodes with at least one connection
        w_rem  : sum of sociability parameters of nodes with no connection
        alpha, sigma, tau : parameters used (resolved, if hyperpriors given)
    """
    rng = rng or np.random.default_rng()

    alpha = _sample_hyper(alpha, rng)
    sigma = _sample_hyper(sigma, rng, is_sigma=True)
    tau = _sample_hyper(tau, rng)

    # Sample the weights (sociability parameters)
    w, _ = ggp_rnd(alpha, sigma, tau, T, rng=rng)

    # Sample the graph conditional on the weights w (conditional Poisson model)
    cumsum_w = np.concatenate(([0.0], np.cumsum(w)))
    W_star = cumsum_w[-1]
    D_star = rng.poisson(W_star ** 2)  # total number of directed edges

    temp = W_star * rng.uniform(size=(D_star, 2))
    bin_idx = np.clip(np.searchsorted(cumsum_w, temp, side="right") - 1, 0, len(w) - 1)

    ind, ib_flat = np.unique(bin_idx, return_inverse=True)
    ib = ib_flat.reshape(bin_idx.shape)
    n_used = len(ind)

    indlog = np.zeros(len(w), dtype=bool)
    indlog[ind] = True
    w_rem = w[~indlog].sum()
    w_used = w[ind]

    G = sp.coo_matrix((np.ones(D_star), (ib[:, 0], ib[:, 1])), shape=(n_used, n_used)).tocsr()
    G = (G + G.T) > 0  # symmetrize (logical OR)

    return G, w_used, w_rem, alpha, sigma, tau


# =============================================================================
# GGPbipgraphrnd -- sample a bipartite GGP graph
# =============================================================================

def ggp_bipgraphrnd(alpha1, sigma1, tau1, alpha2, sigma2, tau2, T=None, rng=None):
    """
    Sample a GGP bipartite graph, from two independent GGPs (one per side).

    Parameters mirror ggp_graphrnd, duplicated for each side ("1"/"2"); each
    of alpha_i/sigma_i/tau_i may likewise be a length-2 hyperprior pair.

    Returns:
        G : scipy.sparse csr_matrix (boolean), shape (n1_used, n2_used)
        w1, w1_rem, w2, w2_rem : sociability parameters (as in ggp_graphrnd)
        alpha1, sigma1, tau1, alpha2, sigma2, tau2 : parameters used
    """
    rng = rng or np.random.default_rng()

    alpha1 = _sample_hyper(alpha1, rng)
    sigma1 = _sample_hyper(sigma1, rng, is_sigma=True)
    tau1 = _sample_hyper(tau1, rng)
    alpha2 = _sample_hyper(alpha2, rng)
    sigma2 = _sample_hyper(sigma2, rng, is_sigma=True)
    tau2 = _sample_hyper(tau2, rng)

    w1, _ = ggp_rnd(alpha1, sigma1, tau1, T, rng=rng)
    w2, _ = ggp_rnd(alpha2, sigma2, tau2, T, rng=rng)

    cumsum_w1 = np.concatenate(([0.0], np.cumsum(w1)))
    cumsum_w2 = np.concatenate(([0.0], np.cumsum(w2)))
    W1_star = cumsum_w1[-1]
    W2_star = cumsum_w2[-1]
    D_star = rng.poisson(W1_star * W2_star)

    temp1 = W1_star * rng.uniform(size=D_star)
    temp2 = W2_star * rng.uniform(size=D_star)
    bin1 = np.clip(np.searchsorted(cumsum_w1, temp1, side="right") - 1, 0, len(w1) - 1)
    bin2 = np.clip(np.searchsorted(cumsum_w2, temp2, side="right") - 1, 0, len(w2) - 1)

    G = sp.coo_matrix((np.ones(D_star), (bin1, bin2)), shape=(len(w1), len(w2))).tocsr()
    G = G > 0

    deg2 = np.asarray(G.sum(axis=0)).ravel()
    ind2 = np.where(deg2 > 0)[0]
    G = G[:, ind2]

    deg1 = np.asarray(G.sum(axis=1)).ravel()
    ind1 = np.where(deg1 > 0)[0]
    G = G[ind1, :]

    mask1 = np.zeros(len(w1), dtype=bool)
    mask1[ind1] = True
    mask2 = np.zeros(len(w2), dtype=bool)
    mask2[ind2] = True

    w1_rem = w1[~mask1].sum()
    w1_used = w1[mask1]
    w2_rem = w2[~mask2].sum()
    w2_used = w2[mask2]

    return G, w1_used, w1_rem, w2_used, w2_rem, alpha1, sigma1, tau1, alpha2, sigma2, tau2


# =============================================================================
# utils: plot_degree / plot_loglog
# =============================================================================

def _histc(x, edges):
    """
    Port of MATLAB's histc(x, edges): counts[k] = #{x in [edges[k], edges[k+1])}
    for k=0..len(edges)-2, and counts[-1] = #{x == edges[-1]}.
    """
    x = np.asarray(x, dtype=float)
    counts = np.zeros(len(edges))
    idx = np.searchsorted(edges, x, side="right") - 1
    at_last_edge = x == edges[-1]
    valid = (idx >= 0) & (idx < len(edges) - 1) & (~at_last_edge)
    for i in idx[valid]:
        counts[i] += 1
    counts[-1] += at_last_edge.sum()
    return counts


def plot_degree(G, ax=None, linespec="o", pas=1, **kwargs):
    """
    Plot the empirical degree distribution of graph G on a log-log scale,
    using logarithmic binning (base-2 bins from 2^0 to 2^12) for a less
    noisy estimate of the pdf.

    Parameters:
        G        : adjacency matrix (scipy.sparse or dense/ndarray)
        ax       : matplotlib axes (created if None)
        linespec : matplotlib marker/line format string
        pas      : bin step, in powers of 2 (default 1)

    Returns:
        (ax, centerbins, freq)
    """
    if sp.issparse(G):
        deg = np.asarray(G.sum(axis=0)).ravel()
    else:
        deg = np.asarray(G).sum(axis=0)
    n = G.shape[0]

    edgebins = 2.0 ** np.arange(0, 12 + pas, pas)
    sizebins = np.diff(edgebins)
    sizebins = np.append(sizebins, 1.0)

    counts = _histc(deg, edgebins)
    freq = counts / sizebins / n

    if ax is None:
        _, ax = plt.subplots()
    ax.loglog(edgebins, freq, linespec, **kwargs)
    ax.set_xlabel("Degree")
    ax.set_ylabel("Distribution")
    return ax, edgebins, freq


def plot_loglog(x, y, ax=None, linespec="*", pas=1, **kwargs):
    """
    Log-log plot of y vs x, with x-values grouped into logarithmic
    (base-2) bins and y averaged within each bin.

    Returns:
        (ax, centerbins, out)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    edgebins = 2.0 ** np.arange(0, 12 + pas, pas)
    idx = np.searchsorted(edgebins, x, side="right") - 1
    at_last_edge = x == edgebins[-1]
    idx = np.where(at_last_edge, len(edgebins) - 1, idx)

    out = np.full(len(edgebins), np.nan)
    for i in range(len(edgebins)):
        sel = (idx == i) & (idx >= 0) & (idx < len(edgebins))
        if sel.any():
            out[i] = y[sel].mean()

    if ax is None:
        _, ax = plt.subplots()
    ax.loglog(edgebins, out, linespec, **kwargs)
    return ax, edgebins, out


def sample_graph_with_n_nodes(target_n, sigma, tau, T=1e-6, rng=None,
                               tol=0.15, max_iter=20, alpha_bounds=(0.1, 30.0)):
    """Bisect on alpha until ggp_graphrnd gives ~target_n nodes."""
    rng = rng or np.random.default_rng()
    alpha_lo, alpha_hi = alpha_bounds
    best, best_diff = None, np.inf

    for _ in range(max_iter):
        alpha_mid = np.sqrt(alpha_lo * alpha_hi)  # geometric-mean bisection
        result = ggp_graphrnd(alpha_mid, sigma, tau, T, rng=rng)
        n = result[0].shape[0]
        diff = abs(n - target_n)
        if diff < best_diff:
            best, best_diff = result, diff
        if diff <= max(1, tol * target_n):
            return result
        if n < target_n:
            alpha_lo = alpha_mid
        else:
            alpha_hi = alpha_mid
    return best  # closest attempt if tolerance was never hit


