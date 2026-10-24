"""Heterogeneous portfolios: exact ASRF loss quantile and Monte Carlo checks.

All obligors share the economic factor y and the climate event xi; idiosyncratic
risk diversifies away in the infinitely granular (ASRF) limit."""

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

from .correlated import joint_cdf_from_channels


def _as_arrays(**kw):
    arrs = {k: np.atleast_1d(np.asarray(v, float)) for k, v in kw.items()}
    n = max(a.size for a in arrs.values())
    return {k: np.broadcast_to(a, (n,)) for k, a in arrs.items()}


def asrf_loss(y, xi, pd0, rho, alpha_hat, lgd0, lgd1, weights):
    """Portfolio loss (share of total EAD) given y and the event indicator xi."""
    y = np.asarray(y, float)[..., None]
    xi = np.asarray(xi, float)[..., None]
    z = (np.sqrt(rho) * y + norm.ppf(pd0) + xi * alpha_hat) / np.sqrt(1 - rho)
    lgd = np.where(xi > 0, lgd1, lgd0)
    return np.sum(weights * lgd * norm.cdf(z), axis=-1)


def portfolio_var_exact(pd0, rho, q, alpha_hat, lgd0, lgd1, weights, Q=0.999, tau=0.0):
    """Exact Q-quantile of the ASRF loss of a heterogeneous portfolio.

    Loss is increasing in y within each event channel, so
    P(L < x) = (1-q) Phi(y0(x)) + q Phi(y1(x)) with yk the channel inverse.
    tau != 0 correlates the event with y (see correlated.py)."""
    p = _as_arrays(pd0=pd0, rho=rho, alpha_hat=alpha_hat, lgd0=lgd0, lgd1=lgd1, weights=weights)
    w = p["weights"] / p["weights"].sum()
    args = (p["pd0"], p["rho"], p["alpha_hat"], p["lgd0"], p["lgd1"], w)

    def y_inv(x, xi):
        f = lambda y: asrf_loss(y, xi, *args) - x
        lo, hi = -40.0, 40.0
        if f(hi) <= 0:
            return np.inf
        if f(lo) >= 0:
            return -np.inf
        return brentq(f, lo, hi, xtol=1e-12)

    if tau == 0.0:
        cdf = lambda x: (1 - q) * norm.cdf(y_inv(x, 0)) + q * norm.cdf(y_inv(x, 1))
    else:
        cdf = lambda x: joint_cdf_from_channels(y_inv(x, 0), y_inv(x, 1), q, tau)
    top = float(np.sum(w * np.maximum(p["lgd0"], p["lgd1"])))
    return brentq(lambda x: cdf(x) - Q, 1e-14, top * (1 - 1e-12), xtol=1e-14)


def simulate_losses(pd0, rho, q, alpha_hat, lgd0, lgd1, weights, n_sims=1_000_000,
                    n_obligors=None, seed=0, chunk=200_000, tau=0.0):
    """Monte Carlo portfolio losses (share of total EAD).

    n_obligors=None uses the ASRF limit (conditional expected loss); an integer
    simulates that many obligors per loan class with binomial default counts.
    tau correlates the event with the economic factor: xi = 1{tau y + sqrt(1-tau^2) eps > Phi^-1(1-q)}."""
    p = _as_arrays(pd0=pd0, rho=rho, alpha_hat=alpha_hat, lgd0=lgd0, lgd1=lgd1, weights=weights)
    w = p["weights"] / p["weights"].sum()
    rng = np.random.default_rng(seed)
    out = []
    for start in range(0, n_sims, chunk):
        m = min(chunk, n_sims - start)
        y = rng.standard_normal(m)
        w_latent = tau * y + np.sqrt(1.0 - tau**2) * rng.standard_normal(m)
        xi = (w_latent > norm.ppf(1.0 - q)).astype(float)
        if n_obligors is None:
            out.append(asrf_loss(y, xi, p["pd0"], p["rho"], p["alpha_hat"], p["lgd0"], p["lgd1"], w))
            continue
        z = (np.sqrt(p["rho"]) * y[:, None] + norm.ppf(p["pd0"]) + xi[:, None] * p["alpha_hat"]) / np.sqrt(1 - p["rho"])
        dr = rng.binomial(n_obligors, norm.cdf(z)) / n_obligors
        lgd = np.where(xi[:, None] > 0, p["lgd1"], p["lgd0"])
        out.append(np.sum(w * lgd * dr, axis=1))
    return np.concatenate(out)
