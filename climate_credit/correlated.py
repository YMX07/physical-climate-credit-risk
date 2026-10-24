"""Climate event correlated with the economic factor (beyond the paper).

The paper assumes the event xi is independent of the economic factor. Here the
dependence is a Gaussian copula: a latent climate variable

    W = tau * y + sqrt(1 - tau^2) * eps,      event  xi = 1{W > k},  k = Phi^-1(1 - q)

where y = -S is the economic stress factor (high y = bad state) and eps is
independent noise. The marginal P(xi = 1) stays q; tau > 0 makes events more
likely in downturns. tau = 0 is the paper's model.

With A = sqrt(rho) S + sqrt(1-rho) B the normalised asset return, corr(A, W) =
-sqrt(rho) tau, and every quantity below reduces to bivariate normal CDFs.
"""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.stats import norm

from .basel import basel_correlation, vasicek_conditional_pd
from .model import _channel_cdf, lgd_climate


def bvn_cdf(a, b, r):
    """P(X < a, Y < b) for standard normals with correlation r (accurate in the tails)."""
    if a == -np.inf or b == -np.inf:
        return 0.0
    if a == np.inf:
        return float(norm.cdf(b))
    if b == np.inf:
        return float(norm.cdf(a))
    if r == 0.0:
        return float(norm.cdf(a) * norm.cdf(b))
    s = np.sqrt(1.0 - r * r)
    f = lambda x: norm.pdf(x) * norm.cdf((b - r * x) / s)
    return quad(f, -np.inf, a, epsabs=1e-15, epsrel=1e-12, limit=200)[0]


def _threshold(q):
    return norm.ppf(1.0 - q)


def event_prob_given_y(y, q, tau):
    """q(y) = P(event | economic factor y)."""
    return norm.cdf((tau * np.asarray(y, float) - _threshold(q)) / np.sqrt(1.0 - tau**2))


# --- Single obligor: PD, calibration, conditional PD, EL -------------------------

def pd_correlated(pd0, rho, q, alpha_hat, tau):
    """PD = P(A < C*, no event) + P(A < C* + a, event)."""
    c, k, r = norm.ppf(pd0), _threshold(q), -np.sqrt(rho) * tau
    return bvn_cdf(c, k, r) + norm.cdf(c + alpha_hat) - bvn_cdf(c + alpha_hat, k, r)


def calibrate_alpha_hat_correlated(pd0, pd, rho, q, tau):
    """Self-consistency (6) under correlation: solve PD(a) = bank's PD for a.

    With tau > 0 the event already coincides with bad economic states, so a
    smaller jump reproduces the same unconditional PD."""
    if pd < pd0:
        raise ValueError("need PD >= PD0")
    if pd == pd0:
        return 0.0
    f = lambda a: pd_correlated(pd0, rho, q, a, tau) - pd
    if f(40.0) <= 0:
        raise ValueError("PD too high for the event probability q")
    return brentq(f, 0.0, 40.0, xtol=1e-13)


def conditional_pd_correlated(pd0, rho, y, q, alpha_hat, tau):
    """PD given y: (1 - q(y)) Phi(z) + q(y) Phi(z + a / sqrt(1 - rho))."""
    z = (np.sqrt(rho) * np.asarray(y, float) + norm.ppf(pd0)) / np.sqrt(1.0 - rho)
    qy = event_prob_given_y(y, q, tau)
    return (1.0 - qy) * norm.cdf(z) + qy * norm.cdf(z + alpha_hat / np.sqrt(1.0 - rho))


def expected_loss_correlated(pd0, rho, q, alpha_hat, lgd0, lgd1, tau):
    c, k, r = norm.ppf(pd0), _threshold(q), -np.sqrt(rho) * tau
    p_no_event = bvn_cdf(c, k, r)
    p_event = norm.cdf(c + alpha_hat) - bvn_cdf(c + alpha_hat, k, r)
    return lgd0 * p_no_event + lgd1 * p_event


# --- Portfolio loss distribution -------------------------------------------------

def joint_cdf_from_channels(y0, y1, q, tau):
    """P(Loss < x) from the channel inverses y0(x), y1(x):
    P(y < y0, no event) + P(y < y1, event)."""
    k = _threshold(q)
    return bvn_cdf(y0, k, tau) + norm.cdf(y1) - bvn_cdf(y1, k, tau)


def _y_inverse(loss, pd0, rho, alpha_hat, lgd):
    """Economic factor at which the channel's ASRF loss equals `loss`."""
    theta = loss / lgd
    if theta >= 1.0:
        return np.inf
    return (np.sqrt(1.0 - rho) * norm.ppf(theta) - norm.ppf(pd0) - alpha_hat) / np.sqrt(rho)


def loss_cdf_correlated(loss, pd0, rho, q, alpha_hat, lgd0, lgd1, tau):
    """Generalises Eq. (14); equals it at tau = 0."""
    if tau == 0.0:
        return float((1 - q) * _channel_cdf(loss / lgd0, pd0, rho, 0.0)
                     + q * _channel_cdf(loss / lgd1, pd0, rho, alpha_hat))
    y0 = _y_inverse(loss, pd0, rho, 0.0, lgd0)
    y1 = _y_inverse(loss, pd0, rho, alpha_hat, lgd1)
    return joint_cdf_from_channels(y0, y1, q, tau)


def var_correlated(pd0, rho, q, alpha_hat, lgd0, lgd1, tau, Q=0.999):
    hi = max(lgd0, lgd1)
    f = lambda L: loss_cdf_correlated(L, pd0, rho, q, alpha_hat, lgd0, lgd1, tau) - Q
    return brentq(f, 1e-14, hi * (1 - 1e-12), xtol=1e-14)


# --- Capital ---------------------------------------------------------------------

def capital_correlated(pd0, pd, q, lgd0, tau, asset_vol=0.30, lgd1=None, rho=None,
                       Q=0.999, recalibrate=True, alpha_hat=None):
    """Capital per unit EAD with a correlated climate event.

    recalibrate=True keeps the bank's PD fixed and re-solves alpha_hat under tau
    (the paper's logic: PDs are inputs). recalibrate=False keeps the
    independent-case alpha_hat (or the one passed in), so PD rises with tau.
    lgd1=None uses Eq. (15) with the alpha_hat in force.

    Returns a dict with the baseline, the diversified-limit ('conditional') and
    fully concentrated ('exact') unexpected losses."""
    rho = float(basel_correlation(pd0)) if rho is None else rho
    if alpha_hat is None:
        if recalibrate:
            alpha_hat = calibrate_alpha_hat_correlated(pd0, pd, rho, q, tau)
        else:
            alpha_hat = calibrate_alpha_hat_correlated(pd0, pd, rho, q, 0.0)
    lgd1_used = float(lgd_climate(lgd0, alpha_hat * asset_vol)) if lgd1 is None else lgd1
    yq = norm.ppf(Q)
    el = expected_loss_correlated(pd0, rho, q, alpha_hat, lgd0, lgd1_used, tau)

    z = (np.sqrt(rho) * yq + norm.ppf(pd0)) / np.sqrt(1.0 - rho)
    qy = float(event_prob_given_y(yq, q, tau))
    cond_loss = (1 - qy) * lgd0 * norm.cdf(z) + qy * lgd1_used * norm.cdf(z + alpha_hat / np.sqrt(1 - rho))

    base = lgd0 * (vasicek_conditional_pd(pd0, rho, Q) - pd0)
    return {
        "tau": tau,
        "alpha_hat": alpha_hat,
        "lgd1": lgd1_used,
        "pd": float(pd_correlated(pd0, rho, q, alpha_hat, tau)),
        "q_given_yQ": qy,
        "expected_loss": el,
        "ul_baseline": float(base),
        "ul_conditional": float(cond_loss - el),
        "ul_exact": var_correlated(pd0, rho, q, alpha_hat, lgd0, lgd1_used, tau, Q) - el,
    }
