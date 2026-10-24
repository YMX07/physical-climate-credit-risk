"""Standard Basel IRB / Vasicek ASRF building blocks (Section 3-4, Eq. 1-3)."""

import numpy as np
from scipy.stats import norm


def basel_correlation(pd):
    """Basel IRB corporate asset correlation (Section 3)."""
    w = (1.0 - np.exp(-50.0 * pd)) / (1.0 - np.exp(-50.0))
    return 0.12 * w + 0.24 * (1.0 - w)


def maturity_b(pd):
    """Basel maturity-adjustment slope b(PD)."""
    return (0.11852 - 0.05478 * np.log(pd)) ** 2


def maturity_adjustment(pd, maturity=2.5):
    """MA = (1 + (M - 2.5) b) / (1 - 1.5 b)."""
    b = maturity_b(pd)
    return (1.0 + (maturity - 2.5) * b) / (1.0 - 1.5 * b)


def vasicek_conditional_pd(pd, rho, Q=0.999):
    """C_V(Q) = Phi((Phi^-1(PD) + sqrt(rho) Phi^-1(Q)) / sqrt(1 - rho))."""
    return norm.cdf((norm.ppf(pd) + np.sqrt(rho) * norm.ppf(Q)) / np.sqrt(1.0 - rho))


def irb_capital(pd, lgd, rho, Q=0.999, maturity=None):
    """Capital requirement K per unit EAD, Eq. (1)/(3): LGD (C_V(Q) - PD) [x MA]."""
    k = lgd * (vasicek_conditional_pd(pd, rho, Q) - pd)
    if maturity is not None:
        k = k * maturity_adjustment(pd, maturity)
    return k


def irb_rwa(pd, lgd, rho, ead=1.0, Q=0.999, maturity=2.5):
    """Model RWA, Eq. (1): K x MA x EAD x 12.5."""
    return irb_capital(pd, lgd, rho, Q, maturity) * ead * 12.5
