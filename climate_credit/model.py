"""Vasicek ASRF model enhanced with a binary physical-risk jump (Section 4).

Notation follows Pozdyshev, Lobanov & Ilinsky (2025), BIS WP 1274:

    pd0        P_D^0, bank's internal PD without physical risk
    pd         P_D, bank's internal PD including physical risk
    q          probability of the (systematic) climate event over the horizon
    alpha_hat  normalised asset jump, alpha / asset volatility
    alpha      asset jump, asset value is multiplied by exp(-alpha) on the event
    lgd0/lgd1  LGD in the no-event / event channel
    rho        correlation with the economic systematic factor
    Q          confidence level (0.999 for IRB)

The climate event xi is common to all obligors (it is the second systematic
factor); the economic factor y = -S is the usual Vasicek factor.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

from .basel import basel_correlation, maturity_adjustment, vasicek_conditional_pd

_SQRT_2PI = np.sqrt(2.0 * np.pi)


# --- q-deformed normal distribution, Eq. (5) -------------------------------

def mixture_cdf(x, probs, shifts):
    """Vector q-deformed normal: sum_i q_i Phi(x + a_i) (Section 5)."""
    x = np.asarray(x, dtype=float)
    return sum(p * norm.cdf(x + a) for p, a in zip(probs, shifts))


def mixture_ppf(p, probs, shifts):
    """Inverse of mixture_cdf (monotone in x, solved by bracketing)."""
    def solve(pi):
        if pi <= 0.0:
            return -np.inf
        if pi >= 1.0:
            return np.inf
        return brentq(lambda x: mixture_cdf(x, probs, shifts) - pi, -60.0, 60.0, xtol=1e-14)
    return np.vectorize(solve, otypes=[float])(p)


def q_deformed_cdf(x, q, a):
    """Phi_{q,a}(x) = (1 - q) Phi(x) + q Phi(x + a), Eq. (5)."""
    return mixture_cdf(x, (1.0 - q, q), (0.0, a))


def q_deformed_ppf(p, q, a):
    """Phi_{q,a}^{-1}(p)."""
    return mixture_ppf(p, (1.0 - q, q), (0.0, a))


# --- Calibration: self-consistency, Eq. (6) and LGD, Eq. (15) ----------------

def calibrate_alpha_hat(pd0, pd, q):
    """Solve Phi^-1(PD0) = Phi_{q,a}^-1(PD) for a (closed form).

    PD = (1-q) Phi(C*) + q Phi(C* + a) with C* = Phi^-1(PD0)
    =>  a = Phi^-1((PD - (1-q) PD0) / q) - C*.
    """
    pd0, pd, q = np.asarray(pd0, float), np.asarray(pd, float), np.asarray(q, float)
    target = (pd - (1.0 - q) * pd0) / q
    if np.any(pd < pd0) or np.any(target >= 1.0):
        raise ValueError("need PD0 <= PD < (1-q) PD0 + q for a non-negative jump")
    return norm.ppf(target) - norm.ppf(pd0)


def pd_from_alpha_hat(pd0, q, alpha_hat):
    """Eq. (4): PD = Phi_{q,a}(Phi^-1(PD0))."""
    return q_deformed_cdf(norm.ppf(pd0), q, alpha_hat)


def alpha_from_alpha_hat(alpha_hat, asset_vol):
    """alpha = alpha_hat x sqrt(sigma_V^2 + beta_V^2) (Annex 1)."""
    return alpha_hat * asset_vol


def lgd_climate(lgd0, alpha):
    """Eq. (15): LGD1 = LGD0 + (1 - e^-alpha)(1 - LGD0) (jump just before maturity)."""
    return lgd0 + (1.0 - np.exp(-alpha)) * (1.0 - lgd0)


# --- Conditional PD, Eq. (7)-(12) ----------------------------------------

def conditional_pd(pd0, rho, y, q, alpha_hat):
    """Exact PD conditional on economic factor y, averaged over the event, Eq. (9)."""
    z = (np.sqrt(rho) * y + norm.ppf(pd0)) / np.sqrt(1.0 - rho)
    return (1.0 - q) * norm.cdf(z) + q * norm.cdf(z + alpha_hat / np.sqrt(1.0 - rho))


def conditional_pd_first_order(pd0, rho, y, q, alpha_hat):
    """First order in q x alpha_hat, Eq. (11)."""
    z = (np.sqrt(rho) * y + norm.ppf(pd0)) / np.sqrt(1.0 - rho)
    return norm.cdf(z) + q * alpha_hat / (_SQRT_2PI * np.sqrt(1.0 - rho)) * np.exp(-0.5 * z**2)


def cv_climate(pd0, rho, q, alpha_hat, Q=0.999, exact=False):
    """C_V(Q): conditional PD at y = Phi^-1(Q). Eq. (12) unless exact=True (Eq. 9)."""
    f = conditional_pd if exact else conditional_pd_first_order
    return f(pd0, rho, norm.ppf(Q), q, alpha_hat)


# --- Expected loss, Eq. (17) ----------------------------------------------

def expected_loss(pd0, q, alpha_hat, lgd0, lgd1):
    """Exact <EL> = LGD0 (1-q) Phi(C*) + LGD1 q Phi(C* + a)."""
    c = norm.ppf(pd0)
    return lgd0 * (1.0 - q) * norm.cdf(c) + lgd1 * q * norm.cdf(c + alpha_hat)


def expected_loss_first_order(pd, q, lgd0, lgd1):
    """Eq. (17): <EL> = LGD0 PD (1 + q (LGD1 - LGD0) / LGD0)."""
    return lgd0 * pd * (1.0 + q * (lgd1 - lgd0) / lgd0)


# --- Portfolio loss distribution, Eq. (14) and VaR, Eq. (18)-(20) --------------

def _channel_cdf(theta, pd0, rho, shift):
    """W(theta): P(defaulted share < theta) in one channel (Annex 2)."""
    theta = np.clip(np.asarray(theta, float), 0.0, 1.0)
    with np.errstate(divide="ignore"):
        arg = (np.sqrt(1.0 - rho) * norm.ppf(theta) - norm.ppf(pd0) - shift) / np.sqrt(rho)
    return norm.cdf(arg)


def portfolio_loss_cdf(loss, pd0, rho, q, alpha_hat, lgd0, lgd1):
    """Eq. (14): P(Loss < L) for an infinitely granular portfolio."""
    return ((1.0 - q) * _channel_cdf(loss / lgd0, pd0, rho, 0.0)
            + q * _channel_cdf(loss / lgd1, pd0, rho, alpha_hat))


def portfolio_var(pd0, rho, q, alpha_hat, lgd0, lgd1, Q=0.999):
    """Exact VaR L*_Q: root of P(Loss < L) = Q, Eq. (14)."""
    hi = max(lgd0, lgd1)
    f = lambda L: portfolio_loss_cdf(L, pd0, rho, q, alpha_hat, lgd0, lgd1) - Q
    return brentq(f, 1e-14, hi * (1 - 1e-14), xtol=1e-14)


def portfolio_var_first_order(pd0, rho, q, alpha_hat, lgd0, lgd1, Q=0.999):
    """Eq. (20)/(22) form: L*_Q = C_V(Q) LGD0 (1 + q (LGD1 - LGD0) / LGD0)."""
    return cv_climate(pd0, rho, q, alpha_hat, Q) * lgd0 * (1.0 + q * (lgd1 - lgd0) / lgd0)


# --- Generalised ASRF formula, Eq. (21)-(22) -----------------------------

def climate_multiplier(q, lgd0, lgd1):
    """(1 + q (LGD1 - LGD0) / LGD0), Eq. (22)."""
    return 1.0 + q * (lgd1 - lgd0) / lgd0


def capital_generalised(pd0, pd, rho, q, alpha_hat, lgd0, lgd1, Q=0.999):
    """Unexpected loss per unit EAD, Eq. (22): LGD0 [C_V(Q) - PD] (1 + q (LGD1-LGD0)/LGD0)."""
    return lgd0 * (cv_climate(pd0, rho, q, alpha_hat, Q) - pd) * climate_multiplier(q, lgd0, lgd1)


def capital_conditional(pd0, rho, q, alpha_hat, lgd0, lgd1, Q=0.999):
    """Exact loss at y = Phi^-1(Q), averaged over the event, minus exact <EL>.

    This is the loan's contribution when it is a negligible share of a portfolio
    whose quantile is driven by y (the exposed share -> 0 limit of the exact model),
    so it stays additive. Eq. (22) is its first-order expansion in q and alpha_hat."""
    z = (np.sqrt(rho) * norm.ppf(Q) + norm.ppf(pd0)) / np.sqrt(1.0 - rho)
    loss = (1.0 - q) * lgd0 * norm.cdf(z) + q * lgd1 * norm.cdf(z + alpha_hat / np.sqrt(1.0 - rho))
    return loss - expected_loss(pd0, q, alpha_hat, lgd0, lgd1)


def capital_exact(pd0, rho, q, alpha_hat, lgd0, lgd1, Q=0.999):
    """Unexpected loss per unit EAD from the exact loss distribution: VaR(14) - <EL>."""
    return (portfolio_var(pd0, rho, q, alpha_hat, lgd0, lgd1, Q)
            - expected_loss(pd0, q, alpha_hat, lgd0, lgd1))


# --- Convenience wrapper ----------------------------------------------------

@dataclass
class PhysicalRiskLoan:
    """A single exposure with bank-internal PD0/PD, an external event probability q
    and LGD0. LGD1 is model-implied by Eq. (15) unless supplied.

    rho and the maturity adjustment are computed from PD0 by default, keeping
    Basel's correlations as given constants (Section 4)."""

    pd0: float
    pd: float
    q: float
    lgd0: float
    lgd1: float | None = None
    asset_vol: float = 0.30
    rho: float | None = None
    maturity: float | None = 2.5
    ead: float = 1.0
    Q: float = 0.999

    def __post_init__(self):
        if self.rho is None:
            self.rho = float(basel_correlation(self.pd0))
        self.alpha_hat = float(calibrate_alpha_hat(self.pd0, self.pd, self.q))
        self.alpha = float(alpha_from_alpha_hat(self.alpha_hat, self.asset_vol))
        self.lgd1_model = float(lgd_climate(self.lgd0, self.alpha))
        if self.lgd1 is None:
            self.lgd1 = self.lgd1_model

    @property
    def _ma(self):
        return 1.0 if self.maturity is None else float(maturity_adjustment(self.pd0, self.maturity))

    def cv_baseline(self):
        return float(vasicek_conditional_pd(self.pd0, self.rho, self.Q))

    def cv(self, exact=False):
        return float(cv_climate(self.pd0, self.rho, self.q, self.alpha_hat, self.Q, exact))

    def ul_baseline(self):
        """Eq. (3) without climate: LGD0 (C_V(Q) - PD0)."""
        return self.lgd0 * (self.cv_baseline() - self.pd0)

    def ul(self, method="paper"):
        """'paper' = first-order Eq. (22); 'conditional' = exact, loan is a small share
        of the exposed portfolio (additive); 'exact' = exact VaR(14) minus exact <EL>,
        whole portfolio exposed to the event (maximum concentration)."""
        if method == "paper":
            return float(capital_generalised(self.pd0, self.pd, self.rho, self.q,
                                             self.alpha_hat, self.lgd0, self.lgd1, self.Q))
        if method == "conditional":
            return float(capital_conditional(self.pd0, self.rho, self.q, self.alpha_hat,
                                             self.lgd0, self.lgd1, self.Q))
        if method == "exact":
            return float(capital_exact(self.pd0, self.rho, self.q, self.alpha_hat,
                                       self.lgd0, self.lgd1, self.Q))
        raise ValueError(method)

    def expected_loss(self, exact=True):
        if exact:
            return float(expected_loss(self.pd0, self.q, self.alpha_hat, self.lgd0, self.lgd1))
        return float(expected_loss_first_order(self.pd, self.q, self.lgd0, self.lgd1))

    def rwa_baseline(self):
        return self.ul_baseline() * self._ma * self.ead * 12.5

    def rwa(self, method="paper"):
        return self.ul(method) * self._ma * self.ead * 12.5

    def summary(self):
        return {
            "rho": self.rho,
            "alpha_hat": self.alpha_hat,
            "alpha": self.alpha,
            "exp(-alpha)": float(np.exp(-self.alpha)),
            "LGD1": self.lgd1,
            "Cv baseline": self.cv_baseline(),
            "Cv climate (Eq.12)": self.cv(),
            "Cv climate (exact Eq.9)": self.cv(exact=True),
            "UL baseline": self.ul_baseline(),
            "UL climate (paper)": self.ul("paper"),
            "UL climate (conditional)": self.ul("conditional"),
            "UL climate (exact)": self.ul("exact"),
            "RWA uplift (paper)": self.ul("paper") / self.ul_baseline() - 1.0,
            "RWA uplift (conditional)": self.ul("conditional") / self.ul_baseline() - 1.0,
            "RWA uplift (exact)": self.ul("exact") / self.ul_baseline() - 1.0,
        }
