"""Section 5 extensions: multi-state climate model, transition risk, damage-index hedging."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

from .model import _channel_cdf, calibrate_alpha_hat, lgd_climate


# --- Multi-state climate model, Eq. (23) -------------------------------------

@dataclass
class ClimateStates:
    """Discrete distribution of systematic asset jumps: state i has probability
    probs[i], normalised jump alpha_hats[i] and LGD lgds[i]."""

    probs: np.ndarray
    alpha_hats: np.ndarray
    lgds: np.ndarray

    def __post_init__(self):
        self.probs = np.asarray(self.probs, float)
        self.alpha_hats = np.asarray(self.alpha_hats, float)
        self.lgds = np.asarray(self.lgds, float)
        if not np.isclose(self.probs.sum(), 1.0):
            raise ValueError("state probabilities must sum to 1")

    @classmethod
    def binary(cls, q, alpha_hat, lgd0, lgd1):
        return cls([1.0 - q, q], [0.0, alpha_hat], [lgd0, lgd1])

    @classmethod
    def from_asset_jumps(cls, probs, alphas, lgd0, asset_vol):
        """Build states from asset-value jumps alpha_i (exp(-alpha_i) haircut), with
        LGD_i from Eq. (15) and alpha_hat_i = alpha_i / asset_vol."""
        alphas = np.asarray(alphas, float)
        return cls(probs, alphas / asset_vol, lgd_climate(lgd0, alphas))


def multistate_pd(pd0, states):
    """Unconditional PD = Phi_{q,a}(Phi^-1(PD0)) for the vector deformation."""
    c = norm.ppf(pd0)
    return float(np.sum(states.probs * norm.cdf(c + states.alpha_hats)))


def multistate_conditional_pd(pd0, rho, y, states):
    """PD conditional on the economic factor y, averaged over climate states."""
    z = (np.sqrt(rho) * np.asarray(y, float)[..., None] + norm.ppf(pd0) + states.alpha_hats) / np.sqrt(1 - rho)
    return np.sum(states.probs * norm.cdf(z), axis=-1)


def multistate_expected_loss(pd0, states):
    c = norm.ppf(pd0)
    return float(np.sum(states.probs * states.lgds * norm.cdf(c + states.alpha_hats)))


def multistate_loss_cdf(loss, pd0, rho, states):
    """Eq. (23): P(Loss < L) = sum_i q_i W_i(L / LGD_i)."""
    loss = np.asarray(loss, float)
    return sum(p * _channel_cdf(loss / lgd, pd0, rho, a)
               for p, a, lgd in zip(states.probs, states.alpha_hats, states.lgds))


def multistate_var(pd0, rho, states, Q=0.999):
    hi = states.lgds.max()
    return brentq(lambda L: multistate_loss_cdf(L, pd0, rho, states) - Q,
                  1e-14, hi * (1 - 1e-14), xtol=1e-14)


def multistate_capital(pd0, rho, states, Q=0.999):
    """Exact unexpected loss per unit EAD: VaR - <EL>."""
    return multistate_var(pd0, rho, states, Q) - multistate_expected_loss(pd0, states)


# --- Physical + transition risk, Eq. (24)-(25) -------------------------------

def joint_states(q, alpha_hat, transition_probs, delta_hats, lgd0, asset_vol):
    """Combine the binary physical jump xi with the three-state transition jump
    omega (none / orderly / disorderly), assumed independent, Eq. (24).
    Log-jumps add: total normalised jump = alpha_hat xi + delta_hat_omega."""
    phys_p, phys_a = np.array([1 - q, q]), np.array([0.0, alpha_hat])
    tr_p = np.asarray(transition_probs, float)
    tr_d = np.concatenate([[0.0], np.asarray(delta_hats, float)])
    probs = np.outer(phys_p, tr_p).ravel()
    shifts = np.add.outer(phys_a, tr_d).ravel()
    return ClimateStates(probs, shifts, lgd_climate(lgd0, shifts * asset_vol))


def calibrate_transition(pd0, pd_physical, pd_transition, pd_joint, q, delta_hats):
    """Solve the self-consistency system (25).

    Given external q and stress-test transition jumps (delta_hat_1, delta_hat_2),
    recover alpha_hat (closed form from PD_physical) and the transition state
    probabilities (p1, p2), which enter linearly and solve a 2x2 system.

    The joint-PD equation is close to collinear with the transition-PD equation
    when jumps are independent, so the returned condition number should be checked.
    """
    c = norm.ppf(pd0)
    a = float(calibrate_alpha_hat(pd0, pd_physical, q))
    d = np.asarray(delta_hats, float)
    g = lambda s: (1 - q) * norm.cdf(c + s) + q * norm.cdf(c + a + s)
    A = np.array([
        [norm.cdf(c + d[0]) - pd0, norm.cdf(c + d[1]) - pd0],
        [g(d[0]) - pd_physical, g(d[1]) - pd_physical],
    ])
    b = np.array([pd_transition - pd0, pd_joint - pd_physical])
    p1, p2 = np.linalg.solve(A, b)
    p = np.array([1 - p1 - p2, p1, p2])
    if np.any(p < -1e-12) or np.any(p > 1 + 1e-12):
        raise ValueError(f"inconsistent PDs: implied transition probabilities {p}")
    return {"alpha_hat": a, "transition_probs": p, "condition_number": float(np.linalg.cond(A))}


# --- Climate damage index swap, Section 5 --------------------------------------

def swap_spread(q, alpha):
    """Fair spread of a digital climate damage index swap: (1 - e^-alpha) q."""
    return (1.0 - np.exp(-alpha)) * q


def hedged_multiplier(q, alpha, lgd0, beta=1.0, hedge_ratio=0.0):
    """(1 + swap x beta x (1 - LGD0) / LGD0) with a fraction hedge_ratio of the
    climate exposure offloaded to an ideal counterparty. hedge_ratio=1 returns the
    pre-climate multiplier of 1; the paper's perfect hedge also resets PD to PD0."""
    return 1.0 + (1.0 - hedge_ratio) * swap_spread(q, alpha) * beta * (1.0 - lgd0) / lgd0
