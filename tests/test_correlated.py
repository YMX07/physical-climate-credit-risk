import numpy as np
import pytest
from scipy.stats import multivariate_normal

import climate_credit as cc

PD0, LGD0, Q = 0.003, 0.10, 0.999
PD = PD0 * (1 + 0.161 * 0.76)
RHO = float(cc.basel_correlation(PD0))


def test_bvn_cdf_matches_scipy():
    for a, b, r in [(-2.7, 1.9, -0.2), (0.5, -1.0, 0.6), (-1.0, -1.0, 0.0)]:
        ref = multivariate_normal([0, 0], [[1, r], [r, 1]]).cdf([a, b])
        assert cc.bvn_cdf(a, b, r) == pytest.approx(ref, abs=1e-6)


def test_tau_zero_is_the_paper_model():
    loan = cc.PhysicalRiskLoan(pd0=PD0, pd=PD, q=0.03, lgd0=LGD0)
    res = cc.capital_correlated(PD0, PD, 0.03, LGD0, tau=0.0)
    assert res["alpha_hat"] == pytest.approx(loan.alpha_hat, abs=1e-9)
    assert res["ul_conditional"] == pytest.approx(loan.ul("conditional"), rel=1e-9)
    assert res["ul_exact"] == pytest.approx(loan.ul("exact"), rel=1e-8)
    assert res["expected_loss"] == pytest.approx(loan.expected_loss(), rel=1e-9)


def test_event_marginal_probability_preserved():
    from scipy.integrate import quad
    from scipy.stats import norm
    for tau in (-0.3, 0.4):
        m = quad(lambda y: norm.pdf(y) * cc.event_prob_given_y(y, 0.03, tau), -np.inf, np.inf)[0]
        assert m == pytest.approx(0.03, abs=1e-10)


@pytest.mark.parametrize("tau", [-0.3, 0.3, 0.6])
def test_calibration_reproduces_pd(tau):
    a = cc.calibrate_alpha_hat_correlated(PD0, PD, RHO, 0.03, tau)
    assert cc.pd_correlated(PD0, RHO, 0.03, a, tau) == pytest.approx(PD, rel=1e-10)


def test_positive_tau_needs_smaller_jump():
    a0 = cc.calibrate_alpha_hat_correlated(PD0, PD, RHO, 0.03, 0.0)
    a3 = cc.calibrate_alpha_hat_correlated(PD0, PD, RHO, 0.03, 0.3)
    assert a3 < a0


def test_correlated_loss_distribution_matches_monte_carlo():
    tau = 0.4
    a = cc.calibrate_alpha_hat_correlated(PD0, PD, RHO, 0.03, tau)
    lgd1 = float(cc.lgd_climate(LGD0, 0.3 * a))
    mc = cc.simulate_losses(PD0, RHO, 0.03, a, LGD0, lgd1, 1.0, n_sims=2_000_000, seed=11, tau=tau)
    for L in (0.003, 0.008, 0.015):
        exact = cc.loss_cdf_correlated(L, PD0, RHO, 0.03, a, LGD0, lgd1, tau)
        assert np.mean(mc < L) == pytest.approx(exact, abs=2e-3)
    var = cc.var_correlated(PD0, RHO, 0.03, a, LGD0, lgd1, tau, Q)
    assert np.quantile(mc, Q) == pytest.approx(var, rel=0.02)
    assert np.mean(mc) == pytest.approx(
        cc.expected_loss_correlated(PD0, RHO, 0.03, a, LGD0, lgd1, tau), rel=0.02)


def test_heterogeneous_var_with_tau_matches_single_loan():
    tau = 0.3
    a = cc.calibrate_alpha_hat_correlated(PD0, PD, RHO, 0.03, tau)
    lgd1 = float(cc.lgd_climate(LGD0, 0.3 * a))
    v1 = cc.var_correlated(PD0, RHO, 0.03, a, LGD0, lgd1, tau, Q)
    v2 = cc.portfolio_var_exact(PD0, RHO, 0.03, a, LGD0, lgd1, 1.0, Q, tau=tau)
    assert v1 == pytest.approx(v2, rel=1e-7)
