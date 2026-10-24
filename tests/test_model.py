import numpy as np
import pytest
from scipy.stats import norm

import climate_credit as cc

PD0, LGD0, Q = 0.003, 0.10, 0.999


def annex_loan(q=0.03, rel=0.76, **kw):
    return cc.PhysicalRiskLoan(pd0=PD0, pd=PD0 * (1 + 0.161 * rel), q=q, lgd0=LGD0, **kw)


# --- Annex 1 numbers ------------------------------------------------------

def test_annex1_base_case():
    loan = annex_loan()
    assert loan.rho == pytest.approx(0.223, abs=5e-4)
    assert loan.pd == pytest.approx(0.00337, abs=5e-6)
    assert loan.alpha_hat == pytest.approx(0.58, abs=0.01)
    assert loan.cv_baseline() == pytest.approx(0.0720, abs=5e-5)
    assert loan.cv() == pytest.approx(0.0747, abs=5e-5)
    assert loan.cv() - loan.pd == pytest.approx(0.0714, abs=5e-5)
    assert loan.ul_baseline() == pytest.approx(0.0069, abs=5e-5)
    assert np.exp(-loan.alpha) == pytest.approx(0.84, abs=0.005)
    assert loan.ul() / loan.ul_baseline() - 1 == pytest.approx(0.079, abs=0.002)


def test_annex1_external_lgd():
    loan = annex_loan(lgd1=0.40)
    assert loan.ul() / loan.ul_baseline() - 1 == pytest.approx(0.127, abs=0.002)


def test_annex1_95pct_case():
    loan = annex_loan(q=0.048, rel=0.048 / 0.017 - 1)
    assert loan.pd == pytest.approx(0.0039, abs=5e-5)
    assert loan.alpha_hat == pytest.approx(0.72, abs=0.01)
    assert loan.cv() - loan.pd == pytest.approx(0.0735, abs=1e-4)
    assert loan.ul() / loan.ul_baseline() - 1 == pytest.approx(0.155, abs=0.003)
    loan40 = annex_loan(q=0.048, rel=0.048 / 0.017 - 1, lgd1=0.40)
    assert loan40.ul() / loan40.ul_baseline() - 1 == pytest.approx(0.22, abs=0.005)


# --- Model identities -----------------------------------------------------

def test_q_deformed_inverse_and_self_consistency():
    a = cc.calibrate_alpha_hat(PD0, 0.0035, 0.03)
    assert cc.q_deformed_ppf(0.0035, 0.03, a) == pytest.approx(norm.ppf(PD0), abs=1e-10)
    assert cc.pd_from_alpha_hat(PD0, 0.03, a) == pytest.approx(0.0035, rel=1e-12)


@pytest.mark.parametrize("q,a", [(0.0, 0.7), (0.03, 0.0)])
def test_reduces_to_vasicek(q, a):
    rho = cc.basel_correlation(PD0)
    lgd1 = cc.lgd_climate(LGD0, 0.3 * a)
    assert cc.cv_climate(PD0, rho, q, a, Q) == pytest.approx(cc.vasicek_conditional_pd(PD0, rho, Q))
    var = cc.portfolio_var(PD0, rho, q, a, LGD0, lgd1, Q)
    assert var == pytest.approx(LGD0 * cc.vasicek_conditional_pd(PD0, rho, Q), rel=1e-9)


def test_loss_cdf_matches_monte_carlo():
    loan = annex_loan()
    losses = cc.simulate_losses(PD0, loan.rho, loan.q, loan.alpha_hat, LGD0, loan.lgd1, 1.0,
                                n_sims=400_000, seed=1)
    for L in (0.002, 0.005, 0.01):
        exact = cc.portfolio_loss_cdf(L, PD0, loan.rho, loan.q, loan.alpha_hat, LGD0, loan.lgd1)
        assert np.mean(losses < L) == pytest.approx(exact, abs=3e-3)
    assert np.mean(losses) == pytest.approx(loan.expected_loss(), rel=0.02)


def test_first_order_close_to_exact_for_small_q():
    # Eq. (22) is a perturbation result: it needs q << 1 - Q and q * alpha_hat small
    pd = float(cc.pd_from_alpha_hat(PD0, 0.0005, 0.1))
    loan = cc.PhysicalRiskLoan(pd0=PD0, pd=pd, q=0.0005, lgd0=LGD0)
    assert loan.ul("paper") == pytest.approx(loan.ul("exact"), rel=1e-3)


def test_annex3_exact_slope_and_small_alpha_breakdown():
    rho = float(cc.basel_correlation(PD0))
    lv = float(cc.vasicek_conditional_pd(PD0, rho, Q))
    w = lambda th, s: norm.cdf((np.sqrt(1 - rho) * norm.ppf(th) - norm.ppf(PD0) - s) / np.sqrt(rho))
    dw0 = (w(lv + 1e-7, 0) - w(lv - 1e-7, 0)) / 2e-7
    a, h = 0.584, 1e-6
    lgd1 = float(cc.lgd_climate(LGD0, 0.3 * a))
    fd = (cc.portfolio_var(PD0, rho, h, a, LGD0, lgd1, Q) - LGD0 * lv) / h
    assert fd == pytest.approx(LGD0 * (Q - w(lv * LGD0 / lgd1, a)) / dw0, rel=1e-3)
    eq20 = (cc.portfolio_var_first_order(PD0, rho, h, a, LGD0, lgd1, Q) - LGD0 * lv) / h
    assert fd / eq20 > 10


def test_first_order_understates_when_q_exceeds_tail():
    loan = annex_loan()
    mc = cc.simulate_losses(PD0, loan.rho, loan.q, loan.alpha_hat, LGD0, loan.lgd1, 1.0,
                            n_sims=2_000_000, seed=3)
    exact = cc.portfolio_var(PD0, loan.rho, loan.q, loan.alpha_hat, LGD0, loan.lgd1, Q)
    assert np.quantile(mc, Q) == pytest.approx(exact, rel=0.02)
    assert exact > 2 * cc.portfolio_var_first_order(PD0, loan.rho, loan.q, loan.alpha_hat,
                                                    LGD0, loan.lgd1, Q)


def test_conditional_capital_is_small_share_limit():
    loan = annex_loan()
    w = 1e-4
    el = w * loan.expected_loss() + (1 - w) * LGD0 * PD0
    ul = cc.portfolio_var_exact([PD0, PD0], loan.rho, loan.q, [loan.alpha_hat, 0.0], LGD0,
                                [loan.lgd1, LGD0], [w, 1 - w], Q) - el
    marginal = (ul - (1 - w) * loan.ul_baseline()) / w
    assert marginal == pytest.approx(loan.ul("conditional"), rel=0.01)
    assert loan.ul("paper") < loan.ul("conditional") < loan.ul("exact")


def test_heterogeneous_var_single_loan_matches_closed_form():
    loan = annex_loan()
    v1 = cc.portfolio_var(PD0, loan.rho, loan.q, loan.alpha_hat, LGD0, loan.lgd1, Q)
    v2 = cc.portfolio_var_exact(PD0, loan.rho, loan.q, loan.alpha_hat, LGD0, loan.lgd1, 1.0, Q)
    assert v1 == pytest.approx(v2, rel=1e-8)


# --- Extensions -----------------------------------------------------------

def test_multistate_binary_equals_two_channel():
    loan = annex_loan()
    st = cc.ClimateStates.binary(loan.q, loan.alpha_hat, LGD0, loan.lgd1)
    assert cc.multistate_var(PD0, loan.rho, st, Q) == pytest.approx(
        cc.portfolio_var(PD0, loan.rho, loan.q, loan.alpha_hat, LGD0, loan.lgd1, Q), rel=1e-10)
    assert cc.multistate_pd(PD0, st) == pytest.approx(loan.pd, rel=1e-10)


def test_transition_calibration_round_trip():
    q, a, p, d = 0.03, 0.6, np.array([0.80, 0.15, 0.05]), np.array([0.3, 1.0])
    c = norm.ppf(PD0)
    pd_phys = cc.pd_from_alpha_hat(PD0, q, a)
    pd_tr = cc.mixture_cdf(c, p, np.r_[0, d])
    joint = cc.joint_states(q, a, p, d, LGD0, 0.3)
    pd_joint = cc.multistate_pd(PD0, joint)
    res = cc.calibrate_transition(PD0, pd_phys, pd_tr, pd_joint, q, d)
    assert res["alpha_hat"] == pytest.approx(a, abs=1e-9)
    np.testing.assert_allclose(res["transition_probs"], p, atol=1e-6)


def test_perfect_hedge_restores_baseline_multiplier():
    assert cc.hedged_multiplier(0.03, 0.17, LGD0, hedge_ratio=1.0) == 1.0
    assert cc.swap_spread(0.03, 0.17) == pytest.approx(0.03 * (1 - np.exp(-0.17)))
