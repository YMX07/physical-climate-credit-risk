"""Checks beyond the paper: (1) Monte Carlo confirmation of the exact 99.9% VaR,
(2) where the first-order formula (22) departs from the exact loss distribution (14),
(3) portfolio invariance for a heterogeneous portfolio.

Run from the project root:  python examples/diagnostics.py
"""

import numpy as np

import climate_credit as cc

PD0, LGD0, Q = 0.003, 0.10, 0.999


def mc_check():
    print("1) Monte Carlo vs closed form, Annex 1 base case (q=3%)")
    loan = cc.PhysicalRiskLoan(pd0=PD0, pd=PD0 * (1 + 0.161 * 0.76), q=0.03, lgd0=LGD0)
    args = (PD0, loan.rho, loan.q, loan.alpha_hat, LGD0, loan.lgd1)
    asrf = cc.simulate_losses(*args, 1.0, n_sims=4_000_000, seed=7)
    gran = cc.simulate_losses(*args, 1.0, n_sims=1_000_000, n_obligors=5_000, seed=8)
    print(f"   exact VaR (Eq. 14)          {cc.portfolio_var(*args, Q):.5f}")
    print(f"   first-order VaR (Eq. 20)    {cc.portfolio_var_first_order(*args, Q):.5f}")
    print(f"   MC ASRF limit, 4m sims      {np.quantile(asrf, Q):.5f}")
    print(f"   MC 5,000 obligors, 1m sims  {np.quantile(gran, Q):.5f}")
    print(f"   Vasicek VaR, no climate     {LGD0 * cc.vasicek_conditional_pd(PD0, loan.rho, Q):.5f}\n")


def approximation_sweep():
    print("2) RWA uplift, paper Eq. (22) vs exact, PD/PD0 fixed at 1.1224")
    print(f"   {'q':>7s}{'alpha_hat':>11s}{'paper':>9s}{'exact':>9s}")
    for q in (0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.03, 0.05):
        try:
            ln = cc.PhysicalRiskLoan(pd0=PD0, pd=PD0 * 1.1224, q=q, lgd0=LGD0)
        except ValueError:
            continue
        b = ln.ul_baseline()
        print(f"   {q:7.4f}{ln.alpha_hat:11.3f}{ln.ul() / b - 1:9.1%}{ln.ul('exact') / b - 1:9.1%}")
    print("   (1 - Q = 0.1%: once q exceeds it, the event channel sets the 99.9% quantile)\n")
    print("   Small q and small alpha_hat together (q=0.0005, alpha_hat=0.1):")
    pd = float(cc.pd_from_alpha_hat(PD0, 0.0005, 0.1))
    ln = cc.PhysicalRiskLoan(pd0=PD0, pd=pd, q=0.0005, lgd0=LGD0)
    print(f"   UL paper {ln.ul():.6f}  exact {ln.ul('exact'):.6f}\n")


def invariance():
    print("3) Portfolio invariance, 3 loan classes, common hurricane event q=3%")
    pd0 = np.array([0.003, 0.01, 0.03])
    pd = pd0 * np.array([1.12, 1.05, 1.0 + 1e-9])  # last class unexposed
    w = np.array([0.5, 0.3, 0.2])
    loans = [cc.PhysicalRiskLoan(pd0=a, pd=b, q=0.03, lgd0=0.25) for a, b in zip(pd0, pd)]
    rho = np.array([l.rho for l in loans])
    ah = np.array([l.alpha_hat for l in loans])
    lgd1 = np.array([l.lgd1 for l in loans])
    el = sum(wi * l.expected_loss() for wi, l in zip(w, loans))
    port = cc.portfolio_var_exact(pd0, rho, 0.03, ah, 0.25, lgd1, w, Q) - el
    paper_sum = sum(wi * l.ul("paper") for wi, l in zip(w, loans))
    exact_sum = sum(wi * l.ul("exact") for wi, l in zip(w, loans))
    print(f"   sum of per-loan UL, paper Eq. (22)   {paper_sum:.5f}")
    print(f"   sum of per-loan UL, exact            {exact_sum:.5f}")
    print(f"   portfolio UL, exact                  {port:.5f}")


def concentration():
    print("\n4) Climate add-on per exposed loan vs share of the portfolio in the hazard zone")
    print("   (Annex 1 loans in Mobile, the rest identical but unexposed)")
    for q, rel in ((0.03, 0.76), (0.048, 0.048 / 0.017 - 1)):
        ln = cc.PhysicalRiskLoan(pd0=PD0, pd=PD0 * (1 + 0.161 * rel), q=q, lgd0=LGD0)
        base = ln.ul_baseline()
        print(f"   q={q:.1%}: paper {ln.ul() / base - 1:+.1%}, "
              f"conditional (share -> 0) {ln.ul('conditional') / base - 1:+.1%}")
        for w in (0.01, 0.05, 0.1, 0.25, 0.5, 1.0):
            k = 1 if w == 1.0 else 2
            W = np.array([w, 1 - w])[:k]
            el = w * ln.expected_loss() + (1 - w) * LGD0 * PD0
            ul = cc.portfolio_var_exact(np.full(k, PD0), np.full(k, ln.rho), q,
                                        np.array([ln.alpha_hat, 0.0])[:k], LGD0,
                                        np.array([ln.lgd1, LGD0])[:k], W, Q) - el
            print(f"     share {w:5.0%}: exact {(ul - (1 - w) * base) / w / base - 1:+.1%}")


def expected_loss_check():
    print("\n5) Expected loss, Annex 1 base case")
    ln = cc.PhysicalRiskLoan(pd0=PD0, pd=PD0 * (1 + 0.161 * 0.76), q=0.03, lgd0=LGD0)
    base = LGD0 * ln.pd
    ex, fo = ln.expected_loss(), ln.expected_loss(exact=False)
    print(f"   exact {ex:.6f} (climate add-on {ex - base:.2e}), "
          f"Eq. (17) {fo:.6f} (add-on {fo - base:.2e}): "
          f"Eq. (17) understates the add-on {(ex - base) / (fo - base):.1f}x, total EL {1 - fo / ex:.0%}")


def transition_sensitivity():
    print("\n6) Transition calibration (Eq. 25): true p = [0.80, 0.15, 0.05]")
    from scipy.stats import norm
    q, a, p, d = 0.03, 0.6, np.array([0.80, 0.15, 0.05]), np.array([0.3, 1.0])
    pd_p = cc.pd_from_alpha_hat(PD0, q, a)
    pd_t = cc.mixture_cdf(norm.ppf(PD0), p, np.r_[0, d])
    pd_j = cc.multistate_pd(PD0, cc.joint_states(q, a, p, d, LGD0, 0.3))
    for err in (0.0, 1e-4, 1e-3):
        r = cc.calibrate_transition(PD0, pd_p, pd_t, pd_j * (1 + err), q, d)
        print(f"   joint PD error {err:.2%}: p = {np.round(r['transition_probs'], 3)}"
              f"  (condition number {r['condition_number']:.0f})")


def annex3_slope():
    print("\n7) Where Annex 3 breaks: d VaR / dq at q -> 0, alpha_hat fixed")
    from scipy.stats import norm
    rho = float(cc.basel_correlation(PD0))
    lv = float(cc.vasicek_conditional_pd(PD0, rho, Q))
    w = lambda th, s: norm.cdf((np.sqrt(1 - rho) * norm.ppf(th) - norm.ppf(PD0) - s) / np.sqrt(rho))
    dw0 = (w(lv + 1e-7, 0) - w(lv - 1e-7, 0)) / 2e-7
    h = 1e-6
    print(f"   {'alpha_hat':>9s}{'finite diff':>13s}{'Annex 3 exact':>15s}{'Eq. 20':>9s}{'ratio':>7s}")
    for a in (0.01, 0.05, 0.1, 0.3, 0.584):
        lgd1 = float(cc.lgd_climate(LGD0, 0.3 * a))
        fd = (cc.portfolio_var(PD0, rho, h, a, LGD0, lgd1, Q) - LGD0 * lv) / h
        exact = LGD0 * (Q - w(lv * LGD0 / lgd1, a)) / dw0
        eq20 = (cc.portfolio_var_first_order(PD0, rho, h, a, LGD0, lgd1, Q) - LGD0 * lv) / h
        print(f"   {a:9.3f}{fd:13.5f}{exact:15.5f}{eq20:9.5f}{fd / eq20:7.2f}")


def storm_rule_sensitivity():
    print("\n8) Sensitivity to the 16.1% storm rule: PD/PD0 = 1 + s (q/0.017 - 1)")
    print(f"   {'s':>6s}{'q':>7s}{'PD/PD0':>8s}{'alpha_hat':>10s}{'paper':>8s}{'cond.':>8s}{'exact':>9s}")
    for s in (0.08, 0.161, 0.24):
        for q in (0.03, 0.048):
            ln = cc.PhysicalRiskLoan(pd0=PD0, pd=PD0 * (1 + s * (q / 0.017 - 1)), q=q, lgd0=LGD0)
            b = ln.ul_baseline()
            print(f"   {s:6.3f}{q:7.1%}{ln.pd / PD0:8.4f}{ln.alpha_hat:10.3f}{ln.ul() / b - 1:+8.1%}"
                  f"{ln.ul('conditional') / b - 1:+8.1%}{ln.ul('exact') / b - 1:+9.1%}")


if __name__ == "__main__":
    mc_check()
    approximation_sweep()
    invariance()
    concentration()
    expected_loss_check()
    transition_sensitivity()
    annex3_slope()
    storm_rule_sensitivity()
