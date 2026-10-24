"""How capital changes when the climate event is correlated with the economy.

Annex 1 loan (BBB, Mobile AL). tau is the Gaussian-copula correlation between the
event and the economic stress factor (tau > 0: hurricanes more likely in downturns).

Run from the project root:  python examples/correlated_climate.py
"""

from pathlib import Path

import numpy as np

import climate_credit as cc

ROOT = Path(__file__).resolve().parents[1]

PD0, LGD0, Q = 0.003, 0.10, 0.999
CASES = [(0.030, 0.76), (0.048, 0.048 / 0.017 - 1)]
TAUS = [-0.2, 0.0, 0.1, 0.2, 0.3, 0.5]


def table(q, rel, recalibrate):
    pd = PD0 * (1 + 0.161 * rel)
    paper = cc.PhysicalRiskLoan(pd0=PD0, pd=pd, q=q, lgd0=LGD0)
    mode = "bank PD held fixed, alpha_hat re-solved" if recalibrate else "alpha_hat held at tau=0 value, PD rises"
    print(f"\nq = {q:.1%}, {mode}   (paper Eq. 22: {paper.ul() / paper.ul_baseline() - 1:+.1%})")
    print(f"  {'tau':>5s}{'alpha_hat':>10s}{'PD':>9s}{'LGD1':>7s}{'q|yQ':>8s}"
          f"{'diversified':>13s}{'concentrated':>14s}")
    rows = []
    for tau in TAUS:
        r = cc.capital_correlated(PD0, pd, q, LGD0, tau, recalibrate=recalibrate)
        b = r["ul_baseline"]
        rows.append((tau, r["ul_conditional"] / b - 1, r["ul_exact"] / b - 1))
        print(f"  {tau:5.1f}{r['alpha_hat']:10.3f}{r['pd']:9.4%}{r['lgd1']:7.1%}{r['q_given_yQ']:8.1%}"
              f"{r['ul_conditional'] / b - 1:+13.1%}{r['ul_exact'] / b - 1:+14.1%}")
    return rows


def concentration(q, rel, tau):
    pd = PD0 * (1 + 0.161 * rel)
    r = cc.capital_correlated(PD0, pd, q, LGD0, tau)
    a, lgd1, base = r["alpha_hat"], r["lgd1"], r["ul_baseline"]
    rho = float(cc.basel_correlation(PD0))
    out = []
    for w in (0.01, 0.1, 0.5):
        W = np.array([w, 1 - w])
        el = w * r["expected_loss"] + (1 - w) * LGD0 * PD0
        ul = cc.portfolio_var_exact([PD0, PD0], rho, q, [a, 0.0], LGD0, [lgd1, LGD0], W, Q, tau=tau) - el
        out.append((ul - (1 - w) * base) / w / base - 1)
    return out


def mc_check():
    tau, q = 0.3, 0.03
    pd = PD0 * (1 + 0.161 * 0.76)
    r = cc.capital_correlated(PD0, pd, q, LGD0, tau)
    rho = float(cc.basel_correlation(PD0))
    mc = cc.simulate_losses(PD0, rho, q, r["alpha_hat"], LGD0, r["lgd1"], 1.0,
                            n_sims=4_000_000, seed=5, tau=tau)
    var = cc.var_correlated(PD0, rho, q, r["alpha_hat"], LGD0, r["lgd1"], tau, Q)
    print(f"\nMonte Carlo check, q=3%, tau=0.3: 99.9% loss closed form {var:.5f}, "
          f"MC (4m sims) {np.quantile(mc, Q):.5f}")


def plot(series):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for (label, rows), color in zip(series, ["#1f77b4", "#ff7f0e"]):
        t = [r[0] for r in rows]
        ax.plot(t, [100 * r[1] for r in rows], "-o", color=color, label=f"{label}, diversified")
        ax.plot(t, [100 * r[2] for r in rows], "--s", color=color, label=f"{label}, concentrated")
    ax.axvline(0, color="k", lw=0.6, ls=":")
    ax.set_yscale("log")
    ax.set_xlabel("tau: correlation of climate event with economic stress")
    ax.set_ylabel("Climate add-on to UL / RWA, % (log)")
    ax.set_title("Annex 1 loan: capital vs climate-economy correlation (bank PD fixed)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "correlated_capital.png", dpi=150)
    print(f"Saved {out / 'correlated_capital.png'}")


def main():
    series = []
    for q, rel in CASES:
        series.append((f"q={q:.1%}", table(q, rel, recalibrate=True)))
    table(0.03, 0.76, recalibrate=False)
    print("\nAdd-on per exposed loan by share of portfolio exposed, q = 3%:")
    print(f"  {'tau':>5s}{'1%':>9s}{'10%':>9s}{'50%':>9s}")
    for tau in (0.0, 0.3):
        print(f"  {tau:5.1f}" + "".join(f"{x:+9.1%}" for x in concentration(0.03, 0.76, tau)))
    mc_check()
    plot(series)


if __name__ == "__main__":
    main()
