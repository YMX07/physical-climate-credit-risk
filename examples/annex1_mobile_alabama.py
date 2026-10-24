"""Reproduce Annex 1 of BIS WP 1274: a BBB loan secured by commercial real estate
in Mobile, Alabama, exposed to Category 3+ hurricanes.

Run from the project root:  python examples/annex1_mobile_alabama.py
"""

from pathlib import Path

import numpy as np

from climate_credit import PhysicalRiskLoan, portfolio_loss_cdf

ROOT = Path(__file__).resolve().parents[1]

PD0 = 0.003            # BBB one-year PD from the bank's internal model
LGD0 = 0.10            # 90% recovery on a secured high-quality loan
ASSET_VOL = 0.30       # implied equity vol, delta ~ 1
Q_LONG_TERM = 0.017    # 1948-2022 annual mean, Category 3, Mobile AL
STORM_SENSITIVITY = 0.161  # +16.1% defaults when storm frequency doubles (De Spiegeleer et al.)

# (label, q, relative increase in hurricane frequency)
CASES = [
    ("2025 forecast q=3.0%", 0.030, 0.76),  # paper rounds 0.03/0.017 - 1 to 76%
    ("95% conf. q=4.8%", 0.048, 0.048 / Q_LONG_TERM - 1.0),
]

# Values reported in the paper, for comparison
PAPER = {
    0.030: {"PD": 0.00337, "alpha_hat": 0.58, "Cv": 0.0747, "Cv-PD": 0.0714, "LGD1": 0.248,
            "uplift_model": 0.079, "uplift_lgd40": 0.127},
    0.048: {"PD": 0.0039, "alpha_hat": 0.72, "Cv-PD": 0.0735, "LGD1": 0.275,
            "uplift_model": 0.155, "uplift_lgd40": 0.22},
}


def main():
    print(__doc__.splitlines()[0])
    print("=" * 78)
    for label, q, rel in CASES:
        pd = PD0 * (1.0 + STORM_SENSITIVITY * rel)
        loan = PhysicalRiskLoan(pd0=PD0, pd=pd, q=q, lgd0=LGD0, asset_vol=ASSET_VOL)
        loan40 = PhysicalRiskLoan(pd0=PD0, pd=pd, q=q, lgd0=LGD0, lgd1=0.40, asset_vol=ASSET_VOL)
        ref = PAPER[q]
        rows = [
            ("rho (Basel, at PD0)", loan.rho, 0.223),
            ("PD with physical risk", pd, ref["PD"]),
            ("alpha_hat (Eq. 6)", loan.alpha_hat, ref["alpha_hat"]),
            ("exp(-alpha)", np.exp(-loan.alpha), None),
            ("Cv(Q) baseline", loan.cv_baseline(), 0.0720),
            ("Cv(Q) climate, Eq. 12", loan.cv(), ref.get("Cv")),
            ("Cv(Q) climate, exact Eq. 9", loan.cv(exact=True), None),
            ("Cv(Q) - PD baseline", loan.cv_baseline() - PD0, 0.0690),
            ("Cv(Q) - PD climate", loan.cv() - pd, ref["Cv-PD"]),
            ("LGD1 model (Eq. 15)", loan.lgd1, ref["LGD1"]),
            ("UL baseline", loan.ul_baseline(), 0.0069),
            ("UL climate (Eq. 22)", loan.ul(), None),
            ("RWA uplift, model LGD1", loan.ul() / loan.ul_baseline() - 1, ref["uplift_model"]),
            ("RWA uplift, LGD1=40%", loan40.ul() / loan40.ul_baseline() - 1, ref["uplift_lgd40"]),
            ("RWA uplift, model LGD1, conditional", loan.ul("conditional") / loan.ul_baseline() - 1, None),
            ("RWA uplift, LGD1=40%, conditional", loan40.ul("conditional") / loan40.ul_baseline() - 1, None),
            ("RWA uplift, model LGD1, exact", loan.ul("exact") / loan.ul_baseline() - 1, None),
            ("RWA uplift, LGD1=40%, exact", loan40.ul("exact") / loan40.ul_baseline() - 1, None),
        ]
        print(f"\n{label}")
        print(f"  {'quantity':34s}{'this code':>12s}{'paper':>12s}")
        for name, ours, paper in rows:
            p = "" if paper is None else f"{paper:12.4f}"
            print(f"  {name:34s}{ours:12.4f}{p}")
        print(f"  RWA per unit EAD, M=2.5: baseline {loan.rwa_baseline():.4f}, "
              f"climate {loan.rwa():.4f} (paper formula), {loan.rwa('exact'):.4f} (exact)")

    plot(PD0)


def plot(pd0):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)
    loss = np.linspace(1e-5, 0.03, 600)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    base = PhysicalRiskLoan(pd0=pd0, pd=pd0 * (1 + 1e-9), q=0.03, lgd0=LGD0)
    ax.plot(loss * 100, 1 - portfolio_loss_cdf(loss, pd0, base.rho, 0.0, 0.0, LGD0, LGD0),
            label="Vasicek, no physical risk", color="#555555")
    for label, q, rel in CASES:
        ln = PhysicalRiskLoan(pd0=pd0, pd=pd0 * (1 + STORM_SENSITIVITY * rel), q=q, lgd0=LGD0)
        ax.plot(loss * 100, 1 - portfolio_loss_cdf(loss, pd0, ln.rho, q, ln.alpha_hat, LGD0, ln.lgd1),
                label=f"with physical risk, {label}")
    ax.axhline(1e-3, ls=":", color="k", lw=0.8)
    ax.text(0.05, 0.75e-3, "1 - Q = 0.1%", ha="left", va="top", fontsize=8)
    ax.set_yscale("log")
    ax.set_xlabel("Portfolio loss, % of EAD")
    ax.set_ylabel("P(Loss > L)")
    ax.set_title("Portfolio loss tail, Eq. (14): BBB loans, Mobile AL")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "annex1_loss_tail.png", dpi=150)
    print(f"\nSaved {out / 'annex1_loss_tail.png'}")


if __name__ == "__main__":
    main()
