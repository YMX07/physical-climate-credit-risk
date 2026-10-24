# Physical climate risk in the Vasicek / IRB credit model

A Python implementation of

> Pozdyshev, V., Lobanov, A. and Ilinsky, K. (2025), *Incorporating physical climate risks into banks' credit risk models*, BIS Working Paper No 1274. ([bis.org](https://www.bis.org/publications/working-paper-1274-incorporating-physical-climate-risks-banks-credit-risk-models))

This is an independent replication. It is not affiliated with or endorsed by the BIS or the authors. The paper is
© BIS and is not included here; download it from the link above.

This README walks through the paper in order. Each section gives the idea, the key equations
(numbered as in the paper) and the function that implements them. At the end, it reproduces the paper's
worked example (Annex 1), reports what testing the model found (section 13), relaxes the paper's independence assumption (section 14) and lists
open limitations (section 15).

**Summary.** The code reproduces every number in the paper. The paper's closed-form capital formula (Eq. 22), though,
understates the capital implied by its own model at the paper's parameters. The error enters in Annex 3, where the paper
simplifies for "small α" (section 13). For the Annex 1 loan, the climate add-on depends on the share of the portfolio in
the hazard zone:

| Share of portfolio exposed | q = 3.0% | q = 4.8% |
|---|---|---|
| Paper, Eq. (22) | +7.9% | +15.5% |
| Diversified limit (exposed share → 0) | +18% | +41% |
| 1% exposed | +19% | +46% |
| 10% exposed | +35% | +101% |
| Whole portfolio exposed | +147% | +345% |

These figures assume, like the paper, that hurricanes are independent of the economic cycle. Section 14 relaxes that,
using a correlation τ that the bank's PDs cannot identify, so it is a sensitivity analysis.

```bash
pip install -e ".[dev]"                    # or: pip install git+https://github.com/YMX07/physical-climate-credit-risk
python examples/annex1_mobile_alabama.py   # reproduces Annex 1, plots outputs/annex1_loss_tail.png
python examples/diagnostics.py             # Monte Carlo checks and approximation sweep
python examples/correlated_climate.py      # capital when the event is correlated with the economy
python -m pytest -q tests                  # 24 tests
```

---

## 1. The problem (paper Sections 1–3)

Regulators (BCBS, ECB, Bank of England) expect banks to bring physical climate risk into their credit risk
and capital, but there is no accepted model for doing so. Most banks' internal models and the Basel IRB formula rest
on the **one-factor Vasicek model** (asymptotic single risk factor, ASRF). The paper asks where
physical risk should enter that model. It considers four options:

1. **A higher confidence level Q.** This is rejected as arbitrary: it raises capital without measuring anything.
2. **A higher PD.** This alone is not enough. IRB PDs are long-run averages, and rare disasters barely move the average.
3. **The correlation ρ.** This is rejected. Basel's correlation falls as PD rises, and a borrower's hazard exposure
   (location, assets, industry) has nothing to do with that.
4. **A new, independent systematic factor.** This is the paper's choice.

The new factor should keep **portfolio invariance**: a loan's capital charge should not depend on
the rest of the portfolio, so that charges can be added loan by loan as in IRB. The paper also assumes the climate
factor is uncorrelated with the economic factor, citing the low correlation between CAT bonds and market risk factors.

## 2. Starting point: Vasicek and the IRB formula (Section 4, Eq. 1–3)

The IRB risk-weighted asset formula is

$$
\text{RWA} = \Big\{ LGD\cdot\Phi\Big(\tfrac{\Phi^{-1}(P_D)+\sqrt{\rho}\,\Phi^{-1}(Q)}{\sqrt{1-\rho}}\Big) - LGD\cdot P_D \Big\}\cdot MA\cdot EAD\cdot 12.5 \qquad (1)
$$

The formula is a scaled difference between VaR and expected loss, $\text{RWA}=\text{Multiples}\cdot(\text{VaR}(Q)-\langle EL\rangle)$ (Eq. 2).
For an infinitely granular portfolio, the VaR is the PD conditional on the systematic factor being at its Q-quantile:

$$
\text{RWA} = \text{Multiples}\cdot LGD\cdot\big(C_V(Q) - P_D\big) \qquad (3)
$$

The paper keeps Basel's correlation as a fixed input:

$$
\rho = 0.12\,\tfrac{1-e^{-50 P_D}}{1-e^{-50}} + 0.24\Big(1-\tfrac{1-e^{-50P_D}}{1-e^{-50}}\Big),
\qquad MA = \tfrac{1+(M-2.5)b}{1-1.5b}
$$

**Code:** [basel.py](climate_credit/basel.py): `basel_correlation`, `maturity_adjustment`,
`vasicek_conditional_pd` (that is, $C_V(Q)$), `irb_capital`, `irb_rwa`.

## 3. Adding a physical-risk jump to the asset process (Eq. 4–5)

In the Merton/Vasicek setup, a borrower's assets follow a geometric Brownian motion driven by a systematic factor $S$ and an
idiosyncratic factor $B$. The paper adds a **one-off downward jump**:

$$
\hat V_t = V_t\,e^{-\alpha}\,\xi + (1-\xi)\,V_t,\qquad \xi\in\{0,1\},\ P(\xi=1)=q
$$

- $q$ is the probability of a climate event above a "normal" threshold over the horizon, for example a Category 3+
  hurricane. It is **external**, supplied by a meteorological model, and **the same for all obligors**.
- $e^{-\alpha}<1$ is the share of asset value left after the event. $\alpha$ depends on the borrower.
- Because $V_t$ is a GBM, the timing of the jump does not matter. It can be treated as happening at maturity.

Default happens when $\hat V_T < L$. Normalising the asset return gives

$$
P_D = (1-q)\,\Phi(C^*) + q\,\Phi(C^*+\hat\alpha),\qquad
C^* = \frac{\ln\frac{L}{V_0}-\mu+\frac12(\sigma_V^2+\beta_V^2)}{\sqrt{\sigma_V^2+\beta_V^2}},\quad
\hat\alpha = \frac{\alpha}{\sqrt{\sigma_V^2+\beta_V^2}} \qquad (4)
$$

The paper calls this the **q-deformed normal distribution**. It is a two-component normal mixture (equal variances,
shifted means), not a q-deformation in the Tsallis or quantum-group sense:

$$
\Phi_{q,\alpha}(x) \equiv (1-q)\,\Phi(x) + q\,\Phi(x+\alpha) \qquad (5)
$$

so that $P_D = \Phi_{q,\hat\alpha}(C^*)$. Without climate risk ($q=0$ or $\alpha=0$), this is the usual Vasicek $P_D^0=\Phi(C^*)$.

**Code:** [model.py](climate_credit/model.py): `q_deformed_cdf`, `q_deformed_ppf`, and the general
`mixture_cdf` / `mixture_ppf` used by the multi-state extension.

## 4. Calibration by self-consistency (Eq. 6)

The model has unobservable parameters, $C^*$ and $\hat\alpha$. The paper does not estimate them directly. It ties them to the
**bank's own internal PDs**: $P_D^0$ (without physical risk) and $P_D$ (with physical risk). Both must imply the same default threshold $C^*$:

$$
\Phi^{-1}(P_D^0) = \Phi^{-1}_{q,\hat\alpha}(P_D) \qquad (6)
$$

Once $q$ is given, the two PDs fix $\hat\alpha$, so no separate calibration is needed. Eq. (6) can be solved in closed form:

$$
\hat\alpha = \Phi^{-1}\!\Big(\frac{P_D-(1-q)P_D^0}{q}\Big) - \Phi^{-1}(P_D^0),
\qquad \text{valid for } P_D^0 \le P_D < (1-q)P_D^0+q .
$$

To get the asset jump itself, multiply by asset volatility: $\alpha=\hat\alpha\cdot\sigma_A$ (Annex 1).

**Code:** `calibrate_alpha_hat`, `pd_from_alpha_hat`, `alpha_from_alpha_hat`.

## 5. PD conditional on the economic factor (Eq. 7–12)

Define $\sqrt\rho = \sigma_V/\sqrt{\sigma_V^2+\beta_V^2}$ and condition on the economic factor $S=-y$.

- No event (Eq. 7, standard Vasicek): $\;P(D\mid y,\xi=0)=\Phi\Big(\frac{\sqrt\rho\,y+\Phi^{-1}(P_D^0)}{\sqrt{1-\rho}}\Big)$
- Event (Eq. 8): $\;P(D\mid y,\xi=1)=\Phi\Big(\frac{\sqrt\rho\,y+\Phi^{-1}(P_D^0)+\hat\alpha}{\sqrt{1-\rho}}\Big)$
- Averaged over the event (Eq. 9): $(1-q)\times$(7) $+\ q\times$(8), which is again a q-deformed normal.

For small $\alpha$ and first order in $q$ (Eq. 10–11), the paper uses

$$
C_V(Q) = \Phi(z_Q) + \frac{q\,\hat\alpha}{\sqrt{2\pi(1-\rho)}}\,e^{-z_Q^2/2},
\qquad z_Q=\frac{\sqrt\rho\,\Phi^{-1}(Q)+\Phi^{-1}(P_D^0)}{\sqrt{1-\rho}} \qquad (12)
$$

This is the climate-adjusted conditional PD that enters the capital formula. It is also useful on its own for stress scenarios.

**Code:** `conditional_pd` (exact Eq. 9), `conditional_pd_first_order` (Eq. 11), and
`cv_climate(..., exact=False)` for Eq. 12 (`exact=True` evaluates Eq. 9 at $y=\Phi^{-1}(Q)$).

## 6. Portfolio loss distribution (Eq. 13–14, Annex 2)

LGD can now differ by channel: $LGD_0$ without the event and $LGD_1$ with it, which captures damage to recoverable assets.
Each channel is a Vasicek-type portfolio, and the paper applies Vasicek's 1987 argument to each one: as $N\to\infty$, the binomial
sum collapses to a step function. The result is the main formula of the paper:

$$
P(\text{Loss}<L) = q\,\Phi\Big(\frac{\sqrt{1-\rho}\,\Phi^{-1}(\Theta_1)-\Phi^{-1}(P_D^0)}{\sqrt\rho}-\frac{\hat\alpha}{\sqrt\rho}\Big)
+(1-q)\,\Phi\Big(\frac{\sqrt{1-\rho}\,\Phi^{-1}(\Theta_0)-\Phi^{-1}(P_D^0)}{\sqrt\rho}\Big) \qquad (14)
$$

Here $\Theta_0=L/LGD_0$ and $\Theta_1=L/LGD_1=\Theta_0\,LGD_0/LGD_1$ are the default rates that produce loss $L$ in each
channel. The only inputs are the bank's PDs and the external $q$. The formula reduces to Vasicek when $q=0$ or $\alpha=0$.

The climate event is a **systematic** factor: one draw of $\xi$ hits every exposed borrower, which is why it creates
co-dependence across the portfolio.

**Code:** `portfolio_loss_cdf` (Eq. 14), `portfolio_var` (its exact Q-quantile).
[portfolio.py](climate_credit/portfolio.py) extends this to heterogeneous portfolios (`portfolio_var_exact`) and
Monte Carlo (`simulate_losses`, either in the ASRF limit or with a finite number of obligors).

## 7. LGD adjusted for physical risk (Eq. 15–16)

Following Basel's preference for conservative LGDs, the paper takes the most conservative case. The jump happens just before maturity,
when assets are only just above the debt level $L$. Then $L-V_T=LGD_0\,L$ and $L-V_Te^{-\alpha}=LGD_1\,L$, which gives

$$
LGD_1 = LGD_0 + (1-e^{-\alpha})(1-LGD_0) \qquad (15)
$$

The second term is the recoverable value destroyed by the event. For small $\alpha$:
$LGD_0/LGD_1 \approx 1-\alpha\,(1-LGD_0)/LGD_0$ (Eq. 16). $LGD_1$ can also be supplied externally; Annex 1 uses 40%.

**Code:** `lgd_climate`. `PhysicalRiskLoan(lgd1=...)` overrides the model-implied value.

## 8. Expected loss (Eq. 17)

$$
\langle EL\rangle = LGD_0(1-q)\Phi(C^*) + LGD_1\,q\,\Phi(C^*+\hat\alpha)
= LGD_0 P_D + q(1-e^{-\alpha})(1-LGD_0)\,\Phi_{\hat\alpha}\big(\Phi^{-1}_{q,\hat\alpha}(P_D)\big)
$$

The first term is the standard expected loss using the climate-adjusted PD. The second is the exposure to the climate event. To lowest order:

$$
\langle EL\rangle \approx LGD_0\,P_D\Big(1+(1-e^{-\alpha})\,q\,\tfrac{1-LGD_0}{LGD_0}\Big) \qquad (17)
$$

**Code:** `expected_loss` (exact), `expected_loss_first_order` (Eq. 17, written with $LGD_1$).

## 9. Value at Risk (Eq. 18–20, Annex 3)

Setting Eq. (14) equal to Q and solving for $L^*_Q$ by perturbation in $q$ gives, to first order (Eq. 18):

$$
L_Q^* = LGD_0\,L_{Vas} + \alpha q(1-LGD_0)L_{Vas} + \frac{\hat\alpha\,q}{\sqrt{2\pi(1-\rho)}}e^{-\frac12(\Phi^{-1}(L_{Vas}))^2}LGD_0,
\qquad L_{Vas}=\Phi\Big(\tfrac{\sqrt\rho\,\Phi^{-1}(Q)+\Phi^{-1}(P_D^0)}{\sqrt{1-\rho}}\Big)
$$

The first and third terms combine into $LGD_0\,C_V(Q)$ from Eq. (12), which gives

$$
L_Q^* = C_V(Q)\cdot LGD_0\cdot\Big(1+(1-e^{-\alpha})\,q\,\tfrac{1-LGD_0}{LGD_0}\Big) \qquad (20)
$$

**Code:** `portfolio_var_first_order` (Eq. 20 in its $LGD_1$ form), and `portfolio_var` for the exact root of Eq. 14.

## 10. The generalised ASRF formula (Eq. 21–22)

Putting (17) and (20) into $\text{RWA}=\text{Multiples}\cdot(L^*_Q-\langle EL\rangle)$ gives the paper's result:

$$
\text{RWA} = \text{Multiples}\cdot LGD_0\cdot[C_V(Q)-P_D]\cdot\Big(1+q\,\tfrac{LGD_1-LGD_0}{LGD_0}\Big) \qquad (22)
$$

This has **the same structure as the IRB formula (3)**, but it is not the same formula. Two things change, and the second
is a separate add-on that today's IRB engines do not compute:
- $C_V(Q)$ is the climate-adjusted conditional PD of Eq. (12), and $P_D$ is the climate-adjusted PD;
- an extra multiplier $1+q(LGD_1-LGD_0)/LGD_0$ reserves for climate damage to recoverable assets.

The formula is computed loan by loan, so it is portfolio invariant by construction.

To use it, a bank needs (paper Section 6):
1. an internal model for $P_D^0$;
2. an internal or third-party model for $P_D$ including climate damage;
3. an external meteorological estimate of $q$;
4. its usual $LGD_0$;
5. a conservative $LGD_1$.

**Code:** `climate_multiplier`, `capital_generalised` (Eq. 22), `capital_conditional` and `capital_exact`
(the two exact measures from section 13),
and the `PhysicalRiskLoan` wrapper:

```python
from climate_credit import PhysicalRiskLoan

loan = PhysicalRiskLoan(pd0=0.003, pd=0.003367, q=0.03, lgd0=0.10, asset_vol=0.30)
loan.alpha_hat       # 0.584, from Eq. (6)
loan.lgd1            # 0.245, from Eq. (15)
loan.cv()            # 0.0747, Eq. (12)
loan.ul()            # UL per unit EAD, Eq. (22)
loan.ul("conditional")  # exact, loan is a small share of the exposed portfolio (section 13)
loan.ul("exact")     # exact VaR(14) - exact EL, whole portfolio exposed
loan.rwa()           # x MA x 12.5 x EAD (rho and MA computed from PD0 by default)
loan.summary()       # all of the above as a dict
```

## 11. Extensions (Section 5)

### Multi-state climate model (Eq. 23)

The binary event generalises to states $\{q_i,\alpha_i\}_{i=0}^{n-1}$, for example hurricane categories:

$$
P(\text{Loss}<L)=\sum_i q_i\,\Phi\Big(\frac{\sqrt{1-\rho}\,\Phi^{-1}(L/LGD_i)-\Phi^{-1}_{\{q,\hat\alpha\}}(P_D)}{\sqrt\rho}-\frac{\hat\alpha_i}{\sqrt\rho}\Big),
\qquad \Phi_{\{q,\hat\alpha\}}(x)=\sum_i q_i\,\Phi(x+\hat\alpha_i)
$$

The paper does not pursue this, to keep the parameter count low enough for regulatory use, but notes that banks could use it in internal models.

**Code:** [extensions.py](climate_credit/extensions.py): `ClimateStates` (`.binary(...)` or
`.from_asset_jumps(probs, alphas, lgd0, asset_vol)`), `multistate_pd`, `multistate_conditional_pd`,
`multistate_expected_loss`, `multistate_loss_cdf`, `multistate_var`, `multistate_capital`.

### Climate damage index swap and hedging

In Eq. (21), $(1-e^{-\alpha})\,q$ is the expected payout of a swap that pays the damage fraction $1-e^{-\alpha}$ if
the event occurs. The paper calls it the swap spread; a pure digital paying 1 on the event would cost $q$. Both ignore discounting
and risk premia. $(1-LGD_0)/LGD_0$ is the size of the systematic exposure to the swap. For non-identical assets the multiplier becomes
$1+\text{swap}\cdot\beta\cdot(1-LGD_0)/LGD_0$, where $\beta$ is the sensitivity of the bank's recoverable-asset damage to the index damage.
A perfect hedge with an ideal counterparty resets $P_D\to P_D^0$ and $LGD_1\to LGD_0$, so Eq. (21) returns to Eq. (3).
The paper suggests such an index could trade and clear like CDX or iTraxx.

**Code:** `swap_spread`, `hedged_multiplier(q, alpha, lgd0, beta, hedge_ratio)`. The partial `hedge_ratio`, which scales the add-on
by $(1-h)$, is my own interpolation: the paper only covers the perfect hedge.

### Physical plus transition risk (Eq. 24–25)

A second, independent jump $\omega\in\{0,1,2\}$ covers no transition, orderly transition and disorderly transition, with jump sizes
$e^{-\delta_i}$ taken from bank stress tests and probabilities $p_i$ to be found:

$$
\hat V_t = V_t\,(e^{-\alpha}\xi+(1-\xi))\cdot e^{-\delta_\omega}
$$

The probabilities are calibrated from three self-consistency conditions (Eq. 25), using the bank's PDs with physical risk only, transition
risk only, and both. Given $\hat\alpha$, the $p_i$ enter linearly, so the system is a 2×2 linear solve.

**Code:** `joint_states` builds the six joint (ξ, ω) states as a `ClimateStates`, adding log-jumps and setting LGD per state from Eq. 15.
`calibrate_transition` solves Eq. (25) and returns a condition number. The paper does not discuss identification, but
with independent jumps the joint-PD equation is nearly collinear with the transition-PD equation. In practice this
leaves the transition probabilities unidentified (see section 13, finding 4).

## 12. Worked example: Annex 1 (BBB loan, Mobile, Alabama)

The example is a loan to an investment-grade company, secured on commercial real estate in Mobile, AL, which is exposed to Category 3+ hurricanes.

**Inputs, as chosen in the paper:**

| Input | Value | Source in paper |
|---|---|---|
| $q$ | 3.0% (2025 forecast), or 4.8% (95% confidence) | S&P hurricane forecast; long-run mean 1.7% |
| $P_D^0$ | 0.30% | BBB one-year PD |
| $P_D/P_D^0$ | $1 + 0.161\,(q/0.017 - 1)$: 1.1224 at q = 3%, 1.2936 at q = 4.8% | De Spiegeleer et al.: +16.1% defaults when storm frequency doubles; the paper assumes PD rises linearly with frequency |
| $LGD_0$ | 10% | 90% recovery on secured high-quality loans |
| $\sigma_A$ | 30% | implied equity volatility, delta ≈ 1 |
| $\rho$ | 22.3% | Basel formula at $P_D^0$ |

**Steps and results** (`python examples/annex1_mobile_alabama.py`):

| Step | q = 3.0%: code | paper | q = 4.8%: code | paper |
|---|---|---|---|---|
| $P_D$ | 0.337% | 0.337% | 0.388% | 0.39% |
| $\hat\alpha$ from Eq. (6) | 0.584 | 0.58 | 0.721 | 0.72 |
| $C_V(Q)$ baseline → Eq. (12) | 0.0720 → 0.0747 | 0.0720 → 0.0747 | 0.0720 → 0.0774 | – |
| $C_V(Q)-P_D$ baseline → climate | 0.0690 → 0.0714 | 0.0690 → 0.0714 | 0.0690 → 0.0735 | 0.0690 → 0.0735 |
| $e^{-\alpha}$, with $\alpha=0.3\hat\alpha$ | 0.839 | 0.84 | 0.806 | 0.805 |
| $LGD_1$ from Eq. (15) | 24.5% | 24.8% | 27.5% | 27.5% |
| UL baseline | 0.0069 | 0.0069 | 0.0069 | 0.0069 |
| **RWA uplift, model $LGD_1$** | **+7.9%** | **+7.9%** | **+15.5%** | **+15.5%** |
| **RWA uplift, $LGD_1$ = 40%** | **+12.7%** | **+12.7%** | **+21.9%** | **≈22%** |

All reported figures match. The only gap is $LGD_1$ at q = 3%: Eq. (15) with the paper's own $e^{-\alpha}=0.84$ gives 24.4%,
not the 24.8% printed. The RWA uplift is the same either way.

**How PD is mapped from q.** The paper's rule is $P_D/P_D^0 = 1 + 0.161\,(q/q_{LT}-1)$, with long-run mean $q_{LT}=1.7\%$.
This is affine in $q$ and equals 1 at the long-run mean. The paper writes the q = 3% case as $0.76$, a rounding of
$3/1.7-1=0.7647$. The q = 4.8% case follows from the same rule: $1+0.161\times1.8235=1.2936$, so $P_D=0.388\%$. The
uplift is not proportional to $q$ because the rule's zero sits at 1.7%, not at 0.

**Sensitivity to the 16.1% rule.** This rule is a single estimate from one study, and every number above depends on it. Halving or
increasing it by half (`python examples/diagnostics.py`, check 8):

| Storm rule s | q | α̂ | Paper Eq. (22) | Diversified | Concentrated |
|---|---|---|---|---|---|
| 0.080 | 3.0% | 0.387 | +5.4% | +9.5% | +45% |
| **0.161** | 3.0% | 0.586 | +7.9% | +18.1% | +148% |
| 0.240 | 3.0% | 0.721 | +9.6% | +25.6% | +268% |
| 0.080 | 4.8% | 0.494 | +11.0% | +22.0% | +129% |
| **0.161** | 4.8% | 0.721 | +15.5% | +40.9% | +345% |
| 0.240 | 4.8% | 0.872 | +18.2% | +56.8% | +557% |

(At s = 0.161 and q = 3%, this table uses the unrounded 0.7647, so α̂ = 0.586 rather than 0.584.) Taken together, the
uncertainty in $q$ and in the storm rule spans about a factor of 6 in the diversified add-on.

**The external LGD₁ is a separate damage scenario.** The model has one damage parameter, $\alpha=\hat\alpha\,\sigma_A$, which
drives both the PD uplift and $LGD_1$. The paper's alternative $LGD_1=40\%$ implies $\alpha=0.405$ (33% asset damage).
With the $\hat\alpha=0.584$ pinned by the PDs, that needs $\sigma_A=0.69$, against the 30% used. So the 40% case assumes
more damage to collateral than the PDs imply, and should be read as a second scenario rather than a refinement of the same calibration. The paper concludes that physical risk raises RWA by
up to about 20% for this high-quality portfolio, **in 2025**, not in a distant IPCC scenario.

## 13. Findings beyond the paper

These came out of checking the closed forms against the exact model and Monte Carlo
(`python examples/diagnostics.py`). The paper's own Eq. (14) is the reference throughout.

**1. Eq. (22) understates capital in the paper's own example, at every level of concentration.**
Eq. (22) is a first-order expansion in $q$ and $\hat\alpha$ (Annex 3). In Annex 1 neither is small next to what matters:
$q=3\%$ is 30 times the tail probability $1-Q=0.1\%$, and $\hat\alpha=0.58$ is not small inside $\Phi$.
How large the error is depends on **how much of the portfolio sits in the hazard zone**, because the event is a single
draw for every exposed loan. The table shows the climate add-on per exposed loan from the exact model (Annex 1 loans in
Mobile, the rest of the portfolio identical but unexposed):

| Share of portfolio exposed | q = 3.0% | q = 4.8% |
|---|---|---|
| Paper, Eq. (22) (any share) | +7.9% | +15.5% |
| → 0, `ul("conditional")` | +18.0% | +40.9% |
| 1% | +19.4% | +46.0% |
| 10% | +34.7% | +100.9% |
| 50% | +98.5% | +276.4% |
| 100%, `ul("exact")` | +146.8% | +344.7% |

The two ends of the table are:
- **Diversified limit (`ul("conditional")`).** The portfolio's 99.9% quantile is still set by the economic factor, and
  each exposed loan contributes its exact expected loss at $y=\Phi^{-1}(Q)$, averaged over the event. That is Eq. (9)
  with channel-specific LGDs, minus the exact EL. This measure is additive loan by loan, like Eq. (22), but without the
  linearisation. At the paper's parameters it is still 2.3–2.6 times the paper's add-on.
- **Fully concentrated (`ul("exact")`).** The whole portfolio is exposed, so the 99.9% quantile falls inside the event
  channel. This is the quantile of Eq. (14), confirmed by Monte Carlo:

  | Annex 1, q = 3%, all loans exposed | 99.9% loss |
  |---|---|
  | Vasicek, no climate | 0.00720 |
  | Eq. (20), paper | 0.00780 |
  | Eq. (14), exact | 0.01743 |
  | Monte Carlo, ASRF limit (4m sims) | 0.01739 |
  | Monte Carlo, 5,000 obligors (1m sims) | 0.01747 |

So the paper's +7.9% to +20% headline is too low for any portfolio. How much too low depends on concentration: a factor
of about 2.3 for a diversified bank, up to about 20 for a lender concentrated on the Gulf coast.

**Where the derivation goes wrong.** Annex 3 expands the exact quantile of Eq. (14) in $q$. Its first expression for the
first-order term, $L^*_{Q,1}=LGD_0\,\big(Q-W_1(L_{Vas}LGD_0/LGD_1)\big)/W_0'(L_{Vas})$, is correct: it matches a numerical
derivative to 5 digits. The paper then simplifies it for "small α" by linearising $\Phi_{-\hat\alpha/\sqrt\rho}$. In the
99.9% tail that step breaks down quickly. The table compares the slope of the 99.9% loss with respect to $q$, at $q\to0$
(`diagnostics.py`, check 7):

| α̂ | Exact slope | Eq. (20) slope | Ratio |
|---|---|---|---|
| 0.01 | 0.00037 | 0.00035 | 1.07 |
| 0.05 | 0.00240 | 0.00174 | 1.38 |
| 0.1 | 0.00651 | 0.00347 | 1.87 |
| 0.3 | 0.0567 | 0.0103 | 5.5 |
| 0.584 (Annex 1) | 0.316 | 0.0195 | 16.2 |

So the problem is not only that $q=3\%$ is large next to $1-Q$. Even for tiny $q$, Eq. (20) gets the climate add-on
right only for jumps of $\hat\alpha \lesssim 0.01$. At $\hat\alpha=0.1$ it already halves the add-on. (Total capital can
still look right when $q$ is tiny, simply because the add-on is negligible.)

Eq. (22) is the common small-α̂ limit of **two different risk measures**:
- the loss at the economic factor's Q-quantile, averaged over the event. This is the diversified measure, and its correction
  terms scale with $\hat\alpha/\sqrt{1-\rho}\approx0.66$;
- the joint Q-quantile of the loss. This is the concentrated measure, and its correction terms scale with $\hat\alpha/\sqrt\rho\approx1.24$.

The two coincide only to first order in $\hat\alpha$, so going to higher order in $q$ alone would not repair Eq. (22).
The fix is to use the exact measure that fits the portfolio. (Credit to an external review for pressing on this measure mismatch.)

**2. Portfolio invariance holds only in the diversified limit.**
Eq. (22) and `ul("conditional")` are additive by construction. The exact capital of a portfolio with material exposure to
one event is not, because the charge per exposed loan rises with the exposed share (table above). In a three-class test
portfolio, the three measures give:
- exact standalone charges summed: 0.0353;
- true portfolio capital: 0.0333 (a 6% diversification effect);
- Eq. (22): 0.0289. Most of this gap is the approximation error from finding 1, not the loss of invariance.

Regulators could get an additive charge from `ul("conditional")` and handle concentration in the hazard zone as a
separate concentration-risk add-on (Pillar 2), the way name concentration is handled today.

**3. Eq. (17) understates the climate part of expected loss by a factor of 4.5.**
The exact climate add-on is $q(LGD_1-LGD_0)\,\Phi(C^*+\hat\alpha)$, which uses the PD given the event (1.52% in Annex 1).
Eq. (17) puts the unconditional $P_D$ (0.337%) in its place. The add-on is understated 4.5 times, and total EL by 13%.
This matters for provisioning (IFRS 9 / CECL), which the paper names as a use of the model.

**4. The transition-risk calibration (Eq. 25) is not identified in practice.**
The condition number of the 2×2 system is about 650. An error of 0.1% in the joint PD moves the orderly-transition
probability from 0.15 to 0.20, which is well below the precision of any internal PD model. Eq. (25) needs extra structure,
such as fixing the ratio $p_2/p_1$ from scenario design or using a prior, before it can be used.

## 14. Sensitivity analysis: climate event correlated with the economy

The paper assumes the climate event $\xi$ is independent of the economic factor. This section tests how robust the
conclusions are when that assumption is dropped. It is a **structural departure from the paper**, not a parameter the paper leaves
free, and its key parameter **τ cannot be identified from the bank's PDs**. Given $\tau$, the two PDs pin $C^*$ and
$\hat\alpha$, and any $\tau$ fits them equally well. So $\tau$ has to come from outside, for example a regression of
regional default rates on disaster indicators. Without that, the results below are a family of models, not one answer.
This is a stronger limitation than the transition-risk calibration (finding 4): there the parameters are identified in
principle but numerically fragile, whereas here they are not identified at all.

The event remains a separate systematic factor. Its noise $\varepsilon$ is drawn once for the whole portfolio, not per
obligor, so the model keeps two systematic factors, $y$ and $\varepsilon$, unless $|\tau|=1$. What changes is that the two factors are
no longer independent.

[correlated.py](climate_credit/correlated.py) links them with a Gaussian copula. A latent climate variable

$$
W = \tau\,y + \sqrt{1-\tau^2}\,\varepsilon,\qquad \xi = \mathbf 1\{W > \Phi^{-1}(1-q)\}
$$

fires the event, where $y=-S$ is the economic stress factor (high $y$ means a bad state) and $\varepsilon$ is independent noise.
- The marginal $P(\xi=1)=q$ is unchanged.
- $\tau>0$ makes events more likely in downturns: $q(y)=\Phi\big((\tau y-\Phi^{-1}(1-q))/\sqrt{1-\tau^2}\big)$.
- $\tau=0$ is exactly the paper's model, and the tests check this.

Since $\mathrm{corr}(A,W)=-\sqrt\rho\,\tau$, every quantity is still closed form in the bivariate normal CDF $\Phi_2$:

$$
P_D = \Phi_2(C^*,k;-\sqrt\rho\tau) + \Phi(C^*+\hat\alpha) - \Phi_2(C^*+\hat\alpha,k;-\sqrt\rho\tau),\qquad k=\Phi^{-1}(1-q)
$$

$$
P(\text{Loss}<L) = \Phi_2\big(y_0(L),k;\tau\big) + \Phi\big(y_1(L)\big) - \Phi_2\big(y_1(L),k;\tau\big)
$$

Here $y_j(L)=\big(\sqrt{1-\rho}\,\Phi^{-1}(L/LGD_j)-C^*-\hat\alpha_j\big)/\sqrt\rho$ is the economic factor at which channel $j$ loses $L$.
At $\tau=0$ this reduces to Eq. (14). The self-consistency condition (6) becomes a one-dimensional root-find for $\hat\alpha$.

**Code:** `capital_correlated(pd0, pd, q, lgd0, tau, ...)` returns α̂, PD, LGD₁, $q(y_Q)$, EL and both capital measures.
The building blocks are `pd_correlated`, `calibrate_alpha_hat_correlated`, `conditional_pd_correlated`,
`expected_loss_correlated`, `loss_cdf_correlated` and `var_correlated`. `portfolio_var_exact` and `simulate_losses`
take `tau=`. The analysis is `python examples/correlated_climate.py`, and the plot is `outputs/correlated_capital.png`.

**Results, Annex 1 loan.** There are two ways to compare, and they answer different questions.

*(a) Bank PD held fixed, α̂ re-solved.* This is the paper's logic, where the PDs are inputs. The climate add-on to UL/RWA:

| τ | α̂ | $q(y_Q)$ | q = 3%: diversified | concentrated | q = 4.8%: diversified | concentrated |
|---|---|---|---|---|---|---|
| −0.2 | 0.765 | 0.5% | +3.9% | +157.7% | +9.5% | +369.1% |
| **0 (paper)** | 0.584 | 3.0% | **+18.0%** | **+146.8%** | **+40.9%** | **+344.7%** |
| 0.1 | 0.507 | 5.7% | +27.9% | +142.3% | +62.5% | +330.9% |
| 0.2 | 0.438 | 9.9% | +38.9% | +138.7% | +86.2% | +316.6% |
| 0.3 | 0.379 | 15.9% | +50.5% | +135.6% | +110.6% | +301.8% |
| 0.5 | 0.282 | 34.9% | +73.3% | +130.0% | +156.1% | +269.6% |

(α̂ and $q(y_Q)$ are shown for q = 3%. Diversified = `ul_conditional`, exposed share → 0. Concentrated = `ul_exact`, whole portfolio exposed.)

*(b) α̂ held at its τ = 0 value, so PD rises with τ.* This is the same physical damage, with events now clustering in downturns.
For q = 3%, the diversified add-on goes +2.6% / +18.0% / +61.6% / +221.8% at τ = −0.2 / 0 / 0.2 / 0.5. The concentrated add-on
goes +58% / +147% / +270% / +474%.

**What it shows**
- **If τ is positive, it matters a lot for a diversified bank.** At the 99.9% economic state, the probability of the
  event rises from 3% to 16% at τ = 0.3. The add-on rises from +18% to +51% even with the bank's PD unchanged. That is 6.4 times
  the paper's +7.9%, which breaks down as 2.3× from the linearisation error (+7.9% → +18%) times 2.8× from correlation
  (+18% → +51%). So at τ = 0.3, correlation is the larger factor; at τ = 0.2 the two are about equal (2.2× vs 2.3×), and below that the linearisation error dominates. A small negative τ
  pushes the add-on close to the paper's figure.
- **For a concentrated lender, the model has no answer that holds regardless of what the PD means.** The sign of the
  effect depends on what the bank's PD already contains. With the PD fixed (comparison a), positive τ means a smaller jump
  explains the same PD, because part of the uplift now comes from the timing of the event. LGD₁ shrinks, and capital falls
  from +147% to +130% at τ = 0.5. With the damage fixed (comparison b), capital rises to +474%. The fall under (a) is a
  consequence of holding PD fixed, not a robust property of correlated climate risk, and it should not be read as one.
- **Concentration and correlation interact.** At τ = 0.3, the add-on per exposed loan is +52% / +63% / +105% for 1% / 10% / 50%
  of the portfolio exposed. At τ = 0 it is +19% / +35% / +99%. Correlation matters most where Eq. (22) is meant to apply: small,
  additive exposures.
- **Which comparison is right depends on what the bank's $P_D$ measures.** If $P_D$ comes from a damage model that
  ignores the economic cycle, α̂ is the physical quantity, comparison (b) applies, and correlation raises capital at every
  concentration level. If $P_D$ comes from data that already includes the clustering of disasters and downturns, (a) applies.
  In the paper's own example, $P_D$ comes from a supply-chain simulation that varies disaster frequency (De Spiegeleer et al.),
  not from default history over the economic cycle. That points to (b) for Annex 1.

**Validation.** The Monte Carlo is not circular. `simulate_losses` draws $y$ and the latent $W$ directly and sets the
event from the threshold. Given $(y,\xi)$, it uses only the standard Vasicek conditional default rate (ASRF mode) or draws
binomial defaults for a finite number of obligors. That makes it an independent check of the joint loss CDF, the
bivariate-normal integration and the calibration. At τ = 0.3 the closed-form 99.9% loss is 0.01666. Over 4 million
simulations the Monte Carlo gives 0.01672; across seeds at 2 million simulations, the results fall within −1.5% to +0.6% of
the closed form, both in the ASRF limit and with 5,000 or 50,000 obligors. The tests repeat the check at τ = 0.4.

**Limits of this extension.**
- **Copula tails.** For a given τ, a Gaussian copula has no tail dependence between the event and a deep downturn. If
  disasters and severe recessions do cluster in the extreme tail, this model **understates** capital, most of all in the
  diversified case, where $q(y_Q)$ drives the result. A t-copula or an extreme-value dependence model would be the next step.
  The copula does not weaken spatial clustering: one draw of ξ still hits every exposed obligor, exactly as in the paper.
  Clustering across years is outside the one-year horizon.
- **τ is uncalibrated** (see the top of this section).
- **Co-occurrence, not causation.** The economic factor's own distribution is unchanged, so the event does not *cause* the downturn; it only co-occurs with it.

## 15. Limitations and open questions

These are assumptions in the paper that the code keeps but does not test. Some are taken from an external review of an earlier version
of this README. Where that review was wrong, the correction is noted.

- **Independence of the event and the economy.** The paper assumes $\xi \perp S$ and cites low CAT-bond correlations with
  market factors. That is evidence about how instruments are priced, not about whether a regional disaster coincides
  with a regional downturn. Section 14 relaxes the assumption. For a diversified bank, the result depends heavily on τ,
  which still has to be calibrated.
- **One event, one q.** The framework has a single systematic event shared by all exposed obligors. Unexposed obligors are
  handled with $\hat\alpha=0$. A bank exposed to several regions or hazards needs several independent events, which Eq. (14)
  does not cover. `ClimateStates` can represent outcomes of one event, not independent events in different regions.
- **Sensitivity to q.** The paper's two cases (q = 3.0% and 4.8%, the S&P central forecast and its 95% bound) are best
  read as an uncertainty band: +7.9% to +15.5% on Eq. (22), +18% to +41% conditional. Note that the paper ties $P_D$
  to $q$ through the "16.1% rule", so changing $q$ alone at fixed $P_D$ is not the paper's experiment.
- **No plausibility check on α̂.** Eq. (6) always returns an $\hat\alpha$ if the PDs allow one. Nothing checks
  that the implied asset loss $1-e^{-\alpha}$ (16% in Annex 1) makes physical sense. `loan.summary()` reports it. A real
  application should compare it with engineering damage functions for the asset type and hazard (for example, FEMA Hazus
  loss ratios for commercial property in Category 3 winds). This README has not done that comparison.
- **LGD₁ (Eq. 15) is an assumption, not a derivation.** The paper presents it as the conservative case: the jump falls just
  before maturity, with assets just above the debt level. It is not derived from the distribution of assets at default.
  The 40% alternative in Annex 1 is the paper's example of an external LGD₁ (about 33% asset damage). It is not consistent with
  the PD-implied jump unless asset volatility is 69% (section 12). The choice moves the conditional uplift from +18% to +31%.
- **Asset volatility.** $\alpha=\hat\alpha\,\sigma_A$ uses diffusion volatility. Implied equity volatility also contains the
  jump variance, which adds $q(1-q)\hat\alpha^2 \approx 1\%$ to the variance in Annex 1, so this is negligible here.
- **Horizon and maturity adjustment.** $q$ is an annual probability and the PDs are one-year PDs, which is consistent. Basel's
  MA was calibrated on diffusion risk only, so applying it to the jump is untested. For multi-year exposures, the paper does not
  say whether $q$ should become $1-(1-q)^M$. The paper gives no maturity for the Annex 1 loan; its results are UL ratios,
  where MA cancels. The code computes MA from $P_D^0$ with $M=2.5$ by default.

The review got two points wrong:
- It said the model prices a two-factor risk with a one-factor correlation. In fact, $\rho$ is the correlation of the
  diffusion part only, and the jump enters Eq. (14) as its own factor. The "effective correlation" of 23.1% is correct arithmetic, but
  correlation is not a sufficient statistic for this non-Gaussian dependence.
- It said "the paper knows" about the ill-conditioning in Eq. (25). That observation was this README's, not the paper's.

## Code layout

| File | Paper content |
|---|---|
| [climate_credit/basel.py](climate_credit/basel.py) | Eq. 1–3, Basel correlation and maturity adjustment |
| [climate_credit/model.py](climate_credit/model.py) | Eq. 4–22 and `PhysicalRiskLoan` |
| [climate_credit/extensions.py](climate_credit/extensions.py) | Section 5: Eq. 23–25 and swap hedging |
| [climate_credit/portfolio.py](climate_credit/portfolio.py) | Heterogeneous portfolios and Monte Carlo (beyond the paper) |
| [climate_credit/correlated.py](climate_credit/correlated.py) | Climate event correlated with the economy, section 14 (beyond the paper) |
| [examples/correlated_climate.py](examples/correlated_climate.py) | Capital vs correlation τ, plot to `outputs/correlated_capital.png` |
| [examples/annex1_mobile_alabama.py](examples/annex1_mobile_alabama.py) | Annex 1 reproduction and loss-tail plot |
| [examples/diagnostics.py](examples/diagnostics.py) | Findings in section 13 |
| [tests/test_model.py](tests/test_model.py) | Annex 1 values, limiting cases, MC agreement, concentration limit, extensions |
| [tests/test_correlated.py](tests/test_correlated.py) | Correlated model: τ = 0 equals the paper, calibration, MC agreement at τ = 0.4 |
