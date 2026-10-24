"""Physical climate risk in the Vasicek / IRB credit risk model.

Implementation of Pozdyshev, Lobanov & Ilinsky (2025), "Incorporating physical
climate risks into banks' credit risk models", BIS Working Paper No 1274.
"""

from .basel import (basel_correlation, irb_capital, irb_rwa, maturity_adjustment,
                    vasicek_conditional_pd)
from .model import (PhysicalRiskLoan, alpha_from_alpha_hat, calibrate_alpha_hat,
                    capital_conditional, capital_exact, capital_generalised, climate_multiplier,
                    conditional_pd, conditional_pd_first_order, cv_climate,
                    expected_loss, expected_loss_first_order, lgd_climate,
                    mixture_cdf, mixture_ppf, pd_from_alpha_hat, portfolio_loss_cdf,
                    portfolio_var, portfolio_var_first_order, q_deformed_cdf,
                    q_deformed_ppf)
from .extensions import (ClimateStates, calibrate_transition, hedged_multiplier,
                         joint_states, multistate_capital, multistate_conditional_pd,
                         multistate_expected_loss, multistate_loss_cdf, multistate_pd,
                         multistate_var, swap_spread)
from .correlated import (bvn_cdf, calibrate_alpha_hat_correlated, capital_correlated,
                         conditional_pd_correlated, event_prob_given_y,
                         expected_loss_correlated, loss_cdf_correlated, pd_correlated,
                         var_correlated)
from .portfolio import asrf_loss, portfolio_var_exact, simulate_losses
