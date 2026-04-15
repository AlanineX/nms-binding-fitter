"""Test Guan model: synthesize, fit, verify recovery, and check gamma->0 limit."""
import os
import sys
import numpy as np
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(ROOT))

from scripts_binding.models import guan


def synth(S, N, beta, gamma, Ks_true, P_tot_M, L_totals_M):
    ln_params = np.concatenate(([np.log(beta), gamma], np.log(Ks_true)))
    F_list = []
    for L_tot in L_totals_M:
        L_free = guan.solve_L_free(L_tot, P_tot_M, ln_params, S, N)
        F_list.append(guan.calculate_fractions_model(L_free, ln_params, S, N))
    return F_list, ln_params


def fit(L_totals_M, F_exps, P_tot_M, S, N, lnK0):
    history = []
    return least_squares(
        guan.residuals, lnK0,
        args=(L_totals_M, P_tot_M, F_exps, S, N, history),
        method="lm",
    )


def main():
    print("=== Guan model test ===")

    S = 3
    N = 3
    Ks_true = np.array([1e5, 2e4, 5e3])
    beta_true = 2e3
    gamma_true = 0.5
    P_tot_M = 1e-6
    L_totals_M = np.array([1, 5, 10, 50, 100, 500, 1000]) * 1e-6

    F_exps, ln_true = synth(S, N, beta_true, gamma_true, Ks_true, P_tot_M, L_totals_M)
    print(f"True beta={beta_true:.3e}, gamma={gamma_true}, "
          f"Ks={Ks_true}, S={S}, N={N}")

    # Perturbed start
    lnK0 = ln_true.copy()
    lnK0[0] += np.log(0.5)  # beta off by 2x
    lnK0[1] = 0.1            # gamma start near 0
    lnK0[2:] += np.log(0.7)  # Ks off

    res = fit(L_totals_M, F_exps, P_tot_M, S, N, lnK0)
    beta_fit = np.exp(res.x[0])
    gamma_fit = res.x[1]
    Ks_fit = np.exp(res.x[2:])
    print(f"\nFit converged: {res.success}, cost = {res.cost:.3e}")
    print(f"Fitted beta={beta_fit:.3e}, gamma={gamma_fit:.4f}")
    print(f"Fitted Ks={Ks_fit}")

    rel_err_beta = abs(beta_fit - beta_true) / beta_true
    abs_err_gamma = abs(gamma_fit - gamma_true)
    rel_err_Ks = np.abs(Ks_fit - Ks_true) / Ks_true
    print(f"|beta rel err|={rel_err_beta:.4f}, |gamma abs err|={abs_err_gamma:.4f}, "
          f"max |Ks rel err|={rel_err_Ks.max():.4f}")
    assert rel_err_beta < 0.01 and abs_err_gamma < 0.01 and rel_err_Ks.max() < 0.01, "FAIL"
    print("PASS: parameters recovered within 1%")

    # Check gamma -> 0 limit (Shimon-like behavior, but Guan partition is different
    # from Shimon's Poisson, so we just check that the fit is stable at gamma=0)
    print("\n=== gamma=0 limit check ===")
    F_exps_g0, ln_true_g0 = synth(S, N, beta_true, 0.0, Ks_true, P_tot_M, L_totals_M)
    lnK0_g0 = ln_true_g0.copy()
    lnK0_g0[0] += np.log(0.5)
    lnK0_g0[1] = 0.5
    lnK0_g0[2:] += np.log(0.7)
    res_g0 = fit(L_totals_M, F_exps_g0, P_tot_M, S, N, lnK0_g0)
    print(f"gamma=0 fit converged: {res_g0.success}, cost = {res_g0.cost:.3e}, "
          f"gamma_fit = {res_g0.x[1]:.4f}")
    assert abs(res_g0.x[1]) < 0.01, "FAIL: did not recover gamma=0"
    print("PASS: gamma=0 limit recovered")


if __name__ == "__main__":
    main()
