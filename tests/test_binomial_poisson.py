"""Test Daubenfeld model: synthesize, fit, verify recovery."""
import os
import sys
import numpy as np
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(ROOT))

from scripts_binding.models import binomial_poisson as model


def main():
    print("=== Daubenfeld model test ===")

    S = 4
    N = 4
    Ks_true = 5e4   # Kd = 20 uM, single noncooperative
    Kn_true = 1e3   # Kd = 1000 uM
    P_tot_M = 1e-6
    L_totals_M = np.array([1, 5, 10, 50, 100, 500, 1000, 2000]) * 1e-6

    ln_true = np.log([Kn_true, Ks_true])
    print(f"True: Kn={Kn_true:.3e} (Kd_n={1/Kn_true*1e6:.1f} uM), "
          f"Ks={Ks_true:.3e} (Kd_s={1/Ks_true*1e6:.1f} uM), S={S}")

    # Synthesize data
    F_exps = []
    for L_tot in L_totals_M:
        L_free = model.solve_L_free(L_tot, P_tot_M, ln_true, S, N)
        F_exps.append(model.calculate_fractions_model(L_free, ln_true, S, N))

    # Fit with perturbed initial guess
    lnK0 = ln_true + np.log(0.3)
    ssr_history = []
    result = least_squares(
        model.residuals, lnK0,
        args=(L_totals_M, P_tot_M, F_exps, S, N, ssr_history),
        method="lm",
    )
    K_fit = np.exp(result.x)
    print(f"\nFit converged: {result.success}, cost = {result.cost:.3e}")
    print(f"Fitted: Kn={K_fit[0]:.3e} (Kd_n={1/K_fit[0]*1e6:.2f} uM), "
          f"Ks={K_fit[1]:.3e} (Kd_s={1/K_fit[1]*1e6:.2f} uM)")

    rel_err = np.abs(K_fit - np.exp(ln_true)) / np.exp(ln_true)
    print(f"Max relative error: {rel_err.max():.4f}")
    assert rel_err.max() < 0.01, f"FAIL: max rel err = {rel_err.max()}"
    print("PASS: parameters recovered within 1%")

    # Print F_i at one mid concentration
    print(f"\nMole fractions at L_tot=100 uM (true model):")
    F_check = F_exps[4]
    for i, f in enumerate(F_check):
        print(f"  I_{i}: {f:.4f}")
    print(f"  sum = {F_check.sum():.6f}")


if __name__ == "__main__":
    main()
