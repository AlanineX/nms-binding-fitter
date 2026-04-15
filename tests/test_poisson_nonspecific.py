"""Test Shimon model: synthesize data with known params, fit, verify recovery."""
import os
import sys
import numpy as np
from scipy.optimize import least_squares

# Resolve absolute paths and import the package
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(ROOT))

from scripts_binding.models import poisson_nonspecific as model


def make_synthetic_data(S, N, Ks_true, Kn_true, P_tot_M, L_totals_M):
    """Generate F_i distributions from Shimon model at given L_totals."""
    ln_params_true = np.log(np.concatenate(([Kn_true], Ks_true)))
    F_list = []
    for L_tot in L_totals_M:
        L_free = model.free_ligand(L_tot, P_tot_M, ln_params_true, S, N)
        F = model.mole_fractions(L_free, ln_params_true, S, N)
        F_list.append(F)
    return F_list, ln_params_true


def fit_shimon(L_totals_M, F_exps, P_tot_M, S, N, lnK0):
    """Fit Shimon model via least_squares."""
    ssr_history = []
    result = least_squares(
        model.residual_vector,
        lnK0,
        args=(L_totals_M, P_tot_M, F_exps, S, N, ssr_history),
        method="lm",
    )
    return result


def main():
    print("=== Shimon model test ===")

    # True parameters (in 1/M)
    S = 3
    N = 3
    Ks_true = np.array([1e5, 2e4, 5e3])  # Kd = 10, 50, 200 uM
    Kn_true = 1e3                         # Kd = 1000 uM
    P_tot_M = 1e-6                        # 1 uM protein
    L_totals_M = np.array([1, 5, 10, 50, 100, 500, 1000]) * 1e-6  # uM -> M

    # Generate synthetic data
    F_exps, ln_true = make_synthetic_data(S, N, Ks_true, Kn_true, P_tot_M, L_totals_M)
    print(f"True ln_params: {ln_true}")
    print(f"True Kn (1/M): {Kn_true:.3e}, Kd_n: {1/Kn_true*1e6:.1f} uM")
    for i, k in enumerate(Ks_true, 1):
        print(f"True Ks_{i} (1/M): {k:.3e}, Kd_s_{i}: {1/k*1e6:.1f} uM")

    # Fit with perturbed initial guess (off by 2x in Kd)
    lnK0 = ln_true + np.log(0.5)  # start with half the true Ka
    result = fit_shimon(L_totals_M, F_exps, P_tot_M, S, N, lnK0)
    K_fit = np.exp(result.x)
    print()
    print(f"Fit converged: {result.success}, cost = {result.cost:.3e}")
    print(f"Fitted ln_params: {result.x}")
    print(f"Fitted Kn: {K_fit[0]:.3e}, Kd_n: {1/K_fit[0]*1e6:.2f} uM")
    for i, k in enumerate(K_fit[1:], 1):
        print(f"Fitted Ks_{i}: {k:.3e}, Kd_s_{i}: {1/k*1e6:.2f} uM")

    # Verify recovery within tolerance
    rel_err = np.abs(K_fit - np.exp(ln_true)) / np.exp(ln_true)
    print(f"\nMax relative error: {rel_err.max():.4f}")
    assert rel_err.max() < 0.01, f"Parameter recovery failed: max rel err = {rel_err.max()}"
    print("PASS: parameters recovered within 1%")

    # Also verify that solving L_free gives consistent mass balance
    print("\n=== Mass balance check ===")
    for L_tot, F_exp in zip(L_totals_M, F_exps):
        L_free = model.free_ligand(L_tot, P_tot_M, ln_true, S, N)
        avg = np.dot(np.arange(len(F_exp)), F_exp)
        L_bound = P_tot_M * avg
        residual = L_tot - L_free - L_bound
        print(f"L_tot={L_tot*1e6:6.1f} uM | L_free={L_free*1e6:8.3f} uM | "
              f"avg_n={avg:.3f} | L_bound={L_bound*1e6:8.3f} uM | residual={residual:+.2e}")
        assert abs(residual) < 1e-12, "Mass balance broken"
    print("PASS: mass balance satisfied")


if __name__ == "__main__":
    main()
