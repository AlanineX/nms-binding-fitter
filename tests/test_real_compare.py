"""Compare 4 NSB models on real SR-GroEL data: AMAC vs EDDA at 25C.

Each model is fit at S=7 (canonical 7 nucleotide pockets), and the per-buffer
parameters, SSR, and BIC are tabulated for direct comparison.

Models:
  - mixed_current  current Kn^(i-j) hybrid (geometric NSB)
  - poisson_nsb         Poisson NSB, stepwise specific
  - binomial_poisson     Poisson NSB, binomial specific (single Ks)
  - powerlaw_nsb           power-law NSB Kn_k = beta/k^gamma, stepwise specific
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(ROOT))

from scripts_binding.models import nonspecific as mixed_current
from scripts_binding.models import poisson_nsb
from scripts_binding.models import binomial_poisson
from scripts_binding.models import powerlaw_nsb


DATA_FILES = {
    "AMAC_25C_1": "/home/alan/working/non_specific_building/binding/corrected_SR1/amac_corrected/AMAC_25C_1.csv",
    "EDDA_25C_1": "/home/alan/working/non_specific_building/binding/corrected_SR1/edda/EDDA_25C_1.csv",
}
P_TOT_M = 1e-6
S_FIXED = 7
N_FIXED = 3


def load_csv(path):
    df = pd.read_csv(path)
    L_totals = df["Entry"].to_numpy()
    valid = L_totals > 0
    L_totals = L_totals[valid]
    F_cols = [c for c in df.columns if c.startswith("I")]
    F_arr = df.loc[valid, F_cols].to_numpy()
    F_exps = [F_arr[i, :] for i in range(F_arr.shape[0])]
    return L_totals, F_exps, F_cols


def pad_F(F_exp, target_len):
    if len(F_exp) >= target_len:
        return F_exp[:target_len]
    return np.concatenate([F_exp, np.zeros(target_len - len(F_exp))])


def bic(ssr, n_obs, n_params):
    return n_obs * np.log(ssr / n_obs) + n_params * np.log(n_obs)


def fit_model(model_mod, lnK0, L_totals, F_exps, P_tot, S, N, n_params):
    target_len = S + N + 1
    F_padded = [pad_F(F, target_len) for F in F_exps]
    history = []
    res = least_squares(
        model_mod.residuals, lnK0,
        args=(L_totals, P_tot, F_padded, S, N, history),
        method="lm",
        max_nfev=5000,
    )
    n_obs = len(F_padded) * target_len
    ssr = float(np.dot(res.fun, res.fun))
    return res, ssr, bic(ssr, n_obs, n_params)


def run_one(label, path):
    print(f"\n=== {label}: {os.path.basename(path)} ===")
    L_totals, F_exps, F_cols = load_csv(path)
    print(f"Data: {len(L_totals)} concentrations ({L_totals.min()*1e6:.1f}-{L_totals.max()*1e6:.1f} uM), "
          f"{len(F_cols)} mass channels ({F_cols[0]}..{F_cols[-1]}), S={S_FIXED}, N={N_FIXED}")

    rows = []

    # 1) mixed_current
    lnK0 = np.log(np.concatenate(([1e3], np.full(S_FIXED, 1e5))))
    res, ssr, bic_v = fit_model(mixed_current, lnK0, L_totals, F_exps, P_TOT_M,
                                 S_FIXED, N_FIXED, S_FIXED + 1)
    K = np.exp(res.x)
    rows.append(("mixed_current", S_FIXED + 1, ssr, bic_v, K[0], K[1:], None))

    # 2) poisson_nsb
    lnK0 = np.log(np.concatenate(([1e3], np.full(S_FIXED, 1e5))))
    res, ssr, bic_v = fit_model(poisson_nsb, lnK0, L_totals, F_exps, P_TOT_M,
                                 S_FIXED, N_FIXED, S_FIXED + 1)
    K = np.exp(res.x)
    rows.append(("poisson_nsb", S_FIXED + 1, ssr, bic_v, K[0], K[1:], None))

    # 3) binomial_poisson
    lnK0 = np.log([1e3, 1e5])
    res, ssr, bic_v = fit_model(binomial_poisson, lnK0, L_totals, F_exps, P_TOT_M,
                                 S_FIXED, N_FIXED, 2)
    K = np.exp(res.x)
    rows.append(("binomial_poisson", 2, ssr, bic_v, K[0], np.array([K[1]]), None))

    # 4) powerlaw_nsb
    lnK0 = np.concatenate(([np.log(1e3), 0.5], np.log(np.full(S_FIXED, 1e5))))
    res, ssr, bic_v = fit_model(powerlaw_nsb, lnK0, L_totals, F_exps, P_TOT_M,
                                 S_FIXED, N_FIXED, S_FIXED + 2)
    beta = np.exp(res.x[0]); gamma = res.x[1]; Ks = np.exp(res.x[2:])
    rows.append(("powerlaw_nsb", S_FIXED + 2, ssr, bic_v, beta, Ks, gamma))

    # Print
    print(f"\n{'Model':<16} {'#p':>3} {'SSR':>11} {'BIC':>10}  Kd_n (uM)   Kd_s (uM)              extra")
    print("-" * 130)
    for name, np_, ssr, bic_v, Kn, Ks, gamma in rows:
        Kd_n = 1e6 / Kn
        Kd_s = " ".join(f"{1e6/k:6.1f}" for k in Ks)
        extra = f"gamma={gamma:.3f}" if gamma is not None else ""
        print(f"{name:<16} {np_:>3} {ssr:11.4e} {bic_v:10.2f}  {Kd_n:9.1f}   {Kd_s:<40}  {extra}")
    best = min(rows, key=lambda r: r[3])
    print(f"BIC winner: {best[0]}")
    return rows


def main():
    all_results = {}
    for label, path in DATA_FILES.items():
        all_results[label] = run_one(label, path)

    # Cross-buffer comparison summary
    print("\n\n=== Cross-buffer summary ===")
    for label, rows in all_results.items():
        best = min(rows, key=lambda r: r[3])
        Kd_n = 1e6 / best[4]
        print(f"{label:<14} BIC winner = {best[0]:<14} (BIC={best[3]:.1f})  Kd_n={Kd_n:.1f} uM")


if __name__ == "__main__":
    main()
