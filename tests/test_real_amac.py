"""Compare 4 NSB models on real AMAC 25C data for SR-GroEL.

Fits each model at S=7 (canonical 7 nucleotide pockets) and reports
parameters, SSR, and BIC for direct comparison.
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
from scripts_binding.models import shimon
from scripts_binding.models import daubenfeld
from scripts_binding.models import guan


DATA_FILE = "/home/alan/working/non_specific_building/binding/corrected_SR1/amac_corrected/AMAC_25C_1.csv"
P_TOT_M = 1e-6   # 1 uM SR-GroEL heptamer
S_FIXED = 7      # 7 canonical nucleotide pockets
N_FIXED = 3      # allow up to 3 NSB ligands beyond S


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
    """Pad F_exp with zeros to target length."""
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


def main():
    print(f"=== Comparing 4 NSB models on {os.path.basename(DATA_FILE)} ===")
    L_totals, F_exps, F_cols = load_csv(DATA_FILE)
    print(f"Data: {len(L_totals)} concentrations, {len(F_cols)} mass channels ({F_cols[0]}..{F_cols[-1]})")
    print(f"L range: {L_totals.min()*1e6:.1f} - {L_totals.max()*1e6:.1f} uM")
    print(f"S={S_FIXED}, N={N_FIXED} (model truncation S+N+1={S_FIXED+N_FIXED+1})\n")

    results = {}

    # 1) Current mixed model: ln_params = [lnKn, lnKs1, ..., lnKsS], S+1 params
    lnK0 = np.log(np.concatenate(([1e3], np.full(S_FIXED, 1e5))))
    res, ssr, bic_v = fit_model(mixed_current, lnK0, L_totals, F_exps, P_TOT_M,
                                 S_FIXED, N_FIXED, S_FIXED + 1)
    K = np.exp(res.x)
    results["mixed_current"] = {"Kn": K[0], "Ks": K[1:], "ssr": ssr, "bic": bic_v,
                                 "params": S_FIXED + 1, "extra": {}}

    # 2) Shimon: same param layout as mixed, S+1 params
    lnK0 = np.log(np.concatenate(([1e3], np.full(S_FIXED, 1e5))))
    res, ssr, bic_v = fit_model(shimon, lnK0, L_totals, F_exps, P_TOT_M,
                                 S_FIXED, N_FIXED, S_FIXED + 1)
    K = np.exp(res.x)
    results["shimon"] = {"Kn": K[0], "Ks": K[1:], "ssr": ssr, "bic": bic_v,
                         "params": S_FIXED + 1, "extra": {}}

    # 3) Daubenfeld: 2 params [lnKn, lnKs] (single Ks for noncooperative binomial)
    lnK0 = np.log([1e3, 1e5])
    res, ssr, bic_v = fit_model(daubenfeld, lnK0, L_totals, F_exps, P_TOT_M,
                                 S_FIXED, N_FIXED, 2)
    K = np.exp(res.x)
    results["daubenfeld"] = {"Kn": K[0], "Ks": np.array([K[1]]), "ssr": ssr, "bic": bic_v,
                             "params": 2, "extra": {}}

    # 4) Guan: ln_params = [ln(beta), gamma, ln(Ks_1)..ln(Ks_S)], S+2 params
    lnK0 = np.concatenate(([np.log(1e3), 0.5], np.log(np.full(S_FIXED, 1e5))))
    res, ssr, bic_v = fit_model(guan, lnK0, L_totals, F_exps, P_TOT_M,
                                 S_FIXED, N_FIXED, S_FIXED + 2)
    beta = np.exp(res.x[0]); gamma = res.x[1]; Ks = np.exp(res.x[2:])
    results["guan"] = {"Kn": beta, "Ks": Ks, "ssr": ssr, "bic": bic_v,
                       "params": S_FIXED + 2, "extra": {"gamma": gamma}}

    # --- Print summary ---
    print(f"{'Model':<18} {'#param':>6} {'SSR':>12} {'BIC':>10}  Kd_n (uM)   Kd_s_1..S (uM)  notes")
    print("-" * 110)
    for name, r in results.items():
        Kd_n = 1e6 / r["Kn"]
        Kd_s = ", ".join(f"{1e6/k:7.1f}" for k in r["Ks"])
        extra = ""
        if "gamma" in r["extra"]:
            extra = f"gamma={r['extra']['gamma']:.3f}"
        print(f"{name:<18} {r['params']:>6} {r['ssr']:12.4e} {r['bic']:10.2f}  "
              f"{Kd_n:9.1f}   {Kd_s}   {extra}")

    print("\nLower BIC = better fit-vs-complexity tradeoff.")
    best = min(results.items(), key=lambda kv: kv[1]["bic"])
    print(f"BIC winner: {best[0]}")


if __name__ == "__main__":
    main()
