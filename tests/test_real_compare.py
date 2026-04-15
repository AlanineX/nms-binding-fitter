"""Compare all NSB models on real SR-GroEL data: AMAC vs EDDA at 25C.

Each model is fit at S=7 (canonical 7 nucleotide pockets), and the per-buffer
parameters, SSR, and BIC are tabulated for direct comparison.

Data paths are CLI args so the test is portable.
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(ROOT))

from scripts_binding.models import REGISTRY


P_TOT_M = 1e-6
S_FIXED = 7
N_FIXED = 3
MODELS_TO_RUN = [
    "geometric_nonspecific",
    "poisson_nonspecific",
    "binomial_poisson_nonspecific",
    "power_law_nonspecific",
]


def load_csv(path):
    df = pd.read_csv(path)
    L_totals = df["Entry"].to_numpy()
    valid = L_totals > 0
    L_totals = L_totals[valid]
    F_cols = [c for c in df.columns if c.startswith("I")]
    F_arr = df.loc[valid, F_cols].to_numpy()
    return L_totals, [F_arr[i, :] for i in range(F_arr.shape[0])], F_cols


def pad_F(F_exp, target_len):
    if len(F_exp) >= target_len:
        return F_exp[:target_len]
    return np.concatenate([F_exp, np.zeros(target_len - len(F_exp))])


def bic(ssr, n_obs, n_params):
    return n_obs * np.log(ssr / n_obs) + n_params * np.log(n_obs)


def fit_model(model, L_totals, F_exps, P_tot, S, N):
    target_len = S + N + 1
    F_padded = [pad_F(F, target_len) for F in F_exps]
    history = []
    res = least_squares(
        model.residual_vector, model.initial_lnK(S),
        args=(L_totals, P_tot, F_padded, S, N, history),
        method="lm", max_nfev=5000,
    )
    n_obs = len(F_padded) * target_len
    ssr = float(np.dot(res.fun, res.fun))
    return res, ssr, bic(ssr, n_obs, model.n_params(S))


def run_one(label, path):
    print(f"\n=== {label}: {os.path.basename(path)} ===")
    L_totals, F_exps, F_cols = load_csv(path)
    print(f"Data: {len(L_totals)} concentrations ({L_totals.min()*1e6:.1f}-{L_totals.max()*1e6:.1f} uM), "
          f"{len(F_cols)} mass channels ({F_cols[0]}..{F_cols[-1]}), S={S_FIXED}, N={N_FIXED}")

    rows = []
    for name in MODELS_TO_RUN:
        model = REGISTRY[name]
        res, ssr, bic_v = fit_model(model, L_totals, F_exps, P_TOT_M, S_FIXED, N_FIXED)
        rows.append((name, model.n_params(S_FIXED), ssr, bic_v, res.x))

    print(f"\n{'Model':<32} {'#p':>3} {'SSR':>11} {'BIC':>10}  parameters")
    print("-" * 130)
    for name, np_, ssr, bic_v, x in rows:
        labels = REGISTRY[name].param_labels(S_FIXED)
        values = [(lbl, v if lbl == "gamma" else float(np.exp(v))) for lbl, v in zip(labels, x)]
        kd_strs = [f"{lbl}={1e6/v:.1f}uM" if lbl != "gamma" else f"gamma={v:.3f}" for lbl, v in values]
        print(f"{name:<32} {np_:>3} {ssr:11.4e} {bic_v:10.2f}  {', '.join(kd_strs)}")
    best = min(rows, key=lambda r: r[3])
    print(f"BIC winner: {best[0]}")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", action="append", required=True,
                        help="Repeated. LABEL:PATH  (e.g. AMAC_25C_1:/path/to/file.csv)")
    args = parser.parse_args()

    all_results = {}
    for spec in args.data:
        label, path = spec.split(":", 1)
        all_results[label] = run_one(label, path)

    print("\n\n=== Cross-buffer summary ===")
    for label, rows in all_results.items():
        best = min(rows, key=lambda r: r[3])
        print(f"{label:<20} BIC winner = {best[0]:<32} (BIC={best[3]:.1f})")


if __name__ == "__main__":
    main()
