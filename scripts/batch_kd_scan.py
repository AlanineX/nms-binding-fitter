"""Run all 4 NSB models on every replicate at every temperature for AMAC and EDDA.

Outputs:
  data_kd/all_fits_long.csv   one row per (buffer, temp, rep, model, parameter)
  data_kd/all_fits_wide.csv   one row per (buffer, temp, rep, model)
  data_kd/report.md           per-buffer summary tables and BIC ranking

Usage:
  PYTHONPATH=/home/alan/working/non_specific_building/binding \\
    python -m scripts_binding.scripts.batch_kd_scan
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from scripts_binding.models import REGISTRY


DATA_ROOTS = {
    "AMAC": "/home/alan/working/non_specific_building/binding/corrected_SR1/amac_corrected",
    "EDDA": "/home/alan/working/non_specific_building/binding/corrected_SR1/edda",
}
OUT_DIR = "/home/alan/working/docking/8b_groel/writing/data_kd"
P_TOT_M = 1e-6
TEMPS = [5, 10, 15, 20, 25, 30, 35]
REPS = [1, 2, 3]

# Per-buffer specific-site count (from the manuscript: AMAC saturates 7, EDDA plateaus at 5)
S_BY_BUFFER = {"AMAC": 7, "EDDA": 5}
N_NSB = 3   # max NSB stoichiometry beyond S we consider

# Models to scan (skip pure specific since it has no NSB component to compare)
MODELS_TO_RUN = ["geometric_nsb", "poisson_nsb", "binomial_poisson", "powerlaw_nsb"]


def load_csv(path):
    df = pd.read_csv(path)
    L_totals = df["Entry"].to_numpy()
    valid = L_totals > 0
    L_totals = L_totals[valid]
    F_cols = [c for c in df.columns if c.startswith("I")]
    F_arr = df.loc[valid, F_cols].to_numpy()
    F_exps = [F_arr[i, :] for i in range(F_arr.shape[0])]
    return L_totals, F_exps


def pad_F(F_exp, target_len):
    if len(F_exp) >= target_len:
        return F_exp[:target_len]
    return np.concatenate([F_exp, np.zeros(target_len - len(F_exp))])


def bic(ssr, n_obs, n_par):
    if ssr <= 0 or n_obs <= n_par:
        return np.nan
    return n_obs * np.log(ssr / n_obs) + n_par * np.log(n_obs)


def fit_one(model, L_totals, F_exps, P_tot, S, N):
    target_len = S + N + 1
    F_padded = [pad_F(F, target_len) for F in F_exps]
    lnK0 = model.initial_lnK(S)
    history = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        res = least_squares(
            model.residual_vector, lnK0,
            args=(L_totals, P_tot, F_padded, S, N, history),
            method="lm", max_nfev=5000,
        )
    n_obs = len(F_padded) * target_len
    ssr = float(np.dot(res.fun, res.fun))
    return res, ssr, bic(ssr, n_obs, model.n_params(S))


def collect_run(buffer_name, root, S, N):
    rows_long = []
    rows_wide = []
    for T in TEMPS:
        for R in REPS:
            csv_path = os.path.join(root, f"{buffer_name}_{T}C_{R}.csv")
            if not os.path.exists(csv_path):
                continue
            L_totals, F_exps = load_csv(csv_path)

            for model_name in MODELS_TO_RUN:
                model = REGISTRY[model_name]
                res, ssr, bic_v = fit_one(model, L_totals, F_exps, P_TOT_M, S, N)
                params = res.x.copy()
                # Convert log-space to physical Ka, except gamma which is linear
                labels = model.param_labels(S)
                K_values = []
                for lbl, val in zip(labels, params):
                    if lbl == "gamma":
                        K_values.append(val)
                    else:
                        K_values.append(float(np.exp(val)))

                # Long-form rows (one per parameter)
                for lbl, val in zip(labels, K_values):
                    Kd_uM = (1e6 / val) if (lbl != "gamma" and val > 0) else np.nan
                    rows_long.append({
                        "buffer": buffer_name, "temp": T, "rep": R,
                        "model": model_name, "param": lbl,
                        "value": val, "Kd_uM": Kd_uM,
                    })

                # Wide-form summary row
                wide = {
                    "buffer": buffer_name, "temp": T, "rep": R,
                    "model": model_name, "S": S, "N": N,
                    "n_params": model.n_params(S),
                    "ssr": ssr, "bic": bic_v, "converged": bool(res.success),
                }
                for lbl, val in zip(labels, K_values):
                    if lbl == "gamma":
                        wide[lbl] = val
                    else:
                        wide[f"{lbl}_Ka"] = val
                        wide[f"{lbl}_Kd_uM"] = (1e6 / val) if val > 0 else np.nan
                rows_wide.append(wide)
    return rows_long, rows_wide


def write_markdown(out_dir, df_wide):
    """Build a per-buffer summary markdown report."""
    lines = []
    lines.append("# NSB Model Comparison: SR-GroEL ADP Binding\n")
    lines.append("Mean Kd values across three replicates per temperature, per buffer, per model.\n")
    lines.append("Models compared:\n")
    lines.append("- **geometric_nsb** stepwise specific Ks_i + Kn^(i-j) NSB (current baseline)")
    lines.append("- **poisson_nsb** stepwise specific Ks_i + Poisson NSB (Shimon 2010)")
    lines.append("- **binomial_poisson** noncooperative single Ks + Poisson NSB (Daubenfeld 2006)")
    lines.append("- **powerlaw_nsb** stepwise Ks_i + power-law Kn_k = beta/k^gamma (Guan 2015)\n")
    lines.append(f"All fits use S=7 for AMAC and S=5 for EDDA, N={N_NSB}, P_tot=1 uM.\n")

    for buf, dfb in df_wide.groupby("buffer"):
        lines.append(f"\n## {buf}\n")

        # Mean Kd_n per model per temperature
        mean_kd_n = (dfb.groupby(["temp", "model"])["Kn_Kd_uM"]
                       .mean().unstack("model"))
        lines.append("### Mean Kd_n (µM) by temperature\n")
        lines.append(mean_kd_n.round(1).to_markdown())
        lines.append("")

        # Mean BIC per model per temperature
        mean_bic = (dfb.groupby(["temp", "model"])["bic"]
                       .mean().unstack("model"))
        lines.append("### Mean BIC by temperature (lower = better)\n")
        lines.append(mean_bic.round(1).to_markdown())
        lines.append("")

        # Mean Ks values at 25C across models
        d25 = dfb[dfb["temp"] == 25]
        ks_cols = sorted([c for c in d25.columns if c.startswith("Ks_") and c.endswith("_Kd_uM")])
        if ks_cols:
            ks_summary = d25.groupby("model")[ks_cols].mean().round(1)
            lines.append(f"### Mean Ks_i Kd values (µM) at 25 C\n")
            lines.append(ks_summary.to_markdown())
            lines.append("")

        # BIC winner per replicate
        winners = (dfb.loc[dfb.groupby(["temp", "rep"])["bic"].idxmin(),
                           ["temp", "rep", "model", "bic"]]
                      .reset_index(drop=True))
        win_counts = winners["model"].value_counts()
        lines.append("### BIC winner counts (lower BIC across all temp x rep)\n")
        for m, c in win_counts.items():
            lines.append(f"- **{m}**: {c} / {len(winners)}")
        lines.append("")

    md_path = os.path.join(out_dir, "report.md")
    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    return md_path


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    all_long = []
    all_wide = []
    for buf, root in DATA_ROOTS.items():
        S = S_BY_BUFFER[buf]
        print(f"\n=== {buf} (S={S}) ===")
        long, wide = collect_run(buf, root, S, N_NSB)
        print(f"  Collected {len(wide)} fits")
        all_long.extend(long)
        all_wide.extend(wide)

    df_long = pd.DataFrame(all_long)
    df_wide = pd.DataFrame(all_wide)
    df_long.to_csv(os.path.join(OUT_DIR, "all_fits_long.csv"), index=False)
    df_wide.to_csv(os.path.join(OUT_DIR, "all_fits_wide.csv"), index=False)
    md_path = write_markdown(OUT_DIR, df_wide)
    print(f"\nWrote {len(df_wide)} fits to:")
    print(f"  {os.path.join(OUT_DIR, 'all_fits_long.csv')}")
    print(f"  {os.path.join(OUT_DIR, 'all_fits_wide.csv')}")
    print(f"  {md_path}")


if __name__ == "__main__":
    main()
