"""Run selected NSB models on every replicate at every temperature across buffers.

Uses the package REGISTRY and shared helpers from fitting.py.

Outputs to OUT_DIR:
  all_fits_long.csv   one row per (buffer, temp, rep, model, parameter)
  all_fits_wide.csv   one row per (buffer, temp, rep, model)
  report.md           per-buffer summary tables and BIC ranking

Usage:
  python -m scripts_binding.scripts.batch_kd_scan \\
      --out-dir /path/to/out \\
      --buffer AMAC:/path/to/amac_dir:7 \\
      --buffer EDDA:/path/to/edda_dir:5 \\
      --temps 5,10,15,20,25,30,35 \\
      --reps 1,2,3 \\
      --p-tot 1e-6 \\
      --models geometric_nonspecific,poisson_nonspecific,binomial_poisson_nonspecific,power_law_nonspecific
"""
import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from scripts_binding.models import REGISTRY


def _parse_buffer(spec):
    """NAME:PATH:S_SPECIFIC → (name, path, S)."""
    name, path, S = spec.split(":")
    return name, path, int(S)


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", required=True, help="Output directory for CSVs and report")
    p.add_argument("--buffer", action="append", required=True, type=_parse_buffer, dest="buffers",
                   help="Repeated. Format: NAME:PATH:S_SPECIFIC  (e.g. AMAC:/data/amac:7)")
    p.add_argument("--temps", default="5,10,15,20,25,30,35",
                   help="Comma-separated temperatures (C)")
    p.add_argument("--reps", default="1,2,3", help="Comma-separated replicate IDs")
    p.add_argument("--n-nsb", type=int, default=3, help="Max NSB stoichiometry beyond S")
    p.add_argument("--p-tot", type=float, default=1e-6, help="Total protein concentration (M)")
    p.add_argument("--models", default="geometric_nonspecific,poisson_nonspecific,binomial_poisson_nonspecific,power_law_nonspecific",
                   help="Comma-separated model names from REGISTRY")
    p.add_argument("--filename-fmt", default="{buffer}_{temp}C_{rep}.csv",
                   help="CSV filename template with {buffer}, {temp}, {rep}")
    return p.parse_args()


def load_csv(path):
    df = pd.read_csv(path)
    L_totals = df["Entry"].to_numpy()
    valid = L_totals > 0
    L_totals = L_totals[valid]
    F_cols = [c for c in df.columns if c.startswith("I")]
    F_arr = df.loc[valid, F_cols].to_numpy()
    return L_totals, [F_arr[i, :] for i in range(F_arr.shape[0])]


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


def collect_run(buffer_name, root, S, N, temps, reps, filename_fmt, p_tot, models_to_run):
    rows_long, rows_wide = [], []
    for T in temps:
        for R in reps:
            csv_path = os.path.join(root, filename_fmt.format(buffer=buffer_name, temp=T, rep=R))
            if not os.path.exists(csv_path):
                continue
            L_totals, F_exps = load_csv(csv_path)

            for model_name in models_to_run:
                model = REGISTRY[model_name]
                res, ssr, bic_v = fit_one(model, L_totals, F_exps, p_tot, S, N)
                labels = model.param_labels(S)
                values = []
                for lbl, raw in zip(labels, res.x):
                    values.append(raw if lbl == "gamma" else float(np.exp(raw)))

                for lbl, val in zip(labels, values):
                    Kd_uM = (1e6 / val) if (lbl != "gamma" and val > 0) else np.nan
                    rows_long.append({
                        "buffer": buffer_name, "temp": T, "rep": R,
                        "model": model_name, "param": lbl,
                        "value": val, "Kd_uM": Kd_uM,
                    })

                wide = {
                    "buffer": buffer_name, "temp": T, "rep": R,
                    "model": model_name, "S": S, "N": N,
                    "n_params": model.n_params(S),
                    "ssr": ssr, "bic": bic_v, "converged": bool(res.success),
                }
                for lbl, val in zip(labels, values):
                    if lbl == "gamma":
                        wide[lbl] = val
                    else:
                        wide[f"{lbl}_Ka"] = val
                        wide[f"{lbl}_Kd_uM"] = (1e6 / val) if val > 0 else np.nan
                rows_wide.append(wide)
    return rows_long, rows_wide


def write_markdown(out_dir, df_wide, models_to_run, n_nsb):
    lines = ["# NSB Model Comparison\n",
             "Mean Kd values across replicates per temperature, per buffer, per model.\n",
             "Models compared:"]
    for m in models_to_run:
        lines.append(f"- **{m}**")
    lines.append(f"\nN_NSB (max NSB stoichiometry beyond S) = {n_nsb}.\n")

    # Coalesce the nonspecific-amplitude column across models: most use Kn,
    # power_law_nonspecific uses beta. The numerical value of 1/beta is the Kd
    # at the first NSB step (k=1), directly comparable to 1/Kn from the other
    # models since k^gamma = 1 at k=1.
    nsb_candidates = ["Kn_Kd_uM", "beta_Kd_uM"]
    df_wide = df_wide.copy()
    df_wide["Kd_NSB_uM"] = np.nan
    for col in nsb_candidates:
        if col in df_wide.columns:
            df_wide["Kd_NSB_uM"] = df_wide["Kd_NSB_uM"].fillna(df_wide[col])

    for buf, dfb in df_wide.groupby("buffer"):
        lines.append(f"\n## {buf}\n")

        if dfb["Kd_NSB_uM"].notna().any():
            mean_kd_n = dfb.groupby(["temp", "model"])["Kd_NSB_uM"].mean().unstack("model")
            lines.append("### Mean Kd_NSB (µM) by temperature\n")
            lines.append("(for power_law_nonspecific, reported value is 1/beta = Kd at first NSB step)\n")
            lines.append(mean_kd_n.round(1).to_markdown())
            lines.append("")

        mean_bic = dfb.groupby(["temp", "model"])["bic"].mean().unstack("model")
        lines.append("### Mean BIC by temperature (lower = better)\n")
        lines.append(mean_bic.round(1).to_markdown())
        lines.append("")

        d25 = dfb[dfb["temp"] == 25]
        ks_cols = sorted([c for c in d25.columns if c.startswith("Ks_") and c.endswith("_Kd_uM")])
        if ks_cols:
            ks_summary = d25.groupby("model")[ks_cols].mean().round(1)
            lines.append("### Mean Ks_i Kd (µM) at 25 °C\n")
            lines.append(ks_summary.to_markdown())
            lines.append("")

        winners = dfb.loc[dfb.groupby(["temp", "rep"])["bic"].idxmin(),
                          ["temp", "rep", "model", "bic"]].reset_index(drop=True)
        win_counts = winners["model"].value_counts()
        lines.append("### BIC winner counts (across temp × rep)\n")
        for m, c in win_counts.items():
            lines.append(f"- **{m}**: {c} / {len(winners)}")
        lines.append("")

    md_path = os.path.join(out_dir, "report.md")
    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    return md_path


def main():
    args = _parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    temps = [int(t) for t in args.temps.split(",")]
    reps = [int(r) for r in args.reps.split(",")]
    models_to_run = [m.strip() for m in args.models.split(",")]

    for m in models_to_run:
        if m not in REGISTRY:
            sys.exit(f"Unknown model: {m}. Available: {sorted(REGISTRY)}")

    all_long, all_wide = [], []
    for buf_name, root, S in args.buffers:
        print(f"\n=== {buf_name} (S={S}, root={root}) ===")
        long, wide = collect_run(buf_name, root, S, args.n_nsb, temps, reps,
                                 args.filename_fmt, args.p_tot, models_to_run)
        print(f"  Collected {len(wide)} fits")
        all_long.extend(long)
        all_wide.extend(wide)

    df_long = pd.DataFrame(all_long)
    df_wide = pd.DataFrame(all_wide)
    df_long.to_csv(os.path.join(args.out_dir, "all_fits_long.csv"), index=False)
    df_wide.to_csv(os.path.join(args.out_dir, "all_fits_wide.csv"), index=False)
    md_path = write_markdown(args.out_dir, df_wide, models_to_run, args.n_nsb)
    print(f"\nWrote {len(df_wide)} fits to:")
    print(f"  {os.path.join(args.out_dir, 'all_fits_long.csv')}")
    print(f"  {os.path.join(args.out_dir, 'all_fits_wide.csv')}")
    print(f"  {md_path}")


if __name__ == "__main__":
    main()
