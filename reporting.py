"""
Results tables, CSV output, per-point summaries.
"""
import numpy as np
import pandas as pd
from numpy.linalg import inv


def print_results_table(param_names, Ka_opt_M, Kd_opt_M, has_errors, cfg,
                        std_Ka_M=None, std_Kd_M=None):
    """Print fitted Ka/Kd table — used by both models."""
    ka_unit = f"{cfg.output_unit}⁻¹"
    kd_unit = cfg.output_unit
    header = (f"{'Parameter':>10} | {f'Fitted Kₐ ({ka_unit})':^22} | {f'Uncertainty (Kₐ)':^20} | "
              f"{f'Fitted Kₔ ({kd_unit})':^22} | {f'Uncertainty (Kₔ)':^20}")
    print(header)
    print("-" * len(header))

    for i, name in enumerate(param_names):
        Ka_out = Ka_opt_M[i] / cfg.scale_m_to_out
        Kd_out = Kd_opt_M[i] * cfg.scale_m_to_out
        if has_errors:
            std_Ka_out = std_Ka_M[i] / cfg.scale_m_to_out
            std_Kd_out = std_Kd_M[i] * cfg.scale_m_to_out
            print(f"{name:>10} | {Ka_out:^22.3e} | {f'± {1.96*std_Ka_out:.2e}':^20} | "
                  f"{Kd_out:^22.3e} | {f'± {1.96*std_Kd_out:.2e}':^20}")
        else:
            print(f"{name:>10} | {Ka_out:^22.3e} | {'N/A':^20} | "
                  f"{Kd_out:^22.3e} | {'N/A':^20}")
    print("-" * len(header), "\n")


def compute_uncertainties(fit, lnK_opt, Ka_opt_M, Kd_opt_M, ssr_history):
    """Compute standard errors from Jacobian — shared logic."""
    J = fit.jac
    N_obs = fit.fun.size
    p = len(lnK_opt)
    has_errors = False
    std_Ka_M = std_Kd_M = None
    if N_obs > p:
        rss = ssr_history[-1]
        sig2 = rss / (N_obs - p)
        try:
            cov_lnK = sig2 * inv(J.T @ J)
            std_lnK = np.sqrt(np.diag(cov_lnK))
            std_Ka_M = Ka_opt_M * std_lnK
            std_Kd_M = Kd_opt_M * std_lnK
            has_errors = True
        except np.linalg.LinAlgError:
            print("Warning: Could not compute uncertainties (Jacobian matrix is singular).")
    else:
        print("Warning: Not enough data points to compute uncertainties.")
    return has_errors, std_Ka_M, std_Kd_M


def save_kd_csv(param_names, Ka_opt_M, Kd_opt_M, has_errors, std_Ka_M, std_Kd_M, kd_csv, cfg):
    """Save Kd CSV — shared logic."""
    ka_unit = f"{cfg.output_unit}⁻¹"
    kd_unit = cfg.output_unit
    kd_records = []
    for i, name in enumerate(param_names):
        Ka_out = Ka_opt_M[i] / cfg.scale_m_to_out
        Kd_out = Kd_opt_M[i] * cfg.scale_m_to_out
        if has_errors:
            std_Ka_out = std_Ka_M[i] / cfg.scale_m_to_out
            std_Kd_out = std_Kd_M[i] * cfg.scale_m_to_out
            kd_records.append((name, Ka_out, std_Ka_out, Kd_out, std_Kd_out))
        else:
            kd_records.append((name, Ka_out, None, Kd_out, None))
    pd.DataFrame(
        kd_records,
        columns=[f"Param", f"Ka({ka_unit})", f"SE_Ka({ka_unit})", f"Kd({kd_unit})", f"SE_Kd({kd_unit})"]
    ).to_csv(kd_csv, index=False, float_format='%.6e')


def print_per_point_summary(df, L_totals_M, F_exps, L_free_list, F_calcs, num_species, cfg):
    """Print per-point summary table — shared logic."""
    rows = []
    for idx, (entry, L_tot_M, F_exp) in enumerate(zip(df['Entry'], L_totals_M, F_exps)):
        L_free_M = L_free_list[idx]
        F_calc = F_calcs[idx]
        ssr_i = np.sum((F_calc - F_exp) ** 2)
        row = {
            'Entry': entry,
            f'[L]tot({cfg.output_unit})': f"{L_tot_M * cfg.scale_m_to_out:.2f}",
            f'[L]free({cfg.output_unit})': f"{L_free_M * cfg.scale_m_to_out:.3f}",
            'SSR_i': f"{ssr_i:.2e}"
        }
        for j in range(num_species):
            row[f"Fexp_{j}"] = f"{F_exp[j]:.3f}"
            row[f"Fcalc_{j}"] = f"{F_calc[j]:.3f}"
        rows.append(row)

    df_report = pd.DataFrame(rows)
    print("--- Per-point Summary ---")
    print(df_report.to_string(index=False, max_colwidth=10))
    print()
