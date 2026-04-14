"""
Fitting orchestration: process_file_specific, process_file_nonspecific, auto_select_S.
"""
import numpy as np
import pandas as pd
import re
import os
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

from .models.specific import (
    compute_F_calc_specific,
    solve_L_free_specific,
    residuals_specific,
)
from .models.nonspecific import (
    calculate_fractions_model,
    compute_deconvolution_weights,
    deconvolve_fractions,
    solve_L_free,
    residuals,
    debug_validate_point,
)
from .plotting import (
    safe_savefig,
    plot_fit_curves,
    plot_convergence,
    plot_deconv_stacked,
)
from .reporting import (
    print_results_table,
    compute_uncertainties,
    save_kd_csv,
    print_per_point_summary,
)


def parse_ligand_conc(entry, scale_l_in_to_m):
    """Extracts numeric concentration from entry string and converts to Molar."""
    if pd.isna(entry):
        return np.nan
    if isinstance(entry, (int, float)):
        return float(entry) * scale_l_in_to_m
    try:
        match = re.search(r'[\d\.]+', str(entry))
        if match:
            return float(match.group(0)) * scale_l_in_to_m
    except (ValueError, TypeError):
        pass
    return np.nan


def fit_nonspecific_quick(L_totals_M, F_exps, P_tot_M, S_trial, N_trial):
    """
    Lightweight nonspecific fit for BIC comparison.
    Runs least_squares silently and returns fit statistics only.
    """
    ssr_history = []

    num_species_model = S_trial + N_trial + 1
    F_exps_padded = F_exps
    if F_exps.shape[1] < num_species_model:
        F_exps_padded = np.pad(F_exps, ((0, 0), (0, num_species_model - F_exps.shape[1])), 'constant')

    lnK0 = np.zeros(S_trial + 1)
    try:
        fit = least_squares(
            residuals,
            lnK0,
            args=(L_totals_M, P_tot_M, F_exps_padded, S_trial, N_trial, ssr_history),
            verbose=0
        )
        rss = float(np.dot(fit.fun, fit.fun))
    except Exception as e:
        print(f"  [Auto-S] S={S_trial} failed: {e}")
        return None

    n_obs = fit.fun.size
    n_params = len(fit.x)
    bic = n_obs * np.log(rss / n_obs) + n_params * np.log(n_obs)

    return {
        "S": S_trial,
        "N": N_trial,
        "n_params": n_params,
        "n_obs": n_obs,
        "SSR": rss,
        "BIC": bic,
        "lnK_opt": fit.x,
    }


def auto_select_S(data_path, cfg):
    """
    Scan S = 1..max_i, fit each, and return the S with lowest BIC.
    Prints a comparison table.
    """
    df = pd.read_csv(data_path).dropna(subset=['Entry']).fillna(0)
    L_totals_M = df.iloc[:, 0].apply(lambda e: parse_ligand_conc(e, cfg.scale_l_in_to_m)).values

    I_cols = sorted([c for c in df.columns if c.startswith('I')], key=lambda c: int(c[1:]))
    max_i = len(I_cols) - 1

    I_vals = df[I_cols].values
    F_exps = (I_vals.T / I_vals.sum(axis=1)).T

    print(f"\n{'='*60}")
    print(f"[Auto-S] Scanning S = 0..{max_i} for: {os.path.basename(data_path)}")
    print(f"{'='*60}")

    results = []
    for S_trial in range(0, max_i + 1):
        N_trial = max_i - S_trial
        info = fit_nonspecific_quick(L_totals_M, F_exps, cfg.p_total_m, S_trial, N_trial)
        if info is not None:
            results.append(info)

    if not results:
        print("[Auto-S] All fits failed. Falling back to S=1.")
        return 1

    header = f"  {'S':>3} {'N':>3} {'params':>6} {'SSR':>12} {'BIC':>12}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    best = min(results, key=lambda r: r["BIC"])
    for r in results:
        marker = " <-- best" if r["S"] == best["S"] else ""
        print(f"  {r['S']:>3} {r['N']:>3} {r['n_params']:>6} {r['SSR']:>12.4e} {r['BIC']:>12.2f}{marker}")

    print(f"\n[Auto-S] Selected S = {best['S']} (BIC = {best['BIC']:.2f})\n")
    return best["S"]


def process_file_specific(data_path, out_dir, cfg):
    """
    Run specific-only stepwise binding fit on a single CSV file.
    """
    ssr_history = []

    df = pd.read_csv(data_path).dropna(subset=['Entry']).fillna(0)
    L_totals_M = df.iloc[:, 0].apply(lambda e: parse_ligand_conc(e, cfg.scale_l_in_to_m)).values

    I_cols = sorted([c for c in df.columns if c.startswith('I')], key=lambda c: int(c[1:]))
    n_steps = len(I_cols) - 1
    num_species = len(I_cols)

    print(f"\n=== [Specific] Processing: {data_path} ===")
    print(f"Model Parameters: {n_steps} stepwise binding constant(s) K₁…K_{n_steps}")
    print(f"Fitting {n_steps} constants: K_1...K_{n_steps}\n")

    I_vals = df[I_cols].values
    F_exps = (I_vals.T / I_vals.sum(axis=1)).T

    # === Trim trailing species with negligible population ===
    trimmed = []
    while n_steps > 1:
        max_frac = F_exps[:, n_steps].max()
        if max_frac < cfg.min_species_frac:
            trimmed.append(f"I{n_steps} (max={max_frac:.4f})")
            F_exps = F_exps[:, :n_steps]
            I_cols = I_cols[:n_steps + 1]
            n_steps -= 1
            num_species -= 1
        else:
            break
    if trimmed:
        print(f"  [Trimmed] Dropped species with max frac < {cfg.min_species_frac}: {', '.join(trimmed)}")
        print(f"  [Trimmed] Fitting {n_steps} steps: K1…K{n_steps}")
    F_exps = (F_exps.T / F_exps.sum(axis=1)).T

    # === Fitting ===
    lnK0 = np.zeros(n_steps)

    print("--- Starting Optimization (Specific) ---")
    fit = least_squares(
        residuals_specific,
        lnK0,
        args=(L_totals_M, cfg.p_total_m, F_exps, ssr_history),
        verbose=2
    )
    print("--- Optimization Finished (Specific) ---\n")

    lnK_opt = fit.x
    Ka_opt_M = np.exp(lnK_opt)
    Kd_opt_M = 1.0 / Ka_opt_M

    ssr_final = float(np.dot(fit.fun, fit.fun))
    n_obs = fit.fun.size

    ssr_history_copy = list(ssr_history)

    # === Uncertainties ===
    has_errors, std_Ka_M, std_Kd_M = compute_uncertainties(
        fit, lnK_opt, Ka_opt_M, Kd_opt_M, ssr_history_copy)

    # === Print results ===
    param_names = [f"K{i+1}" for i in range(n_steps)]
    print_results_table(param_names, Ka_opt_M, Kd_opt_M, has_errors, cfg, std_Ka_M, std_Kd_M)

    # === Per-point summary ===
    K_opt = Ka_opt_M
    L_free_list = np.array([
        solve_L_free_specific(L_tot_M, cfg.p_total_m, K_opt)
        for L_tot_M in L_totals_M
    ])
    F_calcs = np.vstack([
        compute_F_calc_specific(L_free_M, K_opt)
        for L_free_M in L_free_list
    ])

    print_per_point_summary(df, L_totals_M, F_exps, L_free_list, F_calcs, num_species, cfg)

    # === Output paths ===
    stem = os.path.splitext(os.path.basename(data_path))[0]
    fit_svg = os.path.join(out_dir, f"{stem}_fit.svg")
    conv_svg = os.path.join(out_dir, f"{stem}_conv.svg")
    kd_csv = os.path.join(out_dir, f"{stem}_kd.csv")

    # === Plotting ===
    L_grid_M = np.linspace(L_totals_M.min(), L_totals_M.max(), 300)
    F_grid = np.vstack([
        compute_F_calc_specific(solve_L_free_specific(L, cfg.p_total_m, K_opt), K_opt)
        for L in L_grid_M
    ])

    plot_fit_curves(L_totals_M, F_exps, L_grid_M, F_grid, num_species,
                    'Specific Model: Global Fit', fit_svg, cfg)
    plot_convergence(ssr_history_copy, conv_svg, cfg)

    # === Save Kd CSV ===
    save_kd_csv(param_names, Ka_opt_M, Kd_opt_M, has_errors, std_Ka_M, std_Kd_M, kd_csv, cfg)

    return {
        "L_totals_M": L_totals_M,
        "F_exps": F_exps,
        "Kd_out": Kd_opt_M * cfg.scale_m_to_out,
        "num_species": num_species,
        "n_steps": n_steps,
        "stem": stem,
        "lnK_opt": lnK_opt,
        "SSR": ssr_final,
        "n_obs": n_obs,
        "n_params": len(lnK_opt),
    }


def process_file_nonspecific(data_path, out_dir, cfg, S_override=None):
    """
    Run mixed specific/non-specific binding fit on a single CSV file.
    """
    ssr_history = []

    df = pd.read_csv(data_path).dropna(subset=['Entry']).fillna(0)
    L_totals_M = df.iloc[:, 0].apply(lambda e: parse_ligand_conc(e, cfg.scale_l_in_to_m)).values

    I_cols = sorted([c for c in df.columns if c.startswith('I')], key=lambda c: int(c[1:]))
    max_i = len(I_cols) - 1
    S_use = S_override if S_override is not None else cfg.s
    S_eff = S_use
    if S_eff > max_i:
        if cfg.auto_adjust_s:
            print(f"[Warning] S={S_use} exceeds available steps (I0..I{max_i}). Clamping S to {max_i}.")
            S_eff = max_i
        else:
            raise ValueError(f"S={S_use} exceeds available steps (I0..I{max_i}).")
    if cfg.n_override is not None:
        N_eff = cfg.n_override
        if S_eff + N_eff != max_i:
            print(f"[Warning] S+N={S_eff + N_eff} does not match max_i={max_i}.")
    else:
        N_eff = max_i - S_eff
    if N_eff < 0:
        raise ValueError(f"Computed N={N_eff} is negative. Check S or data columns.")

    print(f"\n=== [Nonspecific] Processing: {data_path} ===")
    print(f"Model Parameters: {S_eff} specific site(s), {N_eff} non-specific site(s).")
    print(f"Fitting {S_eff+1} constants: K_n, K_s1...K_s{S_eff}\n")

    I_vals = df[I_cols].values
    F_exps = (I_vals.T / I_vals.sum(axis=1)).T

    num_species_model = S_eff + N_eff + 1
    if F_exps.shape[1] < num_species_model:
        F_exps = np.pad(F_exps, ((0, 0), (0, num_species_model - F_exps.shape[1])), 'constant')

    # === Fitting ===
    lnK0 = np.zeros(S_eff + 1)

    print("--- Starting Optimization (Nonspecific) ---")
    fit = least_squares(
        residuals,
        lnK0,
        args=(L_totals_M, cfg.p_total_m, F_exps, S_eff, N_eff, ssr_history),
        verbose=2
    )
    print("--- Optimization Finished (Nonspecific) ---\n")

    lnK_opt = fit.x
    Ka_opt_M = np.exp(lnK_opt)
    Kd_opt_M = 1.0 / Ka_opt_M

    ssr_final = float(np.dot(fit.fun, fit.fun))
    n_obs = fit.fun.size

    ssr_history_copy = list(ssr_history)

    # === Uncertainties ===
    has_errors, std_Ka_M, std_Kd_M = compute_uncertainties(
        fit, lnK_opt, Ka_opt_M, Kd_opt_M, ssr_history_copy)

    # === Print results ===
    param_names = [f"Kₙ"] + [f"Kₛ{i+1}" for i in range(S_eff)]
    print_results_table(param_names, Ka_opt_M, Kd_opt_M, has_errors, cfg, std_Ka_M, std_Kd_M)

    # === Per-point summary ===
    L_free_list = np.array([
        solve_L_free(L_tot_M, cfg.p_total_m, lnK_opt, S_eff, N_eff)
        for L_tot_M in L_totals_M
    ])
    F_calcs = np.vstack([
        calculate_fractions_model(L_free_M, lnK_opt, S_eff, N_eff)
        for L_free_M in L_free_list
    ])

    if cfg.debug_validate and len(L_totals_M) > 0:
        if cfg.debug_ligand_conc is not None:
            L_out_all = L_totals_M * cfg.scale_m_to_out
            idx = int(np.argmin(np.abs(L_out_all - cfg.debug_ligand_conc)))
        else:
            idx = min(cfg.debug_index, len(L_totals_M) - 1)
        debug_validate_point(idx, L_totals_M[idx], L_free_list[idx], lnK_opt, F_calcs[idx], S_eff, N_eff, cfg)

    print_per_point_summary(df, L_totals_M, F_exps, L_free_list, F_calcs, num_species_model, cfg)

    # === Output paths ===
    stem = os.path.splitext(os.path.basename(data_path))[0]
    fit_svg = os.path.join(out_dir, f"{stem}_fit.svg")
    conv_svg = os.path.join(out_dir, f"{stem}_conv.svg")
    kd_csv = os.path.join(out_dir, f"{stem}_kd.csv")
    deconv_csv = cfg.deconv_csv_path
    if deconv_csv is None:
        deconv_csv = os.path.join(out_dir, f"{stem}_deconv.csv")

    # === Plotting ===
    L_grid_M = np.linspace(L_totals_M.min(), L_totals_M.max(), 300)
    F_grid = np.vstack([
        calculate_fractions_model(solve_L_free(L, cfg.p_total_m, lnK_opt, S_eff, N_eff), lnK_opt, S_eff, N_eff)
        for L in L_grid_M
    ])

    plot_fit_curves(L_totals_M, F_exps, L_grid_M, F_grid, num_species_model,
                    'Nonspecific Model: Global Fit', fit_svg, cfg, n_specific=S_eff)
    plot_convergence(ssr_history_copy, conv_svg, cfg)

    # === Save Kd CSV ===
    save_kd_csv(param_names, Ka_opt_M, Kd_opt_M, has_errors, std_Ka_M, std_Kd_M, kd_csv, cfg)

    # === Deconvolution ===
    if cfg.deconv_enable:
        Kn_opt = Ka_opt_M[0]
        Ks_opt = Ka_opt_M[1:]
        weights_ij = compute_deconvolution_weights(Kn_opt, Ks_opt, S_eff, N_eff)

        if cfg.deconv_use_grid:
            L_vals_M = np.linspace(L_totals_M.min(), L_totals_M.max(), cfg.deconv_grid_points)
            L_free_vals = np.array([
                solve_L_free(L, cfg.p_total_m, lnK_opt, S_eff, N_eff)
                for L in L_vals_M
            ])
            F_source = np.vstack([
                calculate_fractions_model(Lf, lnK_opt, S_eff, N_eff)
                for Lf in L_free_vals
            ])
            source_label = "calc (grid)"
            source_key = "calc_grid"
            entries = [f"{v * cfg.scale_m_to_out:.3g}" for v in L_vals_M]
        else:
            L_vals_M = L_totals_M
            entries = df['Entry'].tolist()
            if cfg.deconv_source.lower() == "exp":
                F_source = F_exps
                source_label = "exp"
                source_key = "exp"
            else:
                F_source = F_calcs
                source_label = "calc"
                source_key = "calc"

        n_points = len(L_vals_M)
        num_species = S_eff + N_eff + 1
        frac_within_all = np.zeros((n_points, num_species, S_eff + 1))
        contrib_all = np.zeros((n_points, num_species, S_eff + 1))
        for idx in range(n_points):
            frac_within, contrib = deconvolve_fractions(F_source[idx], weights_ij, S_eff)
            frac_within_all[idx] = frac_within
            contrib_all[idx] = contrib

        if deconv_csv:
            rows = []
            for idx in range(n_points):
                L_out = L_vals_M[idx] * cfg.scale_m_to_out
                for i in range(num_species):
                    max_j = min(i, S_eff)
                    for j in range(max_j + 1):
                        rows.append({
                            "Entry": entries[idx],
                            f"L_tot({cfg.output_unit})": L_out,
                            "i_total": i,
                            "j_specific": j,
                            "m_nonspecific": i - j,
                            f"F_{source_key}": F_source[idx][i],
                            "fraction_within_i": frac_within_all[idx][i, j],
                            "fraction_total": contrib_all[idx][i, j],
                        })
            pd.DataFrame(rows).to_csv(deconv_csv, index=False)
            print(f"Saved deconvolution table to: {deconv_csv}")

        if cfg.report_ligand_conc:
            L_out_all = L_vals_M * cfg.scale_m_to_out
            print("\n--- Deconvolution report ---")
            for target in cfg.report_ligand_conc:
                idx = int(np.argmin(np.abs(L_out_all - target)))
                print(f"\nTarget {target} {cfg.output_unit} -> using {L_out_all[idx]:.3g} {cfg.output_unit} (Entry={entries[idx]})")
                for i in range(1, num_species):
                    max_j = min(i, S_eff)
                    parts = []
                    for j in range(max_j + 1):
                        pct = 100.0 * frac_within_all[idx][i, j]
                        parts.append(f"{j} spec + {i - j} non: {pct:.1f}%")
                    print(f"  I{i}: " + "; ".join(parts))

        title_prefix = "Deconvoluted fraction of apparent"
        L_vals_out = L_vals_M * cfg.scale_m_to_out
        outline_totals = None
        if not cfg.deconv_use_grid:
            outline_totals = F_exps
        fig3 = plot_deconv_stacked(
            L_vals_out,
            contrib_all,
            S_eff,
            N_eff,
            title_prefix,
            cfg,
            outline_totals=outline_totals,
            outline_label="F_exp total"
        )
        if cfg.save_plots:
            safe_savefig(fig3, os.path.join(out_dir, f"{stem}_deconv.svg"), cfg.max_image_dim)
        if cfg.show_plots:
            plt.show()
        else:
            plt.close(fig3)

    return {
        "L_totals_M": L_totals_M,
        "F_exps": F_exps,
        "Kd_out": Kd_opt_M * cfg.scale_m_to_out,
        "S_eff": S_eff,
        "N_eff": N_eff,
        "num_species_model": num_species_model,
        "stem": stem,
        "lnK_opt": lnK_opt,
        "SSR": ssr_final,
        "n_obs": n_obs,
        "n_params": len(lnK_opt),
    }
