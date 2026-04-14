"""
Multi-replicate summary + cross-model BIC/AIC comparison.
"""
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from .models.specific import compute_F_calc_specific, solve_L_free_specific
from .models.nonspecific import (
    calculate_fractions_model,
    compute_deconvolution_weights,
    deconvolve_fractions,
    solve_L_free,
)
from .plotting import (
    safe_savefig,
    plot_summary_fit,
    plot_deconv_stacked,
)


def compute_model_grid_nonspecific(L_grid_M, lnK_opt, S_eff, N_eff, p_total_m):
    F_grid = []
    for L in L_grid_M:
        Lf = solve_L_free(L, p_total_m, lnK_opt, S_eff, N_eff)
        F_grid.append(calculate_fractions_model(Lf, lnK_opt, S_eff, N_eff))
    return np.array(F_grid)


def compute_model_grid_specific(L_grid_M, lnK_opt, p_total_m):
    K_opt = np.exp(lnK_opt)
    F_grid = []
    for L in L_grid_M:
        Lf = solve_L_free_specific(L, p_total_m, K_opt)
        F_grid.append(compute_F_calc_specific(Lf, K_opt))
    return np.array(F_grid)


def build_summary(all_L_tot, all_F_exp, all_Kd, all_stems, num_species_list,
                  all_lnK, model_grid_fn, model_grid_extra_args_list,
                  out_dir, label, n_params, cfg, n_specific=None):
    """
    Build multi-file summary: mean/std of experimental data, mean model curves,
    summary fit plot, and summary Kd CSV.
    """
    ref_L = all_L_tot[0]
    num_species = max(num_species_list)

    # Interpolate experimental F_exps onto a common ligand grid
    F_exp_interp_list = []
    for L_tot, F_exp in zip(all_L_tot, all_F_exp):
        if F_exp.shape[1] < num_species:
            pad = num_species - F_exp.shape[1]
            F_exp = np.pad(F_exp, ((0, 0), (0, pad)), 'constant')
        F_interp = np.zeros((len(ref_L), num_species))
        for j in range(num_species):
            mask = ~np.isnan(L_tot) & ~np.isnan(F_exp[:, j])
            x, y = np.array(L_tot[mask]), np.array(F_exp[:, j][mask])
            if len(x) == 0:
                F_interp[:, j] = np.nan
                continue
            order = np.argsort(x)
            x, y = x[order], y[order]
            f = interp1d(x, y, bounds_error=False, fill_value=np.nan)
            F_interp[:, j] = f(ref_L)
        F_exp_interp_list.append(F_interp)

    F_arr = np.stack(F_exp_interp_list, axis=0)
    F_exp_mean = np.nanmean(F_arr, axis=0)
    F_exp_std = np.nanstd(F_arr, axis=0)

    # Generate model curves for each run on a common grid
    L_grid_M = np.linspace(ref_L.min(), ref_L.max(), 300)
    calc_runs = []
    for lnK_opt, extra_args in zip(all_lnK, model_grid_extra_args_list):
        F_grid_run = model_grid_fn(L_grid_M, lnK_opt, *extra_args)
        if F_grid_run.shape[1] < num_species:
            pad = num_species - F_grid_run.shape[1]
            F_grid_run = np.pad(F_grid_run, ((0, 0), (0, pad)), 'constant')
        calc_runs.append(F_grid_run)
    calc_arr = np.stack(calc_runs, axis=0)
    F_calc_mean = np.nanmean(calc_arr, axis=0)
    F_calc_std = np.nanstd(calc_arr, axis=0)

    summary_fit_svg = os.path.join(
        out_dir,
        cfg.csv_name_wildcard.replace('*', '').replace('.csv', '') + 'fit_summary.svg'
    )
    plot_summary_fit(ref_L, F_exp_mean, F_exp_std, L_grid_M, F_calc_mean, F_calc_std,
                     summary_fit_svg, num_species, cfg, n_specific=n_specific)
    print(f"[{label} Summary] Wrote summary fit plot to {summary_fit_svg}")

    # Mean Kd (in output units) — pad shorter arrays with NaN
    max_kd_len = max(len(kd) for kd in all_Kd)
    n_params = max_kd_len
    kd_padded = []
    for kd in all_Kd:
        if len(kd) < max_kd_len:
            kd_padded.append(np.pad(kd, (0, max_kd_len - len(kd)), constant_values=np.nan))
        else:
            kd_padded.append(kd)
    kd_values = np.stack(kd_padded, axis=1)
    mean_Kd_out = np.nanmean(kd_values, axis=1)

    # Write summary Kd CSV
    kd_unit = cfg.output_unit
    summary = {
        "Result": [f"K{i+1}" for i in range(n_params)] if label == "Specific" else
                  [f"Kₙ"] + [f"Kₛ{i+1}" for i in range(n_params - 1)],
        f"Mean_Kd_({kd_unit})": mean_Kd_out,
        f"Std_Kd_({kd_unit})": np.nanstd(kd_values, axis=1),
    }
    for col_idx, stem in enumerate(all_stems):
        summary[stem] = kd_values[:, col_idx]
    summary_df = pd.DataFrame(summary)
    summary_name = cfg.csv_name_wildcard.replace('*', '').replace('.csv', '') + 'stat_summary.csv'
    summary_file = os.path.join(out_dir, summary_name)
    summary_df.to_csv(summary_file, index=False, float_format='%.6e')
    print(f"[{label} Summary] Wrote summary Kd table to {summary_file}")

    return ref_L, F_exp_mean, F_exp_std, kd_values, mean_Kd_out, num_species


def build_nonspecific_deconv_summary(
    ref_L, F_exp_mean, F_exp_std, mean_Kd_out, num_species,
    ns_all_S, nonspecific_dir, cfg
):
    """Build summary deconvolution plot and CSV for nonspecific model."""
    summary_S = max(set(ns_all_S), key=ns_all_S.count)
    summary_N = num_species - summary_S - 1
    mean_Kd_trunc = mean_Kd_out[:summary_S + 1]
    mean_Ka_per_out = 1.0 / mean_Kd_trunc
    mean_Ka_M = mean_Ka_per_out * cfg.scale_m_to_out
    lnK_mean = np.log(mean_Ka_M)
    F_calc_mean_ref = []
    for L in ref_L:
        Lf = solve_L_free(L, cfg.p_total_m, lnK_mean, summary_S, summary_N)
        F_calc_mean_ref.append(calculate_fractions_model(Lf, lnK_mean, summary_S, summary_N))
    F_calc_mean_ref = np.array(F_calc_mean_ref)
    if F_calc_mean_ref.shape[1] < num_species:
        pad = num_species - F_calc_mean_ref.shape[1]
        F_calc_mean_ref = np.pad(F_calc_mean_ref, ((0, 0), (0, pad)), 'constant')

    weights_ij = compute_deconvolution_weights(np.exp(lnK_mean)[0], np.exp(lnK_mean)[1:], summary_S, summary_N)
    contrib_all = np.zeros((len(ref_L), num_species, summary_S + 1))
    frac_within_all = np.zeros((len(ref_L), num_species, summary_S + 1))
    for idx in range(len(ref_L)):
        frac_within, contrib = deconvolve_fractions(F_calc_mean_ref[idx], weights_ij, summary_S)
        contrib_all[idx] = contrib
        frac_within_all[idx] = frac_within

    summary_deconv_svg = os.path.join(
        nonspecific_dir,
        cfg.csv_name_wildcard.replace('*', '').replace('.csv', '') + 'deconv_summary.svg'
    )
    fig_summary = plot_deconv_stacked(
        ref_L * cfg.scale_m_to_out,
        contrib_all,
        summary_S,
        summary_N,
        "Deconvoluted fraction of apparent",
        cfg,
        outline_totals=F_exp_mean,
        outline_err=F_exp_std,
        outline_label="F_exp mean"
    )
    safe_savefig(fig_summary, summary_deconv_svg, cfg.max_image_dim)
    if cfg.show_plots:
        plt.show()
    else:
        plt.close(fig_summary)
    print(f"[Nonspecific Summary] Wrote summary deconvolution plot to {summary_deconv_svg}")

    # Summary deconvolution CSV
    summary_deconv_csv = os.path.join(
        nonspecific_dir,
        cfg.csv_name_wildcard.replace('*', '').replace('.csv', '') + 'deconv_summary.csv'
    )
    csv_rows = []
    for idx in range(len(ref_L)):
        L_out = ref_L[idx] * cfg.scale_m_to_out
        for i in range(num_species):
            max_j = min(i, summary_S)
            for j in range(max_j + 1):
                csv_rows.append({
                    f"L_tot({cfg.output_unit})": L_out,
                    "i_total": i,
                    "j_specific": j,
                    "m_nonspecific": i - j,
                    "F_exp_mean": F_exp_mean[idx][i] if i < F_exp_mean.shape[1] else np.nan,
                    "F_exp_std": F_exp_std[idx][i] if i < F_exp_std.shape[1] else np.nan,
                    "F_calc_mean": F_calc_mean_ref[idx][i],
                    "fraction_within_i": frac_within_all[idx][i, j],
                    "fraction_total": contrib_all[idx][i, j],
                })
    pd.DataFrame(csv_rows).to_csv(summary_deconv_csv, index=False)
    print(f"[Nonspecific Summary] Wrote summary deconvolution CSV to {summary_deconv_csv}")

    return summary_S


def compare_models_bic_aic(sp_data_list, ns_data_list, out_dir):
    """
    Cross-model comparison using BIC and AIC.
    """
    def _info_criteria(ssr, n, k):
        bic = n * np.log(ssr / n) + k * np.log(n)
        aic = n * np.log(ssr / n) + 2 * k
        aicc = aic + 2 * k * (k + 1) / (n - k - 1) if n > k + 1 else np.inf
        return bic, aic, aicc

    csv_rows = []
    all_dbic = []
    all_daic = []

    print("\n" + "=" * 70)
    print("CROSS-MODEL COMPARISON: Specific vs Mixed (Nonspecific)")
    print("=" * 70)

    for sp, ns in zip(sp_data_list, ns_data_list):
        stem = sp["stem"]
        sp_bic, sp_aic, sp_aicc = _info_criteria(sp["SSR"], sp["n_obs"], sp["n_params"])
        ns_bic, ns_aic, ns_aicc = _info_criteria(ns["SSR"], ns["n_obs"], ns["n_params"])

        dbic = sp_bic - ns_bic
        daic = sp_aic - ns_aic
        all_dbic.append(dbic)
        all_daic.append(daic)

        print(f"\n=== Model Comparison ({stem}) ===")
        print(f"{'':>20} {'Specific':>14} {'Nonspecific':>14}")
        print(f"{'Parameters':>20} {sp['n_params']:>14d} {ns['n_params']:>14d}")
        print(f"{'SSR':>20} {sp['SSR']:>14.5f} {ns['SSR']:>14.5f}")
        print(f"{'n_obs':>20} {sp['n_obs']:>14d} {ns['n_obs']:>14d}")
        print(f"{'BIC':>20} {sp_bic:>14.2f} {ns_bic:>14.2f}")
        print(f"{'AIC':>20} {sp_aic:>14.2f} {ns_aic:>14.2f}")
        print(f"{'AICc':>20} {sp_aicc:>14.2f} {ns_aicc:>14.2f}")
        print(f"{'ΔBIC (sp−ns)':>20} {dbic:>14.2f}   {'(>0 favours nonspecific)'}")
        print(f"{'ΔAIC (sp−ns)':>20} {daic:>14.2f}")

        for model_label, d in [("Specific", sp), ("Nonspecific", ns)]:
            bic_v, aic_v, aicc_v = _info_criteria(d["SSR"], d["n_obs"], d["n_params"])
            csv_rows.append({
                "File": stem,
                "Model": model_label,
                "n_params": d["n_params"],
                "n_obs": d["n_obs"],
                "SSR": d["SSR"],
                "BIC": bic_v,
                "AIC": aic_v,
                "AICc": aicc_v,
                "dBIC_sp_minus_ns": dbic,
                "dAIC_sp_minus_ns": daic,
            })

    # Overall summary
    print("\n" + "-" * 70)
    print("OVERALL SUMMARY")
    print("-" * 70)
    n_files = len(all_dbic)
    bic_favour_ns = sum(1 for d in all_dbic if d > 0)
    aic_favour_ns = sum(1 for d in all_daic if d > 0)
    mean_dbic = np.mean(all_dbic)
    mean_daic = np.mean(all_daic)

    print(f"Files analysed: {n_files}")
    print(f"BIC favours nonspecific: {bic_favour_ns}/{n_files}")
    print(f"AIC favours nonspecific: {aic_favour_ns}/{n_files}")
    print(f"Mean ΔBIC (sp−ns): {mean_dbic:.2f}")
    print(f"Mean ΔAIC (sp−ns): {mean_daic:.2f}")

    abs_dbic = abs(mean_dbic)
    if abs_dbic > 10:
        strength = "very strong"
    elif abs_dbic > 6:
        strength = "strong"
    elif abs_dbic > 2:
        strength = "positive"
    else:
        strength = "negligible"
    favoured = "nonspecific (mixed)" if mean_dbic > 0 else "specific"
    print(f"Kass-Raftery interpretation: {strength} evidence in favour of {favoured} model")
    print("-" * 70)

    csv_path = os.path.join(out_dir, "model_comparison.csv")
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False, float_format="%.6e")
    print(f"\nSaved model comparison to: {csv_path}")
