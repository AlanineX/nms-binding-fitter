"""Multi-replicate summary and cross-model BIC/AIC comparison."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from .models import REGISTRY, geometric_nonspecific
from .plotting import safe_savefig, plot_summary_fit, plot_deconv_stacked


def compute_model_grid(L_grid_M, lnK_opt, S_eff, N_eff, p_total_m, model_name):
    """Compute F_grid from any model. Thin wrapper over model.mole_fractions."""
    model = REGISTRY[model_name]
    F_grid = []
    for L in L_grid_M:
        Lf = model.free_ligand(L, p_total_m, lnK_opt, S_eff, N_eff)
        F_grid.append(model.mole_fractions(Lf, lnK_opt, S_eff, N_eff))
    return np.array(F_grid)


def build_summary(all_L_tot, all_F_exp, all_Kd, all_stems, num_species_list,
                  all_lnK, all_S, all_N, model_name, out_dir, label, cfg):
    """Build multi-file summary: mean/std of experimental data, mean model curves,
    summary fit plot, and summary Kd CSV."""
    ref_L = all_L_tot[0]
    num_species = max(num_species_list)
    is_specific = (model_name == "specific_binding")

    # Interpolate experimental F_exps onto a common ligand grid
    F_exp_interp_list = []
    for L_tot, F_exp in zip(all_L_tot, all_F_exp):
        if F_exp.shape[1] < num_species:
            F_exp = np.pad(F_exp, ((0, 0), (0, num_species - F_exp.shape[1])), "constant")
        F_interp = np.zeros((len(ref_L), num_species))
        for j in range(num_species):
            mask = ~np.isnan(L_tot) & ~np.isnan(F_exp[:, j])
            x, y = np.array(L_tot[mask]), np.array(F_exp[:, j][mask])
            if len(x) == 0:
                F_interp[:, j] = np.nan
                continue
            order = np.argsort(x)
            x, y = x[order], y[order]
            F_interp[:, j] = interp1d(x, y, bounds_error=False, fill_value=np.nan)(ref_L)
        F_exp_interp_list.append(F_interp)

    F_arr = np.stack(F_exp_interp_list, axis=0)
    F_exp_mean = np.nanmean(F_arr, axis=0)
    F_exp_std = np.nanstd(F_arr, axis=0)

    # Model curves per run on a common grid
    L_grid_M = np.linspace(ref_L.min(), ref_L.max(), 300)
    calc_runs = []
    for lnK_opt, S_eff, N_eff in zip(all_lnK, all_S, all_N):
        F_grid_run = compute_model_grid(L_grid_M, lnK_opt, S_eff, N_eff, cfg.p_total_m, model_name)
        if F_grid_run.shape[1] < num_species:
            F_grid_run = np.pad(F_grid_run, ((0, 0), (0, num_species - F_grid_run.shape[1])), "constant")
        calc_runs.append(F_grid_run)
    calc_arr = np.stack(calc_runs, axis=0)
    F_calc_mean = np.nanmean(calc_arr, axis=0)
    F_calc_std = np.nanstd(calc_arr, axis=0)

    n_specific = None if is_specific else max(set(all_S), key=all_S.count)
    summary_stem = cfg.csv_name_wildcard.replace("*", "").replace(".csv", "")
    summary_fit_svg = os.path.join(out_dir, summary_stem + "fit_summary.svg")
    plot_summary_fit(ref_L, F_exp_mean, F_exp_std, L_grid_M, F_calc_mean, F_calc_std,
                     summary_fit_svg, num_species, cfg, n_specific=n_specific)
    print(f"[{label} Summary] Wrote summary fit plot to {summary_fit_svg}")

    # Mean Kd (in output units) — pad shorter arrays with NaN
    max_kd_len = max(len(kd) for kd in all_Kd)
    kd_padded = [np.pad(kd, (0, max_kd_len - len(kd)), constant_values=np.nan) if len(kd) < max_kd_len else kd
                 for kd in all_Kd]
    kd_values = np.stack(kd_padded, axis=1)
    mean_Kd_out = np.nanmean(kd_values, axis=1)

    kd_unit = cfg.output_unit
    param_names = REGISTRY[model_name].param_labels(n_specific if n_specific is not None else max_kd_len)
    if len(param_names) < max_kd_len:
        param_names += [f"p_{i}" for i in range(len(param_names), max_kd_len)]
    param_names = param_names[:max_kd_len]

    summary = {
        "Result": param_names,
        f"Mean_Kd_({kd_unit})": mean_Kd_out,
        f"Std_Kd_({kd_unit})": np.nanstd(kd_values, axis=1),
    }
    for col_idx, stem in enumerate(all_stems):
        summary[stem] = kd_values[:, col_idx]
    summary_file = os.path.join(out_dir, summary_stem + "stat_summary.csv")
    pd.DataFrame(summary).to_csv(summary_file, index=False, float_format="%.6e")
    print(f"[{label} Summary] Wrote summary Kd table to {summary_file}")

    return ref_L, F_exp_mean, F_exp_std, kd_values, mean_Kd_out, num_species


def build_deconv_summary(ref_L, F_exp_mean, F_exp_std, mean_Kd_out, num_species,
                         all_S, out_dir, cfg):
    """Build summary deconvolution plot/CSV. Geometric-nonspecific only."""
    summary_S = max(set(all_S), key=all_S.count)
    summary_N = num_species - summary_S - 1
    mean_Kd_trunc = mean_Kd_out[:summary_S + 1]
    mean_Ka_per_out = 1.0 / mean_Kd_trunc
    mean_Ka_M = mean_Ka_per_out * cfg.scale_m_to_out
    lnK_mean = np.log(mean_Ka_M)

    F_calc_mean_ref = np.array([
        geometric_nonspecific.mole_fractions(
            geometric_nonspecific.free_ligand(L, cfg.p_total_m, lnK_mean, summary_S, summary_N),
            lnK_mean, summary_S, summary_N,
        )
        for L in ref_L
    ])
    if F_calc_mean_ref.shape[1] < num_species:
        F_calc_mean_ref = np.pad(F_calc_mean_ref, ((0, 0), (0, num_species - F_calc_mean_ref.shape[1])), "constant")

    weights_ij = geometric_nonspecific.compute_deconvolution_weights(
        np.exp(lnK_mean)[0], np.exp(lnK_mean)[1:], summary_S, summary_N
    )
    contrib_all = np.zeros((len(ref_L), num_species, summary_S + 1))
    frac_within_all = np.zeros((len(ref_L), num_species, summary_S + 1))
    for idx in range(len(ref_L)):
        frac_within, contrib = geometric_nonspecific.deconvolve_fractions(
            F_calc_mean_ref[idx], weights_ij, summary_S
        )
        contrib_all[idx] = contrib
        frac_within_all[idx] = frac_within

    summary_stem = cfg.csv_name_wildcard.replace("*", "").replace(".csv", "")
    summary_deconv_svg = os.path.join(out_dir, summary_stem + "deconv_summary.svg")
    fig = plot_deconv_stacked(
        ref_L * cfg.scale_m_to_out, contrib_all, summary_S, summary_N,
        "Deconvoluted fraction of apparent", cfg,
        outline_totals=F_exp_mean, outline_err=F_exp_std, outline_label="F_exp mean",
    )
    safe_savefig(fig, summary_deconv_svg, cfg.max_image_dim)
    if cfg.show_plots:
        plt.show()
    else:
        plt.close(fig)
    print(f"[Geometric-nonspecific Summary] Wrote summary deconvolution plot to {summary_deconv_svg}")

    summary_deconv_csv = os.path.join(out_dir, summary_stem + "deconv_summary.csv")
    csv_rows = []
    for idx in range(len(ref_L)):
        L_out = ref_L[idx] * cfg.scale_m_to_out
        for i in range(num_species):
            max_j = min(i, summary_S)
            for j in range(max_j + 1):
                csv_rows.append({
                    f"L_tot({cfg.output_unit})": L_out,
                    "i_total": i, "j_specific": j, "m_nonspecific": i - j,
                    "F_exp_mean": F_exp_mean[idx][i] if i < F_exp_mean.shape[1] else np.nan,
                    "F_exp_std": F_exp_std[idx][i] if i < F_exp_std.shape[1] else np.nan,
                    "F_calc_mean": F_calc_mean_ref[idx][i],
                    "fraction_within_i": frac_within_all[idx][i, j],
                    "fraction_total": contrib_all[idx][i, j],
                })
    pd.DataFrame(csv_rows).to_csv(summary_deconv_csv, index=False)
    print(f"[Geometric-nonspecific Summary] Wrote summary deconvolution CSV to {summary_deconv_csv}")

    return summary_S


def compare_models_bic_aic(per_model_results, out_dir):
    """Cross-model BIC/AIC comparison.

    per_model_results: dict[model_name -> list[dict(stem, SSR, n_obs, n_params)]]
    All lists must have matching stems.
    """
    def _info(ssr, n, k):
        bic = n * np.log(ssr / n) + k * np.log(n)
        aic = n * np.log(ssr / n) + 2 * k
        aicc = aic + 2 * k * (k + 1) / (n - k - 1) if n > k + 1 else np.inf
        return bic, aic, aicc

    model_names = list(per_model_results.keys())
    if len(model_names) < 2:
        print("[compare_models] Need >= 2 models to compare.")
        return

    stems = [d["stem"] for d in per_model_results[model_names[0]]]
    csv_rows = []

    print("\n" + "=" * 70)
    print(f"CROSS-MODEL COMPARISON: {' | '.join(model_names)}")
    print("=" * 70)

    per_file_winners = []
    for file_idx, stem in enumerate(stems):
        print(f"\n=== {stem} ===")
        header = f"{'Model':<30} {'n_params':>10} {'SSR':>14} {'BIC':>12} {'AIC':>12} {'AICc':>12}"
        print(header)
        print("-" * len(header))
        bics = {}
        for name in model_names:
            d = per_model_results[name][file_idx]
            bic, aic, aicc = _info(d["SSR"], d["n_obs"], d["n_params"])
            bics[name] = bic
            print(f"{name:<30} {d['n_params']:>10d} {d['SSR']:>14.5f} {bic:>12.2f} {aic:>12.2f} {aicc:>12.2f}")
            csv_rows.append({
                "File": stem, "Model": name, "n_params": d["n_params"], "n_obs": d["n_obs"],
                "SSR": d["SSR"], "BIC": bic, "AIC": aic, "AICc": aicc,
            })
        winner = min(bics, key=bics.get)
        per_file_winners.append(winner)
        print(f"BIC winner: {winner}")

    print("\n" + "-" * 70)
    print("OVERALL SUMMARY")
    print("-" * 70)
    print(f"Files analysed: {len(stems)}")
    for name in model_names:
        wins = per_file_winners.count(name)
        print(f"  {name}: BIC winner in {wins}/{len(stems)}")
    print("-" * 70)

    csv_path = os.path.join(out_dir, "model_comparison.csv")
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False, float_format="%.6e")
    print(f"\nSaved model comparison to: {csv_path}")
