"""Registry-driven fitting orchestration.

One process_file entry point dispatches to any model in REGISTRY.
CSV loading, S-scan, uncertainty computation, and plotting are shared.
"""
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

from .models import REGISTRY
from .models import geometric_nonspecific
from .plotting import safe_savefig, plot_fit_curves, plot_convergence, plot_deconv_stacked
from .reporting import (
    print_results_table,
    compute_uncertainties,
    save_kd_csv,
    print_per_point_summary,
)


def parse_ligand_conc(entry, scale_l_in_to_m):
    """Extract numeric concentration from entry and convert to Molar."""
    if pd.isna(entry):
        return np.nan
    if isinstance(entry, (int, float)):
        return float(entry) * scale_l_in_to_m
    try:
        match = re.search(r"[\d\.]+", str(entry))
        if match:
            return float(match.group(0)) * scale_l_in_to_m
    except (ValueError, TypeError):
        pass
    return np.nan


def load_binding_csv(data_path, cfg):
    """Load a titration CSV; return df, L_totals_M, I_cols, F_exps (row-normalized)."""
    df = pd.read_csv(data_path).dropna(subset=["Entry"]).fillna(0)
    L_totals_M = df.iloc[:, 0].apply(lambda e: parse_ligand_conc(e, cfg.scale_l_in_to_m)).values

    I_cols = sorted([c for c in df.columns if c.startswith("I")], key=lambda c: int(c[1:]))
    I_vals = df[I_cols].values
    F_exps = (I_vals.T / I_vals.sum(axis=1)).T
    return df, L_totals_M, I_cols, F_exps


def _bic(rss, n_obs, n_par):
    if rss <= 0 or n_obs <= n_par:
        return np.inf
    return n_obs * np.log(rss / n_obs) + n_par * np.log(n_obs)


def fit_quick(model, L_totals_M, F_exps, P_tot_M, S, N):
    """Lightweight fit returning stats only. Used for auto-S BIC scans."""
    num_species_model = S + N + 1
    F_exps_padded = F_exps
    if F_exps.shape[1] < num_species_model:
        F_exps_padded = np.pad(F_exps, ((0, 0), (0, num_species_model - F_exps.shape[1])), "constant")

    lnK0 = model.initial_lnK(S)
    ssr_history = []
    try:
        fit = least_squares(
            model.residual_vector,
            lnK0,
            args=(L_totals_M, P_tot_M, F_exps_padded, S, N, ssr_history),
            verbose=0,
        )
    except Exception as e:
        print(f"  [Auto-S] S={S} failed: {e}")
        return None

    rss = float(np.dot(fit.fun, fit.fun))
    n_obs = fit.fun.size
    n_par = len(fit.x)
    return {
        "S": S, "N": N, "n_params": n_par, "n_obs": n_obs,
        "SSR": rss, "BIC": _bic(rss, n_obs, n_par), "lnK_opt": fit.x,
    }


def auto_select_S(data_path, cfg, model_name):
    """Scan S = 0..max_i for the given model; return S with lowest BIC."""
    model = REGISTRY[model_name]
    df, L_totals_M, I_cols, F_exps = load_binding_csv(data_path, cfg)
    max_i = len(I_cols) - 1

    print(f"\n{'='*60}")
    print(f"[Auto-S | {model_name}] Scanning S = 0..{max_i} for: {os.path.basename(data_path)}")
    print(f"{'='*60}")

    results = []
    for S in range(0, max_i + 1):
        info = fit_quick(model, L_totals_M, F_exps, cfg.p_total_m, S, max_i - S)
        if info is not None:
            results.append(info)

    if not results:
        print("[Auto-S] All fits failed. Falling back to S=1.")
        return 1

    best = min(results, key=lambda r: r["BIC"])
    header = f"  {'S':>3} {'N':>3} {'params':>6} {'SSR':>12} {'BIC':>12}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in results:
        marker = " <-- best" if r["S"] == best["S"] else ""
        print(f"  {r['S']:>3} {r['N']:>3} {r['n_params']:>6} {r['SSR']:>12.4e} {r['BIC']:>12.2f}{marker}")

    print(f"\n[Auto-S] Selected S = {best['S']} (BIC = {best['BIC']:.2f})\n")
    return best["S"]


def _resolve_S_N(cfg, model_name, max_i, S_override):
    """Resolve effective S and N from config + data width."""
    if model_name == "specific_binding":
        return max_i, 0

    S_use = S_override if S_override is not None else cfg.s
    S_eff = S_use
    if S_eff > max_i:
        if cfg.auto_adjust_s:
            print(f"[Warning] S={S_use} exceeds I0..I{max_i}. Clamping to {max_i}.")
            S_eff = max_i
        else:
            raise ValueError(f"S={S_use} exceeds I0..I{max_i}.")

    if cfg.n_override is not None:
        N_eff = cfg.n_override
        if S_eff + N_eff != max_i:
            print(f"[Warning] S+N={S_eff + N_eff} does not match max_i={max_i}.")
    else:
        N_eff = max_i - S_eff

    if N_eff < 0:
        raise ValueError(f"Computed N={N_eff} is negative.")
    return S_eff, N_eff


def _trim_low_pop_species(F_exps, I_cols, min_frac):
    """Drop trailing species whose max fraction is below min_frac. Renormalize."""
    trimmed = []
    n_steps = len(I_cols) - 1
    while n_steps > 1:
        max_frac = F_exps[:, n_steps].max()
        if max_frac < min_frac:
            trimmed.append(f"I{n_steps} (max={max_frac:.4f})")
            F_exps = F_exps[:, :n_steps]
            I_cols = I_cols[:n_steps + 1]
            n_steps -= 1
        else:
            break
    if trimmed:
        print(f"  [Trimmed] Dropped species with max frac < {min_frac}: {', '.join(trimmed)}")
    F_exps = (F_exps.T / F_exps.sum(axis=1)).T
    return F_exps, I_cols


def _run_deconvolution(model, df, L_totals_M, F_exps, F_calcs, lnK_opt, S, N, cfg, out_dir, stem):
    """Deconvolve F into (j specific, i-j nonspecific) contributions. Geometric-nonspecific only."""
    if not hasattr(model, "compute_deconvolution_weights"):
        return

    Ka_opt = np.exp(lnK_opt)
    Kn_opt = Ka_opt[0]
    Ks_opt = Ka_opt[1:]
    weights_ij = model.compute_deconvolution_weights(Kn_opt, Ks_opt, S, N)

    if cfg.deconv_use_grid:
        L_vals_M = np.linspace(L_totals_M.min(), L_totals_M.max(), cfg.deconv_grid_points)
        L_free_vals = np.array([model.free_ligand(L, cfg.p_total_m, lnK_opt, S, N) for L in L_vals_M])
        F_source = np.vstack([model.mole_fractions(Lf, lnK_opt, S, N) for Lf in L_free_vals])
        source_label = "calc (grid)"
        source_key = "calc_grid"
        entries = [f"{v * cfg.scale_m_to_out:.3g}" for v in L_vals_M]
    else:
        L_vals_M = L_totals_M
        entries = df["Entry"].tolist()
        if cfg.deconv_source.lower() == "exp":
            F_source = F_exps
            source_label = "exp"
            source_key = "exp"
        else:
            F_source = F_calcs
            source_label = "calc"
            source_key = "calc"

    n_points = len(L_vals_M)
    num_species = S + N + 1
    frac_within_all = np.zeros((n_points, num_species, S + 1))
    contrib_all = np.zeros((n_points, num_species, S + 1))
    for idx in range(n_points):
        frac_within, contrib = model.deconvolve_fractions(F_source[idx], weights_ij, S)
        frac_within_all[idx] = frac_within
        contrib_all[idx] = contrib

    deconv_csv = cfg.deconv_csv_path or os.path.join(out_dir, f"{stem}_deconv.csv")
    rows = []
    for idx in range(n_points):
        L_out = L_vals_M[idx] * cfg.scale_m_to_out
        for i in range(num_species):
            max_j = min(i, S)
            for j in range(max_j + 1):
                rows.append({
                    "Entry": entries[idx],
                    f"L_tot({cfg.output_unit})": L_out,
                    "i_total": i, "j_specific": j, "m_nonspecific": i - j,
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
                max_j = min(i, S)
                parts = [f"{j} spec + {i-j} non: {100.0*frac_within_all[idx][i,j]:.1f}%" for j in range(max_j + 1)]
                print(f"  I{i}: " + "; ".join(parts))

    outline_totals = None if cfg.deconv_use_grid else F_exps
    fig = plot_deconv_stacked(
        L_vals_M * cfg.scale_m_to_out, contrib_all, S, N,
        "Deconvoluted fraction of apparent", cfg,
        outline_totals=outline_totals, outline_label="F_exp total",
    )
    if cfg.save_plots:
        safe_savefig(fig, os.path.join(out_dir, f"{stem}_deconv.svg"), cfg.max_image_dim)
    if cfg.show_plots:
        plt.show()
    else:
        plt.close(fig)


def process_file(data_path, out_dir, cfg, model_name, S_override=None):
    """Fit one titration CSV with the named model. Returns result dict."""
    model = REGISTRY[model_name]
    is_specific = (model_name == "specific_binding")

    ssr_history = []
    df, L_totals_M, I_cols, F_exps = load_binding_csv(data_path, cfg)

    if is_specific:
        F_exps, I_cols = _trim_low_pop_species(F_exps, I_cols, cfg.min_species_frac)

    max_i = len(I_cols) - 1
    S_eff, N_eff = _resolve_S_N(cfg, model_name, max_i, S_override)
    num_species_model = S_eff + N_eff + 1

    # Pad F_exps if the model expects more species than data provides
    if F_exps.shape[1] < num_species_model:
        F_exps = np.pad(F_exps, ((0, 0), (0, num_species_model - F_exps.shape[1])), "constant")

    print(f"\n=== [{model_name}] Processing: {data_path} ===")
    print(f"Model parameters: S={S_eff}, N={N_eff}, n_params={model.n_params(S_eff)}")

    lnK0 = model.initial_lnK(S_eff)

    print(f"--- Starting optimization ({model_name}) ---")
    fit = least_squares(
        model.residual_vector, lnK0,
        args=(L_totals_M, cfg.p_total_m, F_exps, S_eff, N_eff, ssr_history),
        verbose=2,
    )
    print(f"--- Optimization finished ({model_name}) ---\n")

    lnK_opt = fit.x
    Ka_opt_M = np.exp(lnK_opt)
    Kd_opt_M = 1.0 / np.where(Ka_opt_M > 0, Ka_opt_M, np.nan)
    ssr_final = float(np.dot(fit.fun, fit.fun))
    n_obs = fit.fun.size

    has_errors, std_Ka_M, std_Kd_M = compute_uncertainties(
        fit, lnK_opt, Ka_opt_M, Kd_opt_M, list(ssr_history)
    )

    param_names = model.param_labels(S_eff)
    print_results_table(param_names, Ka_opt_M, Kd_opt_M, has_errors, cfg, std_Ka_M, std_Kd_M)

    L_free_list = np.array([model.free_ligand(L, cfg.p_total_m, lnK_opt, S_eff, N_eff) for L in L_totals_M])
    F_calcs = np.vstack([model.mole_fractions(Lf, lnK_opt, S_eff, N_eff) for Lf in L_free_list])

    if cfg.debug_validate and model_name == "geometric_nonspecific" and len(L_totals_M) > 0:
        if cfg.debug_ligand_conc is not None:
            L_out_all = L_totals_M * cfg.scale_m_to_out
            idx = int(np.argmin(np.abs(L_out_all - cfg.debug_ligand_conc)))
        else:
            idx = min(cfg.debug_index, len(L_totals_M) - 1)
        geometric_nonspecific.debug_validate_point(
            idx, L_totals_M[idx], L_free_list[idx], lnK_opt, F_calcs[idx], S_eff, N_eff, cfg
        )

    print_per_point_summary(df, L_totals_M, F_exps, L_free_list, F_calcs, num_species_model, cfg)

    stem = os.path.splitext(os.path.basename(data_path))[0]
    fit_svg = os.path.join(out_dir, f"{stem}_fit.svg")
    conv_svg = os.path.join(out_dir, f"{stem}_conv.svg")
    kd_csv = os.path.join(out_dir, f"{stem}_kd.csv")

    L_grid_M = np.linspace(L_totals_M.min(), L_totals_M.max(), 300)
    F_grid = np.vstack([
        model.mole_fractions(model.free_ligand(L, cfg.p_total_m, lnK_opt, S_eff, N_eff), lnK_opt, S_eff, N_eff)
        for L in L_grid_M
    ])
    n_specific_for_plot = None if is_specific else S_eff
    plot_fit_curves(
        L_totals_M, F_exps, L_grid_M, F_grid, num_species_model,
        f"{model_name}: global fit", fit_svg, cfg, n_specific=n_specific_for_plot,
    )
    plot_convergence(list(ssr_history), conv_svg, cfg)
    save_kd_csv(param_names, Ka_opt_M, Kd_opt_M, has_errors, std_Ka_M, std_Kd_M, kd_csv, cfg)

    if cfg.deconv_enable and not is_specific:
        _run_deconvolution(model, df, L_totals_M, F_exps, F_calcs, lnK_opt, S_eff, N_eff, cfg, out_dir, stem)

    return {
        "model_name": model_name,
        "L_totals_M": L_totals_M,
        "F_exps": F_exps,
        "Kd_out": Kd_opt_M * cfg.scale_m_to_out,
        "num_species": num_species_model,
        "S_eff": S_eff, "N_eff": N_eff,
        "stem": stem, "lnK_opt": lnK_opt,
        "SSR": ssr_final, "n_obs": n_obs, "n_params": len(lnK_opt),
    }
