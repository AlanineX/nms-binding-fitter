"""
Main CLI entry point — replaces both main() in test_6.py and run_all_temps.py.

Usage:
    python -m scripts_binding.runner config.yaml
"""
import sys
import os
from glob import glob

from .config import RunConfig, load_configs
from .plotting import setup_matplotlib
from .fitting import process_file_specific, process_file_nonspecific, auto_select_S
from .summary import (
    compute_model_grid_specific,
    compute_model_grid_nonspecific,
    build_summary,
    build_nonspecific_deconv_summary,
    compare_models_bic_aic,
)


class TeeLogger:
    """Duplicates stdout to both the console and a log file."""
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log = open(log_path, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


def gather_data_paths(cfg):
    """Find CSV files matching the config pattern."""
    if cfg.data_path:
        return [cfg.data_path]
    pattern = os.path.join(cfg.base_dir, cfg.csv_name_wildcard)
    return sorted(glob(pattern))


def run_single(cfg):
    """Run one (system, temperature) analysis. Returns per-model results."""
    setup_matplotlib(cfg)

    data_paths = gather_data_paths(cfg)
    if not data_paths:
        raise FileNotFoundError(
            f"No CSV files found for pattern: {os.path.join(cfg.base_dir, cfg.csv_name_wildcard)}")

    # Create subdirectories
    specific_dir = os.path.join(cfg.out_dir, "specific")
    nonspecific_dir = os.path.join(cfg.out_dir, "nonspecific")
    if cfg.specific_enable:
        os.makedirs(specific_dir, exist_ok=True)
    if cfg.nonspecific_enable:
        os.makedirs(nonspecific_dir, exist_ok=True)

    # --- Accumulators for specific model ---
    sp_all_L_tot = []
    sp_all_F_exp = []
    sp_all_Kd = []
    sp_all_stems = []
    sp_num_species_list = []
    sp_all_lnK = []
    sp_all_ssr = []
    sp_all_nobs = []

    # --- Accumulators for nonspecific model ---
    ns_all_L_tot = []
    ns_all_F_exp = []
    ns_all_Kd = []
    ns_all_stems = []
    ns_num_species_list = []
    ns_all_lnK = []
    ns_all_S = []
    ns_all_N = []
    ns_all_ssr = []
    ns_all_nobs = []

    # =================================================================
    # === PER-FILE FITTING =============================================
    # =================================================================

    # --- Specific model: start log ---
    if cfg.specific_enable:
        sp_log_path = os.path.join(specific_dir, "specific_log.txt")
        sp_logger = TeeLogger(sp_log_path)
        sys.stdout = sp_logger

    try:
        for path in data_paths:
            if cfg.specific_enable:
                info_sp = process_file_specific(path, specific_dir, cfg)
                sp_all_ssr.append(info_sp["SSR"])
                sp_all_nobs.append(info_sp["n_obs"])
                if cfg.summary_enable:
                    sp_all_L_tot.append(info_sp["L_totals_M"])
                    sp_all_F_exp.append(info_sp["F_exps"])
                    sp_all_Kd.append(info_sp["Kd_out"])
                    sp_all_stems.append(info_sp["stem"])
                    sp_num_species_list.append(info_sp["num_species"])
                    sp_all_lnK.append(info_sp["lnK_opt"])
    finally:
        if cfg.specific_enable:
            sys.stdout = sp_logger.terminal
            sp_logger.close()
            print(f"[Specific] Log saved to {sp_log_path}")

    # --- Nonspecific model: start log ---
    if cfg.nonspecific_enable:
        ns_log_path = os.path.join(nonspecific_dir, "nonspecific_log.txt")
        ns_logger = TeeLogger(ns_log_path)
        sys.stdout = ns_logger

    try:
        if cfg.nonspecific_enable:
            # --- Auto-S: first pass to find per-replicate best S, then mode-force ---
            if cfg.s_mode.lower() == "auto":
                per_rep_S = []
                for path in data_paths:
                    S_best = auto_select_S(path, cfg)
                    per_rep_S.append(S_best)
                mode_S = max(set(per_rep_S), key=per_rep_S.count)
                print(f"\n[Auto-S] Per-replicate best S: {per_rep_S}")
                print(f"[Auto-S] Mode S = {mode_S} — forcing all replicates to use S = {mode_S}\n")
            else:
                mode_S = None  # use cfg.s

            for path in data_paths:
                info_ns = process_file_nonspecific(path, nonspecific_dir, cfg, S_override=mode_S)
                ns_all_ssr.append(info_ns["SSR"])
                ns_all_nobs.append(info_ns["n_obs"])
                if cfg.summary_enable:
                    ns_all_L_tot.append(info_ns["L_totals_M"])
                    ns_all_F_exp.append(info_ns["F_exps"])
                    ns_all_Kd.append(info_ns["Kd_out"])
                    ns_all_stems.append(info_ns["stem"])
                    ns_num_species_list.append(info_ns["num_species_model"])
                    ns_all_lnK.append(info_ns["lnK_opt"])
                    ns_all_S.append(info_ns["S_eff"])
                    ns_all_N.append(info_ns["N_eff"])
    finally:
        if cfg.nonspecific_enable:
            sys.stdout = ns_logger.terminal
            ns_logger.close()
            print(f"[Nonspecific] Log saved to {ns_log_path}")

    # =================================================================
    # === SPECIFIC MODEL SUMMARY ======================================
    # =================================================================
    if cfg.specific_enable and cfg.summary_enable and sp_all_L_tot:
        sp_logger = TeeLogger.__new__(TeeLogger)
        sp_logger.terminal = sys.stdout
        sp_logger.log = open(sp_log_path, 'a')
        sys.stdout = sp_logger
        try:
            n_params = sp_all_lnK[0].size
            extra_args_list = [(cfg.p_total_m,) for _ in sp_all_lnK]
            build_summary(
                sp_all_L_tot, sp_all_F_exp, sp_all_Kd, sp_all_stems,
                sp_num_species_list, sp_all_lnK,
                compute_model_grid_specific, extra_args_list,
                specific_dir, "Specific", n_params, cfg
            )
        finally:
            sys.stdout = sp_logger.terminal
            sp_logger.close()

    # =================================================================
    # === NONSPECIFIC MODEL SUMMARY ===================================
    # =================================================================
    if cfg.nonspecific_enable and cfg.summary_enable and ns_all_L_tot:
        ns_logger = TeeLogger.__new__(TeeLogger)
        ns_logger.terminal = sys.stdout
        ns_logger.log = open(ns_log_path, 'a')
        sys.stdout = ns_logger
        try:
            n_params = ns_all_lnK[0].size
            extra_args_list = [(S_eff, N_eff, cfg.p_total_m) for S_eff, N_eff in zip(ns_all_S, ns_all_N)]

            summary_S_for_plot = max(set(ns_all_S), key=ns_all_S.count)

            ref_L, F_exp_mean, F_exp_std, kd_values, mean_Kd_out, num_species = build_summary(
                ns_all_L_tot, ns_all_F_exp, ns_all_Kd, ns_all_stems,
                ns_num_species_list, ns_all_lnK,
                compute_model_grid_nonspecific, extra_args_list,
                nonspecific_dir, "Nonspecific", n_params, cfg,
                n_specific=summary_S_for_plot
            )

            build_nonspecific_deconv_summary(
                ref_L, F_exp_mean, F_exp_std, mean_Kd_out, num_species,
                ns_all_S, nonspecific_dir, cfg
            )
        finally:
            sys.stdout = ns_logger.terminal
            ns_logger.close()

    # =================================================================
    # === CROSS-MODEL COMPARISON ======================================
    # =================================================================
    if cfg.specific_enable and cfg.nonspecific_enable and sp_all_ssr and ns_all_ssr:
        sp_data = [{"stem": s, "SSR": ssr, "n_obs": n, "n_params": p}
                   for s, ssr, n, p in zip(sp_all_stems, sp_all_ssr, sp_all_nobs,
                                           [len(lk) for lk in sp_all_lnK])]
        ns_data = [{"stem": s, "SSR": ssr, "n_obs": n, "n_params": p}
                   for s, ssr, n, p in zip(ns_all_stems, ns_all_ssr, ns_all_nobs,
                                           [len(lk) for lk in ns_all_lnK])]

        cmp_log_path = os.path.join(cfg.out_dir, "model_comparison_log.txt")
        cmp_logger = TeeLogger(cmp_log_path)
        sys.stdout = cmp_logger
        try:
            compare_models_bic_aic(sp_data, ns_data, cfg.out_dir)
        finally:
            sys.stdout = cmp_logger.terminal
            cmp_logger.close()
        print(f"[Model Comparison] Log saved to {cmp_log_path}")


def run_all(yaml_path):
    """Load YAML config and run all (system, temperature) analyses."""
    configs = load_configs(yaml_path)
    print(f"Loaded {len(configs)} run configuration(s) from {yaml_path}")
    for i, cfg in enumerate(configs):
        print(f"\n{'='*60}")
        print(f"  Run {i+1}/{len(configs)}: {cfg.csv_name_wildcard} | S={cfg.s} | S_MODE={cfg.s_mode}")
        print(f"  Base: {cfg.base_dir}")
        print(f"  Out:  {cfg.out_dir}")
        print(f"{'='*60}")
        run_single(cfg)


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m scripts_binding.runner <config.yaml>")
        sys.exit(1)
    run_all(sys.argv[1])


if __name__ == "__main__":
    main()
