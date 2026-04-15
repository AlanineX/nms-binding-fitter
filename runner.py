"""Main CLI entry point.

Usage:
    python -m scripts_binding <config.yaml>
"""
import os
import sys
from contextlib import contextmanager
from glob import glob

from .config import load_configs
from .plotting import setup_matplotlib
from .fitting import process_file, auto_select_S
from .summary import build_summary, build_deconv_summary, compare_models_bic_aic


class TeeLogger:
    """Duplicate stdout to a log file. Context-manager friendly."""
    def __init__(self, log_path, mode="w"):
        self.path = log_path
        self.mode = mode
        self.terminal = None
        self.log = None

    def __enter__(self):
        self.terminal = sys.stdout
        self.log = open(self.path, self.mode)
        sys.stdout = self
        return self

    def __exit__(self, *exc):
        sys.stdout = self.terminal
        self.log.close()

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


@contextmanager
def tee_to(log_path, mode="w"):
    with TeeLogger(log_path, mode) as logger:
        yield logger


def gather_data_paths(cfg):
    if cfg.data_path:
        return [cfg.data_path]
    pattern = os.path.join(cfg.base_dir, cfg.csv_name_wildcard)
    return sorted(glob(pattern))


def _resolve_mode_S(data_paths, cfg, model_name):
    """For auto-S: pick modal S across replicates; else return None."""
    if cfg.s_mode.lower() != "auto":
        return None
    per_rep_S = [auto_select_S(path, cfg, model_name) for path in data_paths]
    mode_S = max(set(per_rep_S), key=per_rep_S.count)
    print(f"\n[Auto-S] Per-replicate best S: {per_rep_S}")
    print(f"[Auto-S] Mode S = {mode_S} — forcing all replicates to use S = {mode_S}\n")
    return mode_S


def _run_model(model_name, data_paths, cfg):
    """Fit every data file with one model. Returns list of per-file info dicts."""
    model_dir = os.path.join(cfg.out_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)

    log_path = os.path.join(model_dir, f"{model_name}_log.txt")
    results = []

    with tee_to(log_path, mode="w"):
        mode_S = _resolve_mode_S(data_paths, cfg, model_name) if model_name != "specific_binding" else None
        for path in data_paths:
            info = process_file(path, model_dir, cfg, model_name, S_override=mode_S)
            results.append(info)

        if cfg.summary_enable and results:
            ref_L, F_exp_mean, F_exp_std, _, mean_Kd_out, num_species = build_summary(
                all_L_tot=[r["L_totals_M"] for r in results],
                all_F_exp=[r["F_exps"] for r in results],
                all_Kd=[r["Kd_out"] for r in results],
                all_stems=[r["stem"] for r in results],
                num_species_list=[r["num_species"] for r in results],
                all_lnK=[r["lnK_opt"] for r in results],
                all_S=[r["S_eff"] for r in results],
                all_N=[r["N_eff"] for r in results],
                model_name=model_name,
                out_dir=model_dir,
                label=model_name,
                cfg=cfg,
            )
            if model_name == "geometric_nonspecific":
                build_deconv_summary(
                    ref_L, F_exp_mean, F_exp_std, mean_Kd_out, num_species,
                    [r["S_eff"] for r in results], model_dir, cfg,
                )

    print(f"[{model_name}] Log saved to {log_path}")
    return results


def run_single(cfg):
    """Run one (system, temperature) analysis across every model in cfg.models."""
    setup_matplotlib(cfg)

    data_paths = gather_data_paths(cfg)
    if not data_paths:
        raise FileNotFoundError(
            f"No CSV files found for pattern: {os.path.join(cfg.base_dir, cfg.csv_name_wildcard)}"
        )

    if not cfg.models:
        raise ValueError("cfg.models is empty — specify at least one model name.")

    per_model = {name: _run_model(name, data_paths, cfg) for name in cfg.models}

    if len(cfg.models) >= 2:
        cmp_log = os.path.join(cfg.out_dir, "model_comparison_log.txt")
        with tee_to(cmp_log, mode="w"):
            compare_models_bic_aic(
                {name: [{"stem": r["stem"], "SSR": r["SSR"],
                         "n_obs": r["n_obs"], "n_params": r["n_params"]}
                        for r in results]
                 for name, results in per_model.items()},
                cfg.out_dir,
            )
        print(f"[Model Comparison] Log saved to {cmp_log}")


def run_all(yaml_path):
    configs = load_configs(yaml_path)
    print(f"Loaded {len(configs)} run configuration(s) from {yaml_path}")
    for i, cfg in enumerate(configs):
        print(f"\n{'='*60}")
        print(f"  Run {i+1}/{len(configs)}: {cfg.csv_name_wildcard} | S={cfg.s} | S_MODE={cfg.s_mode}")
        print(f"  Models: {cfg.models}")
        print(f"  Base: {cfg.base_dir}")
        print(f"  Out:  {cfg.out_dir}")
        print(f"{'='*60}")
        run_single(cfg)


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m scripts_binding <config.yaml>")
        sys.exit(1)
    run_all(sys.argv[1])


if __name__ == "__main__":
    main()
