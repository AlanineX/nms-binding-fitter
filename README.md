# scripts_binding

A modular Python package for fitting protein-ligand binding distributions from native mass spectrometry (nMS) titration data.

## Purpose

Given a titration of protein with ligand measured by native MS, this package fits the observed mole fractions of bound-state peaks (I0, I1, ..., Ik) to a statistical binding model and extracts site-resolved binding constants (Kd, Ka), per-ligand thermodynamics, and the partition between specific and nonspecific binding contributions.

## Models

Every model exposes a uniform interface (`mole_fractions`, `free_ligand`, `residual_vector`, `n_params`, `initial_lnK`, `param_labels`, `MODEL_NAME`) and is registered in `models.REGISTRY`.

| File | `MODEL_NAME` | Description |
|---|---|---|
| `models/specific_binding.py` | `specific_binding` | Stepwise specific only, no nonspecific. K1..Kn |
| `models/geometric_nonspecific.py` | `geometric_nonspecific` | Stepwise specific + geometric nonspecific. Z[i] = Σ_j prod(Ks_1..Ks_j) · Kn^(i-j) |
| `models/poisson_nonspecific.py` | `poisson_nonspecific` | Stepwise specific + Poisson nonspecific (Shimon 2010). Adds 1/(i-j)! weighting |
| `models/binomial_poisson_nonspecific.py` | `binomial_poisson_nonspecific` | Noncooperative binomial specific + Poisson nonspecific (Daubenfeld 2006). 2 free params |
| `models/power_law_nonspecific.py` | `power_law_nonspecific` | Stepwise specific + power-law nonspecific (Guan 2015). K_app_k = Ks_k + β/k^γ |

## Usage

```bash
python -m scripts_binding configs/my_config.yaml
```

Configs are YAML files listing the models to run in `models: [...]`. See `config.py` for the full `RunConfig` schema.

## Tests

```
tests/
  test_poisson_nonspecific.py            synthetic recovery for Shimon model
  test_binomial_poisson_nonspecific.py   synthetic recovery for Daubenfeld model
  test_power_law_nonspecific.py          synthetic recovery + gamma=0 limit for Guan model
  test_real_compare.py                   all NSB models on real data (takes --data LABEL:PATH)
```

Run any synthetic test directly:

```bash
python tests/test_poisson_nonspecific.py
```

Run the real-data comparison with explicit paths:

```bash
python tests/test_real_compare.py \
    --data AMAC_25C_1:/path/to/AMAC_25C_1.csv \
    --data EDDA_25C_1:/path/to/EDDA_25C_1.csv
```

## Batch scan across buffers/temps/reps

```bash
python -m scripts_binding.scripts.batch_kd_scan \
    --out-dir /path/to/out \
    --buffer AMAC:/path/to/amac_dir:7 \
    --buffer EDDA:/path/to/edda_dir:5 \
    --temps 5,10,15,20,25,30,35 \
    --reps 1,2,3 \
    --p-tot 1e-6
```

## Module layout

```
scripts_binding/
  config.py          RunConfig dataclass + YAML loader
  runner.py          Orchestrator (per-model fit, summary, comparison)
  fitting.py         Registry-driven process_file + auto_select_S
  reporting.py       Tables + CSV output
  summary.py         Multi-replicate aggregation + N-model BIC comparison
  plotting.py        Fit curves, convergence, deconvolution bars
  models/            Five models behind a uniform interface (see above)
  scripts/
    batch_kd_scan.py Cross-buffer batch scan (CLI)
```

## License

Internal research code. Contact author for reuse.
