# scripts_binding

A modular Python package for fitting protein-ligand binding distributions from native mass spectrometry (nMS) titration data.

## Purpose

Given a titration of protein with ligand measured by native MS, this package fits the observed mole fractions of bound-state peaks (I0, I1, ..., Ik) to a statistical binding model and extracts site-resolved binding constants (Kd, Ka), per-ligand thermodynamics (with variable-temperature extension), and the partition between specific and nonspecific binding contributions.

## Models

Each model is one file under `models/` and exposes the same minimal interface
(`calculate_fractions_model`, `solve_L_free`, `residuals`) so they can all be
driven by the same fitting and comparison code.

1. **`models/specific.py`** — stepwise specific only, no NSB. K1..Kn.
2. **`models/nonspecific.py`** — stepwise specific + geometric NSB (existing baseline).
   Partition: Z[i] = Sum_j prod(Ks_1..Ks_j) * Kn^(i-j).
3. **`models/poisson_nsb.py`** — stepwise specific + Poisson NSB (Shimon 2010).
   Partition: Z[i] = Sum_j prod(Ks_1..Ks_j) * Kn^(i-j) / (i-j)!. The factorial
   weighting is the only difference from `nonspecific.py`.
4. **`models/binomial_poisson.py`** — noncooperative binomial specific + Poisson NSB
   (Daubenfeld 2006). Two free parameters: ln(Kn), ln(Ks). Single Ks across all
   S sites, no cooperativity captured.
5. **`models/powerlaw_nsb.py`** — stepwise specific + power-law NSB (Guan 2015).
   Apparent stepwise Ka_k = Ks_k + beta/k^gamma. Reduces to Shimon at gamma = 0.

## Tests

```
tests/
  test_poisson_nsb.py        synthetic recovery for Shimon model
  test_binomial_poisson.py   synthetic recovery for Daubenfeld model
  test_powerlaw_nsb.py       synthetic recovery + gamma=0 limit for Guan model
  test_real_compare.py       all 4 NSB models on real SR-GroEL AMAC + EDDA data
```

Run any test directly:

```bash
python tests/test_poisson_nsb.py
python tests/test_real_compare.py
```

## Usage

```bash
python -m scripts_binding configs/my_config.yaml
```

Configs are YAML files describing data paths, units, and model choices. See `config.py` for the full `RunConfig` schema.

## Module layout

```
scripts_binding/
  config.py        RunConfig dataclass + YAML loader
  runner.py        Orchestrator (scan temperatures, files, models)
  fitting.py       Fitting wrappers (process_file_specific, process_file_nonspecific, auto_select_S)
  reporting.py     Tables + CSV output
  summary.py       Multi-replicate aggregation
  plotting.py      Fit curves, convergence, deconvolution bars
  models/
    specific.py    Stepwise specific-only
    nonspecific.py Mixed model (current formulation)
```

## License

Internal research code. Contact author for reuse.
