# scripts_binding

A modular Python package for fitting protein-ligand binding distributions from native mass spectrometry (nMS) titration data.

## Purpose

Given a titration of protein with ligand measured by native MS, this package fits the observed mole fractions of bound-state peaks (I0, I1, ..., Ik) to a statistical binding model and extracts site-resolved binding constants (Kd, Ka), per-ligand thermodynamics (with variable-temperature extension), and the partition between specific and nonspecific binding contributions.

## Current models

1. **Specific-only stepwise** (`models/specific.py`): stepwise association constants K1..Kn, no nonspecific component.
2. **Mixed specific/nonspecific** (`models/nonspecific.py`): S specific sites with individual Ks_i plus N nonspecific sites with a single lumped Kn, partition function Z[i] = Sum_j prod(Ks_1..Ks_j) * Kn^(i-j), with Bayesian-Information-Criterion auto-selection of S.

## Planned models (to be added)

3. **Daubenfeld 2006 binomial-Poisson** — specific binding as binomial over S sites with constant Ks, nonspecific binding as Poisson mean.
4. **Shimon-Sharon-Horovitz 2010 single-lumped subtraction** — subtract one Kn_lump from stoichiometries beyond the specific-site count.
5. **Guan 2015 power-law decay** — Kn_j = beta / j^gamma over ligand number (gamma = 0 recovers Shimon case).

See `PLAN.md` for the implementation roadmap and `docs/` for mathematical descriptions.

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
