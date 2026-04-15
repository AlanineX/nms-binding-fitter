"""Binding models for native MS titration analysis.

All model modules expose the same minimal interface so the same fitting
driver can call any of them:

    MODEL_NAME : str
    mole_fractions(L_free_M, ln_params, S, N) -> ndarray
    free_ligand(L_tot_M, P_tot_M, ln_params, S, N) -> float
    residual_vector(ln_params, L_totals_M, P_tot_M, F_exps, S, N, ssr_history) -> ndarray
    n_params(S) -> int
    initial_lnK(S, ...) -> ndarray
    param_labels(S) -> list[str]

Switch model in user code via the REGISTRY:

    from scripts_binding.models import REGISTRY
    model = REGISTRY['poisson_nsb']
    lnK0 = model.initial_lnK(S=7)
    res = least_squares(model.residual_vector, lnK0, args=(...))
"""
from . import specific
from . import nonspecific
from . import poisson_nsb
from . import binomial_poisson
from . import powerlaw_nsb

REGISTRY = {
    specific.MODEL_NAME: specific,                  # "specific"
    nonspecific.MODEL_NAME: nonspecific,            # "geometric_nsb"
    poisson_nsb.MODEL_NAME: poisson_nsb,            # "poisson_nsb"
    binomial_poisson.MODEL_NAME: binomial_poisson,  # "binomial_poisson"
    powerlaw_nsb.MODEL_NAME: powerlaw_nsb,          # "powerlaw_nsb"
}
