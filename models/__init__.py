"""Binding models for native MS titration analysis.

All model modules expose the same interface so the same fitting driver
can call any of them:

    MODEL_NAME : str
    mole_fractions(L_free_M, ln_params, S, N) -> ndarray
    free_ligand(L_tot_M, P_tot_M, ln_params, S, N) -> float
    residual_vector(ln_params, L_totals_M, P_tot_M, F_exps, S, N, ssr_history) -> ndarray
    n_params(S) -> int
    initial_lnK(S, ...) -> ndarray
    param_labels(S) -> list[str]

Select a model via the REGISTRY:

    from scripts_binding.models import REGISTRY
    model = REGISTRY["poisson_nonspecific"]
    lnK0 = model.initial_lnK(S=7)
    res = least_squares(model.residual_vector, lnK0, args=(...))
"""
from . import specific_binding
from . import geometric_nonspecific
from . import poisson_nonspecific
from . import binomial_poisson_nonspecific
from . import power_law_nonspecific

REGISTRY = {
    specific_binding.MODEL_NAME: specific_binding,
    geometric_nonspecific.MODEL_NAME: geometric_nonspecific,
    poisson_nonspecific.MODEL_NAME: poisson_nonspecific,
    binomial_poisson_nonspecific.MODEL_NAME: binomial_poisson_nonspecific,
    power_law_nonspecific.MODEL_NAME: power_law_nonspecific,
}
