"""
Specific-only stepwise binding model.
K_list = [K1, K2, ..., Kn] are stepwise association constants.
alpha_i = L^i * prod(K1..Ki)
"""
import numpy as np
from scipy.optimize import brentq


def compute_F_calc_specific(L_free, K_list):
    """
    Computes predicted mole fractions for a simple stepwise binding model.
    K_list = [K1, K2, ..., Kn] are stepwise association constants.
    alpha_i = L^i * prod(K1..Ki)
    """
    n = len(K_list)
    a = np.ones(n + 1)
    for i in range(1, n + 1):
        a[i] = (L_free ** i) * np.prod(K_list[:i])
    return a / a.sum()


def free_ligand_residual_specific(L_free, L_tot, P_tot, K_list):
    """Residual for root-finding [L]_free in specific-only model."""
    F = compute_F_calc_specific(L_free, K_list)
    avg = np.dot(np.arange(F.size), F)
    return L_free - (L_tot - P_tot * avg)


def solve_L_free_specific(L_tot, P_tot, K_list):
    """Finds [L]_free for specific-only model using Brentq."""
    try:
        return brentq(free_ligand_residual_specific, 0, L_tot,
                      args=(L_tot, P_tot, K_list))
    except ValueError:
        return L_tot


def residuals_specific(lnK, L_totals_M, P_tot_M, F_exps, ssr_history):
    """Residual vector for specific-only least-squares."""
    K_list = np.exp(lnK)
    res = []
    for L_tot, F_exp in zip(L_totals_M, F_exps):
        Lf = solve_L_free_specific(L_tot, P_tot_M, K_list)
        Fc = compute_F_calc_specific(Lf, K_list)
        if Fc.size > F_exp.size:
            F_exp = np.pad(F_exp, (0, Fc.size - F_exp.size), 'constant')
        res.append(Fc - F_exp)
    v = np.concatenate(res)
    ssr_history.append(v.dot(v))
    return v
