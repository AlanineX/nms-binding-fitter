"""
Shimon-Sharon-Horovitz 2010 model: stepwise specific binding + Poisson nonspecific.

Differs from the current mixed model (nonspecific.py) by adding a 1/(i-j)!
factorial weighting on the nonspecific term, which corresponds to a Poisson
distribution of nonspecific ligands with mean lambda = Kn * [L]_free.

Partition function:
    Z[i] = sum_{j=0}^{min(i,S)} prod(Ks_1..Ks_j) * Kn^(i-j) / (i-j)!

Reference:
    Shimon, L.; Sharon, M.; Horovitz, A. Biophys. J. 2010, 99, 1645-1649.
    DOI: 10.1016/j.bpj.2010.06.062
"""
import numpy as np
from math import factorial
from scipy.optimize import brentq


def calculate_fractions_model(L_free_M, ln_params, S, N):
    """Predicted mole fractions F_calc[0..S+N] for Shimon Poisson-NSB model.

    ln_params = [ln(Kn), ln(Ks_1), ..., ln(Ks_S)]
    """
    params = np.exp(np.asarray(ln_params))
    Kn = params[0]
    Ks = params[1:]

    Z = np.zeros(S + N + 1)
    Z[0] = 1.0
    for i in range(1, S + N + 1):
        z_sum = 0.0
        for j in range(min(i, S) + 1):
            prod_ks = np.prod(Ks[:j]) if j > 0 else 1.0
            m = i - j
            z_sum += prod_ks * (Kn ** m) / factorial(m)
        Z[i] = z_sum

    alpha = Z * (L_free_M ** np.arange(S + N + 1))
    return alpha / alpha.sum()


def free_ligand_residual(L_free_M, L_tot_M, P_tot_M, ln_params, S, N):
    F_calc = calculate_fractions_model(L_free_M, ln_params, S, N)
    avg_bound = np.dot(np.arange(len(F_calc)), F_calc)
    return L_free_M - (L_tot_M - P_tot_M * avg_bound)


def solve_L_free(L_tot_M, P_tot_M, ln_params, S, N):
    try:
        return brentq(free_ligand_residual, 0, L_tot_M,
                      args=(L_tot_M, P_tot_M, ln_params, S, N))
    except ValueError:
        return L_tot_M


def residuals(ln_params, L_totals_M, P_tot_M, F_exps, S, N, ssr_history):
    res_list = []
    for L_tot_M, F_exp in zip(L_totals_M, F_exps):
        L_free_M = solve_L_free(L_tot_M, P_tot_M, ln_params, S, N)
        F_calc = calculate_fractions_model(L_free_M, ln_params, S, N)
        res_list.append(F_calc - F_exp)
    vec = np.concatenate(res_list)
    ssr_history.append(np.dot(vec, vec))
    return vec
