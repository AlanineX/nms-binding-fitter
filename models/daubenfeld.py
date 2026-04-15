"""
Daubenfeld 2006 model: binomial specific (single Ks over S sites) + Poisson NSB.

Specific binding is noncooperative: S identical sites, all with the same Ka = Ks.
Nonspecific binding is Poisson with mean lambda = Kn * [L]_free.

Partition function:
    Z[i] = sum_{j=0}^{min(i,S)} C(S,j) * Ks^j * Kn^(i-j) / (i-j)!

Only two free parameters: ln(Kn), ln(Ks).
S is fixed a priori (known number of specific sites).

Reference:
    Daubenfeld, T.; Bouin, A.-P.; van der Rest, G.
    J. Am. Soc. Mass Spectrom. 2006, 17, 1239-1248.
    DOI: 10.1016/j.jasms.2006.05.005
"""
import numpy as np
from math import factorial, comb
from scipy.optimize import brentq


def calculate_fractions_model(L_free_M, ln_params, S, N):
    """Predicted mole fractions F_calc[0..S+N].

    ln_params = [ln(Kn), ln(Ks)]  (only two parameters)
    """
    params = np.exp(np.asarray(ln_params))
    Kn = params[0]
    Ks = params[1]

    Z = np.zeros(S + N + 1)
    Z[0] = 1.0
    for i in range(1, S + N + 1):
        z_sum = 0.0
        for j in range(min(i, S) + 1):
            m = i - j
            z_sum += comb(S, j) * (Ks ** j) * (Kn ** m) / factorial(m)
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
