"""Binomial specific (single Ks across S sites) + Poisson NSB (Daubenfeld 2006).

Specific binding is noncooperative: S identical sites all with Ka = Ks.
Nonspecific binding is Poisson with mean lambda = Kn * [L]_free.

Parameter vector layout:
    ln_params = [ln(Kn), ln(Ks)]    (only TWO parameters; S is fixed externally)

Partition function:
    Z[i] = sum_{j=0}^{min(i,S)} C(S,j) * Ks^j * Kn^(i-j) / (i-j)!

Reference:
    Daubenfeld, T.; Bouin, A.-P.; van der Rest, G.
    J. Am. Soc. Mass Spectrom. 2006, 17, 1239-1248.
"""
import numpy as np
from math import factorial, comb
from scipy.optimize import brentq

MODEL_NAME = "binomial_poisson"


def mole_fractions(L_free_M, ln_params, S, N):
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


def n_params(S):
    return 2  # always 2 regardless of S


def initial_lnK(S, Kn=1e3, Ks=1e5):
    return np.log([Kn, Ks])


def param_labels(S):
    return ["Kn", "Ks"]


def _balance(L_free_M, L_tot_M, P_tot_M, ln_params, S, N):
    F = mole_fractions(L_free_M, ln_params, S, N)
    return L_free_M - (L_tot_M - P_tot_M * np.dot(np.arange(len(F)), F))


def free_ligand(L_tot_M, P_tot_M, ln_params, S, N):
    try:
        return brentq(_balance, 0, L_tot_M, args=(L_tot_M, P_tot_M, ln_params, S, N))
    except ValueError:
        return L_tot_M


def residual_vector(ln_params, L_totals_M, P_tot_M, F_exps, S, N, ssr_history):
    res_list = []
    for L_tot, F_exp in zip(L_totals_M, F_exps):
        Lf = free_ligand(L_tot, P_tot_M, ln_params, S, N)
        Fc = mole_fractions(Lf, ln_params, S, N)
        res_list.append(Fc - F_exp)
    vec = np.concatenate(res_list)
    ssr_history.append(float(np.dot(vec, vec)))
    return vec


# Backward-compat aliases
calculate_fractions_model = mole_fractions
solve_L_free = free_ligand
residuals = residual_vector
