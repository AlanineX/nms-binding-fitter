"""Stepwise specific Ks_i + Poisson NSB (Shimon-Sharon-Horovitz 2010).

Differs from geometric_nsb (nonspecific.py) by adding 1/(i-j)! factorial
weighting on the nonspecific term (Poisson distribution with mean
lambda = Kn * [L]_free).

Parameter vector layout:
    ln_params = [ln(Kn), ln(Ks_1), ..., ln(Ks_S)]

Partition function:
    Z[i] = sum_{j=0}^{min(i,S)} prod(Ks_1..Ks_j) * Kn^(i-j) / (i-j)!

Reference:
    Shimon, L.; Sharon, M.; Horovitz, A. Biophys. J. 2010, 99, 1645-1649.
"""
import numpy as np
from math import factorial
from scipy.optimize import brentq

MODEL_NAME = "poisson_nsb"


def mole_fractions(L_free_M, ln_params, S, N):
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


def n_params(S):
    return S + 1


def initial_lnK(S, Kn=1e3, Ks=1e5):
    return np.log(np.concatenate(([Kn], np.full(S, Ks))))


def param_labels(S):
    return ["Kn"] + [f"Ks_{i+1}" for i in range(S)]


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
