"""Specific-only stepwise binding model (no nonspecific component).

Parameter vector layout:
    ln_params = [ln(Ks_1), ln(Ks_2), ..., ln(Ks_S)]

Partition function:
    alpha[i] = L_free^i * prod(Ks_1..Ks_i)
    F[i] = alpha[i] / sum(alpha)

N is accepted in every signature for interface uniformity but ignored.
"""
import numpy as np
from scipy.optimize import brentq

MODEL_NAME = "specific_binding"


def mole_fractions(L_free_M, ln_params, S, N):
    K = np.exp(np.asarray(ln_params))
    n = len(K)
    a = np.ones(n + 1)
    for i in range(1, n + 1):
        a[i] = (L_free_M ** i) * np.prod(K[:i])
    return a / a.sum()


def n_params(S):
    return S


def initial_lnK(S, Kn=None, Ks=1e5):
    return np.full(S, np.log(Ks))


def param_labels(S):
    return [f"Ks_{i+1}" for i in range(S)]


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
        if Fc.size > F_exp.size:
            F_exp = np.pad(F_exp, (0, Fc.size - F_exp.size), "constant")
        res_list.append(Fc - F_exp)
    vec = np.concatenate(res_list)
    ssr_history.append(float(np.dot(vec, vec)))
    return vec
