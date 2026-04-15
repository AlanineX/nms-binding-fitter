"""Stepwise specific + power-law nonspecific Kn_k = beta/k^gamma (Guan 2015).

For ligand binding step k (1-indexed), apparent stepwise Ka:
    K_app_k = Ks_k + beta/k^gamma   for k <= S
    K_app_k = beta/k^gamma           for k > S

Partition function via successive stepwise products:
    alpha[j] = L_free^j * prod_{k=1..j} K_app_k

Reduces to constant-nonspecific (Shimon-like) when gamma = 0.

Parameter vector layout:
    ln_params = [ln(beta), gamma, ln(Ks_1), ..., ln(Ks_S)]
    NOTE: gamma is in linear space (so it can be exactly 0).

Reference:
    Guan, S. et al. Anal. Chem. 2015, 87, 8541-8546.
"""
import numpy as np
from scipy.optimize import brentq

MODEL_NAME = "power_law_nonspecific"


def _unpack(ln_params):
    arr = np.asarray(ln_params)
    beta = np.exp(arr[0])
    gamma = arr[1]
    Ks = np.exp(arr[2:])
    return beta, gamma, Ks


def mole_fractions(L_free_M, ln_params, S, N):
    beta, gamma, Ks = _unpack(ln_params)
    n_max = S + N

    K_app = np.zeros(n_max + 1)
    for k in range(1, n_max + 1):
        denom = k ** gamma
        Kn_k = beta / denom if denom > 0 and np.isfinite(denom) else 0.0
        K_app[k] = (Ks[k - 1] if k <= S else 0.0) + Kn_k

    alpha = np.zeros(n_max + 1)
    alpha[0] = 1.0
    for j in range(1, n_max + 1):
        alpha[j] = alpha[j - 1] * L_free_M * K_app[j]

    total = alpha.sum()
    if total <= 0 or not np.isfinite(total):
        out = np.zeros(n_max + 1)
        out[0] = 1.0
        return out
    return alpha / total


def n_params(S):
    return S + 2


def initial_lnK(S, Kn=1e3, Ks=1e5, gamma=0.5):
    return np.concatenate(([np.log(Kn), gamma], np.log(np.full(S, Ks))))


def param_labels(S):
    return ["beta", "gamma"] + [f"Ks_{i+1}" for i in range(S)]


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
