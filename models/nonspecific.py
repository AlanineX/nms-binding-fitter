"""Geometric-NSB model: stepwise specific Ks_i + single Kn raised to (i-j)
power without combinatorial weighting.

Parameter vector layout:
    ln_params = [ln(Kn), ln(Ks_1), ..., ln(Ks_S)]

Partition function:
    Z[i] = sum_{j=0}^{min(i,S)} prod(Ks_1..Ks_j) * Kn^(i-j)
    F[i] = Z[i] * L_free^i / sum

This is the original mixed model in this package. The Poisson alternative
(adds 1/(i-j)! weighting) is in poisson_nsb.py.
"""
import numpy as np
from scipy.optimize import brentq

MODEL_NAME = "geometric_nsb"


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
            z_sum += prod_ks * (Kn ** (i - j))
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


# --- Deconvolution helpers (specific to this model) ---

def compute_deconvolution_weights(Kn, Ks, S, N):
    prod_ks = np.concatenate(([1.0], np.cumprod(Ks)))
    weights = np.zeros((S + N + 1, S + 1))
    for i in range(S + N + 1):
        max_j = min(i, S)
        for j in range(max_j + 1):
            weights[i, j] = prod_ks[j] * (Kn ** (i - j))
    return weights


def deconvolve_fractions(F_vals, weights, S):
    num_species = len(F_vals)
    frac_within = np.zeros((num_species, S + 1))
    contrib = np.zeros((num_species, S + 1))
    for i in range(num_species):
        max_j = min(i, S)
        denom = weights[i, :max_j + 1].sum()
        if denom <= 0:
            continue
        frac = weights[i, :max_j + 1] / denom
        frac_within[i, :max_j + 1] = frac
        contrib[i, :max_j + 1] = F_vals[i] * frac
    return frac_within, contrib


def debug_validate_point(idx, L_tot_M, L_free_M, lnK_opt, F_calc, S, N, cfg):
    """Validate fit at a single data point — prints detailed diagnostics."""
    params = np.exp(lnK_opt)
    Kn = params[0]
    Ks = params[1:]
    weights = compute_deconvolution_weights(Kn, Ks, S, N)
    frac_within, contrib = deconvolve_fractions(F_calc, weights, S)

    avg_bound = float(np.dot(np.arange(len(F_calc)), F_calc))
    L_free_check = L_tot_M - cfg.p_total_m * avg_bound
    balance_err = float(L_free_M - L_free_check)
    valid = np.array([1.0 if weights[i, :min(i, S) + 1].sum() > 0 else 0.0
                      for i in range(S + N + 1)])
    max_within_err = float(np.max(np.abs(frac_within.sum(axis=1) - valid)))
    max_total_err = float(np.max(np.abs(contrib.sum(axis=1) - F_calc)))

    print("\n--- DEBUG VALIDATION ---")
    print(f"Row index: {idx}")
    print(f"L_tot (M): {L_tot_M:.6e}, L_free (M): {L_free_M:.6e}")
    print(f"Mass-balance error (M): {balance_err:.3e}")
    print(f"Max |sum_j contrib(i,j) - F_i|: {max_total_err:.3e}")
    print(f"Max |sum_j frac_within(i,j) - 1|: {max_within_err:.3e}")

    i_focus = min(max(cfg.debug_i_index, 0), S + N)
    parts = [f"{j} spec + {i_focus - j} non: {100.0 * frac_within[i_focus, j]:.1f}%"
             for j in range(min(i_focus, S) + 1)]
    print(f"I{i_focus} composition: " + "; ".join(parts))


# Backward-compat aliases (for fitting.py)
calculate_fractions_model = mole_fractions
solve_L_free = free_ligand
residuals = residual_vector
