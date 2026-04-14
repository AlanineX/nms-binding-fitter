"""
Mixed specific/nonspecific binding model + deconvolution math.
"""
import numpy as np
from scipy.optimize import brentq


def calculate_fractions_model(L_free_M, ln_params, S, N):
    """
    Computes predicted mole fractions (F_calc) for a given [L]_free.
    This function contains the specific/non-specific binding model logic.
    """
    params = np.exp(np.asarray(ln_params))
    Kn = params[0]
    Ks = params[1:]  # Ks = [Ks1, Ks2, ..., KsS]

    Z = np.zeros(S + N + 1)
    Z[0] = 1.0  # By definition

    for i in range(1, S + N + 1):
        z_sum = 0
        for j in range(min(i, S) + 1):
            prod_ks = np.prod(Ks[:j]) if j > 0 else 1.0
            term = prod_ks * (Kn**(i - j))
            z_sum += term
        Z[i] = z_sum

    alpha = Z * (L_free_M ** np.arange(S + N + 1))
    return alpha / alpha.sum()


def compute_deconvolution_weights(Kn, Ks, S, N):
    """
    For each apparent bound count i, compute weights over j specific bindings.
    weights[i, j] = (K_s1...K_sj) * K_n^(i-j), with j in [0, min(i, S)].
    """
    prod_ks = np.concatenate(([1.0], np.cumprod(Ks)))
    weights = np.zeros((S + N + 1, S + 1))
    for i in range(S + N + 1):
        max_j = min(i, S)
        for j in range(max_j + 1):
            weights[i, j] = prod_ks[j] * (Kn ** (i - j))
    return weights


def deconvolve_fractions(F_vals, weights, S):
    """
    Deconvolve apparent fractions F_vals[i] into contributions by j-specific.
    Returns:
      frac_within[i, j]: fraction of apparent i that is j-specific
      contrib[i, j]: fraction of total population in (i, j) bucket
    """
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


def free_ligand_residual(L_free_M, L_tot_M, P_tot_M, ln_params, S, N):
    """Residual for root-finding [L]_free. [L]_free = [L]_tot - [L]_bound."""
    F_calc = calculate_fractions_model(L_free_M, ln_params, S, N)
    avg_bound = np.dot(np.arange(len(F_calc)), F_calc)
    return L_free_M - (L_tot_M - P_tot_M * avg_bound)


def solve_L_free(L_tot_M, P_tot_M, ln_params, S, N):
    """Finds [L]_free that satisfies mass balance using a root-finder."""
    try:
        return brentq(free_ligand_residual, 0, L_tot_M,
                      args=(L_tot_M, P_tot_M, ln_params, S, N))
    except ValueError:
        return L_tot_M


def residuals(ln_params, L_totals_M, P_tot_M, F_exps, S, N, ssr_history):
    """
    Computes the residual vector for the least-squares optimizer.
    This is the core function called repeatedly by the optimizer.
    """
    res_list = []
    for L_tot_M, F_exp in zip(L_totals_M, F_exps):
        L_free_M = solve_L_free(L_tot_M, P_tot_M, ln_params, S, N)
        F_calc = calculate_fractions_model(L_free_M, ln_params, S, N)
        res_list.append(F_calc - F_exp)

    vec = np.concatenate(res_list)
    ssr_history.append(np.dot(vec, vec))
    return vec


def debug_validate_point(idx, L_tot_M, L_free_M, lnK_opt, F_calc, S, N, cfg):
    """Validate fit at a single data point — prints detailed diagnostics."""
    params = np.exp(lnK_opt)
    Kn = params[0]
    Ks = params[1:]
    Z_manual = []
    for i in range(S + N + 1):
        z_sum = 0.0
        for j in range(min(i, S) + 1):
            prod_ks = np.prod(Ks[:j]) if j > 0 else 1.0
            z_sum += prod_ks * (Kn ** (i - j))
        Z_manual.append(z_sum)
    Z_manual = np.array(Z_manual)
    alpha_manual = Z_manual * (L_free_M ** np.arange(S + N + 1))
    F_manual = alpha_manual / alpha_manual.sum()
    max_diff = np.max(np.abs(F_manual - F_calc))

    avg_bound = np.dot(np.arange(len(F_calc)), F_calc)
    L_free_check = L_tot_M - cfg.p_total_m * avg_bound
    balance_err = float(L_free_M - L_free_check)

    weights = compute_deconvolution_weights(Kn, Ks, S, N)
    frac_within, contrib = deconvolve_fractions(F_calc, weights, S)
    valid = np.array([1.0 if weights[i, :min(i, S) + 1].sum() > 0 else 0.0 for i in range(S + N + 1)])
    max_within_err = np.max(np.abs(frac_within.sum(axis=1) - valid))
    max_total_err = np.max(np.abs(contrib.sum(axis=1) - F_calc))

    print("\n--- DEBUG VALIDATION ---")
    print(f"Row index: {idx}")
    print(f"L_tot (M): {L_tot_M:.6e}")
    print(f"L_free (M): {L_free_M:.6e}")
    print(f"Mass-balance error (M): {balance_err:.3e}")
    print(f"Max |F_manual - F_calc|: {max_diff:.3e}")
    print(f"Max |sum_j contrib(i,j) - F_i|: {max_total_err:.3e}")
    print(f"Max |sum_j frac_within(i,j) - 1|: {max_within_err:.3e}")

    i_focus = min(max(cfg.debug_i_index, 0), S + N)
    max_j = min(i_focus, S)
    Z_i = Z_manual[i_focus]
    alpha_i = alpha_manual[i_focus]
    denom = alpha_manual.sum()
    F_i_manual = alpha_i / denom if denom > 0 else np.nan
    print(f"I{i_focus} manual: Z_i={Z_i:.3e}, alpha_i={alpha_i:.3e}, denom={denom:.3e}, F_i={F_i_manual:.3e}")
    parts = []
    for j in range(max_j + 1):
        parts.append(f"{j} spec + {i_focus - j} non: {100.0 * frac_within[i_focus, j]:.1f}%")
    print(f"I{i_focus} composition: " + "; ".join(parts))
