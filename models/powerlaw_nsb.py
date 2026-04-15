"""
Guan 2015 model: stepwise specific Ka + power-law decaying NSB Ka.

For ligand binding step k (1-indexed), the apparent stepwise association
constant is the sum of a specific contribution (only for k <= S) and a
nonspecific contribution that decays as a power-law with k:

    Kn_k = beta / k^gamma

    K_app_k = Ks_k + Kn_k    for k <= S
    K_app_k = Kn_k           for k > S

Partition function via successive stepwise products:
    alpha_j = L^j * prod_{k=1..j} K_app_k

The model reduces to the Shimon constant-NSB case when gamma = 0.

Free parameters: ln(beta), gamma (linear, can be 0 or positive), ln(Ks_1..Ks_S).
Parameter vector layout:
    ln_params = [ln(beta), gamma, ln(Ks_1), ..., ln(Ks_S)]
    NOTE: gamma is NOT in log space (so it can be exactly 0).

Reference:
    Guan, S.; Trnka, M. J.; Bushnell, D. A.; Robinson, P. J. J.;
    Gestwicki, J. E.; Burlingame, A. L.
    Anal. Chem. 2015, 87, 8541-8546. DOI: 10.1021/acs.analchem.5b02258
"""
import numpy as np
from scipy.optimize import brentq


def _unpack(ln_params):
    """Unpack [ln(beta), gamma, ln(Ks_1), ..., ln(Ks_S)] into beta, gamma, Ks."""
    arr = np.asarray(ln_params)
    beta = np.exp(arr[0])
    gamma = arr[1]
    Ks = np.exp(arr[2:])
    return beta, gamma, Ks


def calculate_fractions_model(L_free_M, ln_params, S, N):
    """Predicted mole fractions F_calc[0..S+N] for Guan power-law model."""
    beta, gamma, Ks = _unpack(ln_params)
    n_max = S + N

    # Build apparent stepwise Ka values: K_app_k for k = 1..n_max
    K_app = np.zeros(n_max + 1)  # index 0 unused
    for k in range(1, n_max + 1):
        Kn_k = beta / (k ** gamma)
        if k <= S:
            K_app[k] = Ks[k - 1] + Kn_k
        else:
            K_app[k] = Kn_k

    # Successive products: alpha_j = L^j * prod K_app_1..j
    alpha = np.zeros(n_max + 1)
    alpha[0] = 1.0
    for j in range(1, n_max + 1):
        alpha[j] = alpha[j - 1] * L_free_M * K_app[j]

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
