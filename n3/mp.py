"""Nugget 3: one TrueSkill match update as Gaussian message passing on a factor graph.
Model (one match, team A vs team B):
  skill        s_i ~ N(mu_i, sigma_i^2)                    prior factor
  performance  p_i ~ N(s_i, beta^2)                         likelihood factor
  team perf    t_A = sum_i w_i p_i (+ h if A at home)       weighted-sum factor
  difference   d   = t_A - t_B                              difference factor
  result       1[d > 0]  (A won)                            truncation factor  (the only non-Gaussian one)
Gaussians are stored in precision form: pi = 1/var, tau = mu/var.  Product = add, division = subtract.
"""
import math
import numpy as np
from scipy.special import log_ndtr, ndtr

LOG_SQRT_2PI = 0.9189385332046727          # log of sqrt(2*pi)


def VW(t):
    """Nugget 2's functions.  V(t) = phi(t) / Phi(t),  W(t) = V(t) * (V(t) + t).
    phi(t) / Phi(t) is computed as exp(log phi(t) - log Phi(t)), so it does not break for very negative t."""
    log_phi = -0.5 * t * t - LOG_SQRT_2PI        # log of the standard normal PDF at t
    log_Phi = log_ndtr(t)                         # log of the standard normal CDF at t
    V = math.exp(log_phi - log_Phi)
    W = V * (V + t)
    return V, W


def message_passing(muA, varA, wA, muB, varB, wB, beta, h=0.0, A_won=True, log=None):
    """Algorithm 1 (MessagePassing): one match.
    Input : for each player of team A and team B: mean mu, variance var, weight w;
            beta (performance noise); h (home term for team A: +home advantage if A is at home,
            -home advantage if B is at home, 0 at a neutral site); A_won (True or False).
    Output: (new means of A, new variances of A), (new means of B, new variances of B).
    If `log` is a list, every message is added to it as (name, mean, variance), only for printing."""

    beta2 = beta * beta

    # ---- winner W and loser L, and the home term eta from the winner's side
    if A_won:
        names_W, names_L = "A", "B"
        mu_W, var_W, w_W = list(muA), list(varA), list(wA)
        mu_L, var_L, w_L = list(muB), list(varB), list(wB)
        eta = h
    else:
        names_W, names_L = "B", "A"
        mu_W, var_W, w_W = list(muB), list(varB), list(wB)
        mu_L, var_L, w_L = list(muA), list(varA), list(wA)
        eta = -h

    # ================= DOWN: predict the gap =================
    # line 1: a_i = w_i * mu_i,  b_i = w_i^2 * (sigma_i^2 + beta^2);  mu_T = sum of a_i,  v_T = sum of b_i
    a_W = []
    b_W = []
    for i in range(len(mu_W)):
        a_W.append(w_W[i] * mu_W[i])
        b_W.append(w_W[i] ** 2 * (var_W[i] + beta2))
    a_L = []
    b_L = []
    for i in range(len(mu_L)):
        a_L.append(w_L[i] * mu_L[i])
        b_L.append(w_L[i] ** 2 * (var_L[i] + beta2))

    mu_TW = 0.0
    v_TW = 0.0
    for i in range(len(a_W)):
        mu_TW = mu_TW + a_W[i]
        v_TW = v_TW + b_W[i]
    mu_TL = 0.0
    v_TL = 0.0
    for i in range(len(a_L)):
        mu_TL = mu_TL + a_L[i]
        v_TL = v_TL + b_L[i]

    # line 2: the gap d = winner - loser:  m = mu_W - mu_L + eta,  c^2 = v_W + v_L,  t = m / c
    m = mu_TW - mu_TL + eta
    c2 = v_TW + v_TL
    c = math.sqrt(c2)
    t = m / c

    # ================= CUT: the winner won, so d > 0 (Nugget 2) =================
    # line 3: m_hat = m + c * V(t),  v_hat = c^2 * (1 - W(t))
    V, W = VW(t)
    m_hat = m + c * V
    v_hat = c2 * (1 - W)

    # ================= UP: send only the evidence back =================
    # line 4: divide out the prediction:  pi_e = 1/v_hat - 1/c^2,  m_e = (m_hat/v_hat - m/c^2) / pi_e
    pi_e = 1.0 / v_hat - 1.0 / c2
    m_e = (m_hat / v_hat - m / c2) / pi_e

    # line 5: message to the winning team:  team W = d + team L
    mu_to_TW = m_e + mu_TL - eta
    v_to_TW = 1.0 / pi_e + v_TL
    # line 6: message to the losing team:   team L = team W - d
    mu_to_TL = mu_TW + eta - m_e
    v_to_TL = 1.0 / pi_e + v_TW

    # lines 7-9: for every player of both teams
    new = {}
    teams = [(names_W, mu_W, var_W, w_W, a_W, b_W, mu_TW, v_TW, mu_to_TW, v_to_TW),
             (names_L, mu_L, var_L, w_L, a_L, b_L, mu_TL, v_TL, mu_to_TL, v_to_TL)]
    for team in teams:
        name, mu, var, w, a, b, mu_T, v_T, mu_to_T, v_to_T = team
        new_mu = []
        new_var = []
        for i in range(len(mu)):
            # line 8: take the teammates out and divide by her weight, then add her own beta^2
            #         (mu_T - a_i is the teammates' part of the team mean)
            mu_to_i = (mu_to_T - mu_T + a[i]) / w[i]
            v_to_i = (v_to_T + v_T - b[i]) / w[i] ** 2 + beta2
            # line 9: multiply into her old rating: precisions add
            pi_i = 1.0 / var[i] + 1.0 / v_to_i
            mu_i = (mu[i] / var[i] + mu_to_i / v_to_i) / pi_i
            var_i = 1.0 / pi_i
            new_mu.append(mu_i)
            new_var.append(var_i)
            if log is not None:
                log.append((f"what the match says about {name}{i+1}", mu_to_i, v_to_i))
                log.append((f"new rating {name}{i+1}", mu_i, var_i))
        new[name] = (np.array(new_mu), np.array(new_var))

    if log is not None:
        log.insert(0, ("team " + names_W + " (winner), predicted", mu_TW, v_TW))
        log.insert(1, ("team " + names_L + " (loser), predicted", mu_TL, v_TL))
        log.insert(2, ("gap d, predicted", m, c2))
        log.insert(3, ("gap d after the cut (Nugget 2)", m_hat, v_hat))
        log.insert(4, ("news about d (prediction divided out)", m_e, 1.0 / pi_e))
        log.insert(5, ("message to team " + names_W, mu_to_TW, v_to_TW))
        log.insert(6, ("message to team " + names_L, mu_to_TL, v_to_TL))

    return new["A"], new["B"]


def closed_form(muA, varA, wA, muB, varB, wB, beta, h=0.0, A_won=True):
    """Nugget 2's formulas, written for weighted teams (road 2 of the verification).
    Each player moves by  +/- w_i * sigma_i^2 / c * V(t)  and her variance shrinks by the factor
    1 - w_i^2 * sigma_i^2 / c^2 * W(t)."""
    beta2 = beta * beta
    mean_A = 0.0
    var_A = 0.0
    for i in range(len(muA)):
        mean_A = mean_A + wA[i] * muA[i]
        var_A = var_A + wA[i] ** 2 * (varA[i] + beta2)
    mean_B = 0.0
    var_B = 0.0
    for i in range(len(muB)):
        mean_B = mean_B + wB[i] * muB[i]
        var_B = var_B + wB[i] ** 2 * (varB[i] + beta2)
    mu_d = mean_A - mean_B + h
    c2 = var_A + var_B
    c = math.sqrt(c2)
    if A_won:
        sign = 1.0
    else:
        sign = -1.0
    V, W = VW(sign * mu_d / c)
    newA_mu = []
    newA_var = []
    for i in range(len(muA)):
        newA_mu.append(muA[i] + sign * wA[i] * varA[i] / c * V)
        newA_var.append(varA[i] * (1 - wA[i] ** 2 * varA[i] / c2 * W))
    newB_mu = []
    newB_var = []
    for i in range(len(muB)):
        newB_mu.append(muB[i] - sign * wB[i] * varB[i] / c * V)
        newB_var.append(varB[i] * (1 - wB[i] ** 2 * varB[i] / c2 * W))
    return (np.array(newA_mu), np.array(newA_var)), (np.array(newB_mu), np.array(newB_var))


# ---- The two checks below are written with numpy arrays on purpose: they add up millions of grid
# ---- points or random samples, which plain Python loops would take minutes to do.
# ---- They are checks only; the model itself (Algorithms 1 and 2) never uses them.

def brute_force_grid(muA, varA, wA, muB, varB, wB, beta, h=0.0, A_won=True, n=None, half_width=7.0):
    """Exact posterior mean/var of every skill by direct numerical integration on a grid (Bayes' rule).
    The performance noise is integrated analytically: P(A wins | skills) = Phi((sum wA sA - sum wB sB + h)/sqrt(beta^2 sum w^2)).
    n grid values per skill over mean +/- 7 sd: 801 for 1 vs 1, 61 for 2 vs 2."""
    mus = np.r_[muA, muB]; sds = np.sqrt(np.r_[varA, varB]); ws = np.r_[wA, -np.asarray(wB, float)]
    k = len(mus)
    if n is None:
        if k == 2:
            n = 801
        elif k == 4:
            n = 61
        else:
            n = 41
    axes = []
    for j in range(k):
        axes.append(np.linspace(mus[j] - half_width * sds[j], mus[j] + half_width * sds[j], n))
    grids = np.meshgrid(*axes, indexing="ij", sparse=True)      # every combination of grid values
    logprior = 0.0
    diff = h
    for j in range(k):
        logprior = logprior - 0.5 * ((grids[j] - mus[j]) / sds[j]) ** 2      # log of the Gaussian prior
        diff = diff + ws[j] * grids[j]                                          # sum wA sA - sum wB sB + h
    if A_won:
        sign = 1.0
    else:
        sign = -1.0
    lik = ndtr(sign * diff / np.sqrt(beta ** 2 * np.sum(ws ** 2)))             # P(result | skills)
    wgt = np.exp(logprior - logprior.max()) * lik                               # prior x likelihood
    Z = wgt.sum()
    means = []
    variances = []
    for j in range(k):
        m = (wgt * grids[j]).sum() / Z
        means.append(m)
        variances.append((wgt * (grids[j] - m) ** 2).sum() / Z)
    k1 = len(muA)
    return (np.array(means[:k1]), np.array(variances[:k1])), (np.array(means[k1:]), np.array(variances[k1:]))


def monte_carlo(muA, varA, wA, muB, varB, wB, beta, h=0.0, A_won=True, N=4_000_000, seed=0):
    """Same exact posterior, by sampling skills from the prior and weighting each sample by P(result | skills)."""
    rng = np.random.default_rng(seed)
    mus = np.r_[muA, muB]; sds = np.sqrt(np.r_[varA, varB]); ws = np.r_[wA, -np.asarray(wB, float)]
    S = mus + sds * rng.standard_normal((N, len(mus)))          # N random draws of every skill
    if A_won:
        sign = 1.0
    else:
        sign = -1.0
    lik = ndtr(sign * (S @ ws + h) / np.sqrt(beta ** 2 * np.sum(ws ** 2)))
    Z = lik.sum()
    m = (lik @ S) / Z
    v = (lik @ (S - m) ** 2) / Z
    k1 = len(muA)
    return (m[:k1], v[:k1]), (m[k1:], v[k1:])
