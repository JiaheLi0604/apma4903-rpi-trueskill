"""Nugget 3: one TrueSkill match update as Gaussian message passing on a factor graph.
Model (one match, team A vs team B):
  skill        s_i ~ N(mu_i, sigma_i^2)                    prior factor
  performance  p_i ~ N(s_i, beta^2)                         likelihood factor
  team perf    t_A = sum_i w_i p_i (+ h if A at home)       weighted-sum factor
  difference   d   = t_A - t_B                              difference factor
  result       1[d > 0]  (A won)                            truncation factor  (the only non-Gaussian one)
Gaussians are stored in precision form: pi = 1/var, tau = mu/var.  Product = add, division = subtract.
"""
import numpy as np
from scipy.special import log_ndtr

LOG_SQRT_2PI = 0.9189385332046727

class G:
    """A Gaussian in precision form."""
    def __init__(s, pi=0.0, tau=0.0): s.pi, s.tau = float(pi), float(tau)
    @staticmethod
    def mv(mu, var): return G(1.0/var, mu/var)
    @property
    def mu(s): return s.tau/s.pi
    @property
    def var(s): return 1.0/s.pi
    def __mul__(s, o): return G(s.pi+o.pi, s.tau+o.tau)        # product of Gaussians: precisions add
    def __truediv__(s, o): return G(s.pi-o.pi, s.tau-o.tau)    # division: precisions subtract
    def __repr__(s): return f"N(mean={s.mu:.4f}, sd={np.sqrt(s.var):.4f})"

def VW(t):
    v = float(np.exp(-0.5*t*t - LOG_SQRT_2PI - log_ndtr(t))); return v, v*(v+t)

def message_passing(muA, varA, wA, muB, varB, wB, beta, h=0.0, A_won=True, log=None):
    """Return posterior (mu, var) for every player of A and B.  `log` (a list) collects every message."""
    def L(name, g):
        if log is not None: log.append((name, g.mu, g.var))
        return g
    b2 = beta*beta
    if not A_won:                                         # relabel so that the winner is always 'A'
        (mB2, vB2), (mA2, vA2) = message_passing(muB, varB, wB, muA, varA, wA, beta, -h, True, log)
        return (mA2, vA2), (mB2, vB2)
    teams = {"A": (np.asarray(muA,float), np.asarray(varA,float), np.asarray(wA,float)),
             "B": (np.asarray(muB,float), np.asarray(varB,float), np.asarray(wB,float))}
    # 1) prior factor -> skill;  2) skill -> performance (add beta^2)
    prior, perf_down, team_down = {}, {}, {}
    for k,(m,v,w) in teams.items():
        prior[k] = [L(f"prior->s_{k}{i+1}", G.mv(m[i], v[i])) for i in range(len(m))]
        perf_down[k] = [L(f"s_{k}{i+1}->p_{k}{i+1}", G.mv(g.mu, g.var+b2)) for i,g in enumerate(prior[k])]
        # 3) weighted-sum factor -> team performance:  mean sum w*mu, var sum w^2*var
        tm = sum(w[i]*perf_down[k][i].mu for i in range(len(m))) + (h if k=="A" else 0.0)
        tv = sum(w[i]**2*perf_down[k][i].var for i in range(len(m)))
        team_down[k] = L(f"sum->t_{k}", G.mv(tm, tv))
    # 4) difference factor -> d
    d_in = L("diff->d", G.mv(team_down["A"].mu-team_down["B"].mu, team_down["A"].var+team_down["B"].var))
    # 5) truncation factor: moment-match the truncated marginal of d (uses V, W from Nugget 2)
    c = np.sqrt(d_in.var); t = d_in.mu/c; V, W = VW(t)
    d_post = L("d marginal (moment-matched)", G.mv(d_in.mu + c*V, d_in.var*(1-W)))
    d_up = L("trunc->d (= marginal / incoming)", d_post / d_in)
    # 6) difference factor -> team performances (going back up)
    team_up = {"A": L("diff->t_A", G.mv(d_up.mu + team_down["B"].mu, d_up.var + team_down["B"].var)),
               "B": L("diff->t_B", G.mv(team_down["A"].mu - d_up.mu, d_up.var + team_down["A"].var))}
    out = {}
    for k,(m,v,w) in teams.items():
        tu = team_up[k]; base = tu.mu - (h if k=="A" else 0.0)
        post_mu, post_var = [], []
        for i in range(len(m)):
            others_m = sum(w[j]*perf_down[k][j].mu for j in range(len(m)) if j!=i)
            others_v = sum(w[j]**2*perf_down[k][j].var for j in range(len(m)) if j!=i)
            # 7) weighted-sum factor -> p_i :  p_i = (t - sum_{j!=i} w_j p_j)/w_i
            p_up = L(f"sum->p_{k}{i+1}", G.mv((base-others_m)/w[i], (tu.var+others_v)/w[i]**2))
            # 8) performance -> skill (add beta^2);  9) posterior = prior x upward message
            s_up = L(f"p_{k}{i+1}->s_{k}{i+1}", G.mv(p_up.mu, p_up.var+b2))
            post = L(f"posterior s_{k}{i+1}", prior[k][i]*s_up)
            post_mu.append(post.mu); post_var.append(post.var)
        out[k] = (np.array(post_mu), np.array(post_var))
    return out["A"], out["B"]

def closed_form(muA, varA, wA, muB, varB, wB, beta, h=0.0, A_won=True):
    """Nugget 2's formulas, written for weighted teams."""
    muA,varA,wA,muB,varB,wB = map(lambda x: np.asarray(x,float), (muA,varA,wA,muB,varB,wB))
    b2 = beta*beta
    mu_d = wA@muA - wB@muB + h; c2 = (wA**2)@(varA+b2) + (wB**2)@(varB+b2)
    s = 1.0 if A_won else -1.0; c = np.sqrt(c2); V, W = VW(s*mu_d/c)
    nA = (muA + s*wA*varA/c*V, varA*(1-wA**2*varA/c2*W))
    nB = (muB - s*wB*varB/c*V, varB*(1-wB**2*varB/c2*W))
    return nA, nB

def brute_force_grid(muA, varA, wA, muB, varB, wB, beta, h=0.0, A_won=True, n=None, half_width=7.0):
    """Exact posterior mean/var of every skill by direct numerical integration on a grid.
    The performance noise is integrated analytically: P(A wins | skills) = Phi((sum wA sA - sum wB sB + h)/sqrt(beta^2 sum w^2)).
    Everything else is a plain sum over grid points: no Gaussian approximation anywhere."""
    from scipy.special import ndtr
    mus = np.r_[muA, muB]; sds = np.sqrt(np.r_[varA, varB]); ws = np.r_[wA, -np.asarray(wB,float)]
    k = len(mus); n = n or {2: 801, 4: 61}.get(k, 41)
    axes = [np.linspace(m-half_width*s, m+half_width*s, n) for m,s in zip(mus, sds)]
    grids = np.meshgrid(*axes, indexing="ij", sparse=True)
    logprior = sum(-0.5*((g-m)/s)**2 for g,m,s in zip(grids, mus, sds))
    diff = sum(w*g for w,g in zip(ws, grids)) + h
    lik = ndtr((1 if A_won else -1)*diff/np.sqrt(beta**2*np.sum(ws**2)))
    wgt = np.exp(logprior - logprior.max())*lik; Z = wgt.sum()
    means, variances = [], []
    for g in grids:
        m = (wgt*g).sum()/Z; means.append(m); variances.append((wgt*(g-m)**2).sum()/Z)
    k1 = len(muA)
    return (np.array(means[:k1]), np.array(variances[:k1])), (np.array(means[k1:]), np.array(variances[k1:]))

def monte_carlo(muA, varA, wA, muB, varB, wB, beta, h=0.0, A_won=True, N=4_000_000, seed=0):
    """Same exact posterior, by sampling skills from the prior and weighting by P(result | skills)."""
    from scipy.special import ndtr
    rng = np.random.default_rng(seed)
    mus = np.r_[muA, muB]; sds = np.sqrt(np.r_[varA, varB]); ws = np.r_[wA, -np.asarray(wB,float)]
    S = mus + sds*rng.standard_normal((N, len(mus)))
    lik = ndtr((1 if A_won else -1)*(S@ws + h)/np.sqrt(beta**2*np.sum(ws**2)))
    Z = lik.sum(); m = (lik@S)/Z; v = (lik@(S-m)**2)/Z
    k1 = len(muA)
    return (m[:k1], v[:k1]), (m[k1:], v[k1:])
