"""TrueSkill Through Time (TTT) for volleyball teams with playing-time weights.

Model (same as the online Nugget 2 model, plus links through time):
    skill of player i in time step t :  s_i^t
    first appearance                 :  s_i ~ N(mu0, sigma0^2)
    between two of her time steps    :  s_i^t | s_i^{t'} ~ N(s_i^{t'}, drift)   (tau_w^2 per week inside a season,
                                                                                tau_s^2 across a season break)
    a match                          :  d = sum_A w p - sum_B w p + h*Home,  p = s + N(0, beta^2);  A wins iff d > 0

Inference: expectation propagation (EP).  Every Gaussian is stored in natural parameters
    pi  = 1/sigma^2          (precision)
    eta = mu/sigma^2         (precision-adjusted mean)
so that multiplying Gaussians = adding (pi, eta) and dividing = subtracting.

Each player-time node keeps three incoming messages:
    fwd : from her previous time step (or the prior)        -> carries the past
    bwd : from her next time step                            -> carries the future
    lik : sum of the messages from the matches in this step  -> carries this week's results
belief = fwd * bwd * lik  (i.e. add the natural parameters).
"""
import numpy as np
from math import sqrt
from scipy.special import ndtr
from ts_core import v_w, on_court_weights


# ------------------------------------------------------------------ 1. lineups: identical to the online engine
def record_lineups(D, P):
    """Replay the online engine's playing-time memory and record, for every match, the players of each side
    and their weights (from PAST sets only).  TTT keeps these weights fixed; only the skills are re-estimated."""
    decay = P["decay"]
    sets = np.zeros(D["n_players"]); team_sets = np.zeros(D["n_teams"]); seen = np.zeros(D["n_players"], bool)
    roster = [[] for _ in range(D["n_teams"])]
    lineups = [None] * D["n_matches"]

    def expected_lineup(tid):
        r = np.array(roster[tid], dtype=int); ts = team_sets[tid]
        if r.size == 0 or ts <= 0: return r[:0], np.zeros(0)
        w = np.minimum(sets[r] / ts, 1.0); keep = w > 0
        if not keep.any(): return r[:0], np.zeros(0)
        return r[keep], on_court_weights(w[keep])

    cur = None
    for di in D["days"]:
        season = D["season_of_day"].get(di, cur)
        if cur is not None and season != cur:
            sets *= decay; team_sets *= decay
        cur = season
        box = D["box"].get(di, {})
        for tid, (ids, s) in box.items():
            new = ids[~seen[ids]]
            if new.size: seen[new] = True; roster[tid].extend(new.tolist())
        for i in D["m_by_day"].get(di, []):
            lineups[i] = (expected_lineup(D["A"][i]), expected_lineup(D["B"][i]))
        for tid, (ids, s) in box.items():
            sets[ids] += s; team_sets[tid] += s.max()
    return lineups


# ------------------------------------------------------------------ 2. build the graph of player-time nodes
def build_graph(D, lineups, batch="week", use=None):
    """batch='week' groups matches by calendar week; batch='day' by day (used to check against the online model)."""
    dates = D["date_list"]; d0 = dates[0]
    n = D["n_matches"]
    step_of_day = {}
    for di in D["days"]:
        step_of_day[di] = di if batch == "day" else (dates[di] - d0).days // 7
    m_step = np.array([step_of_day[D["m_di"][i]] for i in range(n)])
    m_season = D["m_season"]
    if use is None: use = np.ones(n, bool)
    steps = np.unique(m_step[use])
    step_season = {int(s): int(m_season[m_step == s][0]) for s in steps}

    node_of = {}; node_player = []; node_step = []
    side = []                                        # per match: (nodesA, wA, nodesB, wB)
    for i in range(n):
        if not use[i]:
            side.append((np.zeros(0, int), np.zeros(0), np.zeros(0, int), np.zeros(0))); continue
        (ia, wa), (ib, wb) = lineups[i]
        st = int(m_step[i]); out = []
        for idx in (ia, ib):
            ids = []
            for p in idx:
                k = (int(p), st)
                if k not in node_of:
                    node_of[k] = len(node_player); node_player.append(int(p)); node_step.append(st)
                ids.append(node_of[k])
            out.append(np.array(ids, dtype=int))
        side.append((out[0], wa, out[1], wb))
    node_player = np.array(node_player); node_step = np.array(node_step)
    N = len(node_player)

    # time links: for each player, her nodes in order
    prev = np.full(N, -1); nxt = np.full(N, -1); same_season_gap = np.zeros(N); cross = np.zeros(N)
    order = np.lexsort((node_step, node_player))
    for a, b in zip(order[:-1], order[1:]):
        if node_player[a] == node_player[b]:
            prev[b] = a; nxt[a] = b
            if step_season[int(node_step[a])] == step_season[int(node_step[b])]:
                same_season_gap[b] = node_step[b] - node_step[a]
            else:                                    # number of season breaks crossed
                cross[b] = step_season[int(node_step[b])] - step_season[int(node_step[a])]
    # a player is "registered" on her first box-score day (as in the online engine); if her first rated match
    # comes in a later season, the season drift for the breaks in between is added to her prior
    reg_season = {}
    for di in D["days"]:
        for tid, (ids, s) in D["box"].get(di, {}).items():
            for p in ids:
                reg_season.setdefault(int(p), D["season_of_day"][di])
    prior_breaks = np.zeros(N)
    first = prev < 0
    prior_breaks[first] = [step_season[int(node_step[k])] - reg_season[int(node_player[k])] for k in np.where(first)[0]]
    # matches and nodes grouped by step, in the original (online) order
    matches_in = {int(s): list(np.where((m_step == s) & use)[0]) for s in steps}
    nodes_in = {int(s): np.where(node_step == s)[0] for s in steps}
    return dict(N=N, node_player=node_player, node_step=node_step, prev=prev, nxt=nxt,
                gap=same_season_gap, cross=cross, prior_breaks=prior_breaks, side=side, m_step=m_step, steps=[int(s) for s in steps],
                matches_in=matches_in, nodes_in=nodes_in, step_season=step_season)


# ------------------------------------------------------------------ 3. the TTT algorithm
class TTT:
    def __init__(self, D, G, P, tau_w=0.0, fallback=(6.0, 1.9)):
        self.D, self.G = D, G
        self.mu0, self.s0 = P["mu0"], P["sigma0"]
        self.beta2 = P["beta"] ** 2; self.h = P["home"]
        N = G["N"]
        # drift variance on the link INTO each node from its previous node
        self.drift = np.where(G["cross"] > 0, G["cross"] * P["tau_season"] ** 2, (tau_w ** 2) * G["gap"])
        z = lambda: np.zeros(N)
        self.fwd_pi, self.fwd_eta = z(), z()          # message from the past (or the prior)
        self.bwd_pi, self.bwd_eta = z(), z()          # message from the future
        self.lik_pi, self.lik_eta = z(), z()          # sum of this step's match messages
        first = G["prev"] < 0                          # first appearance: the prior
        v0 = self.s0 ** 2 + G["prior_breaks"][first] * P["tau_season"] ** 2
        self.fwd_pi[first] = 1.0 / v0; self.fwd_eta[first] = self.mu0 / v0
        # one stored message per (match, player) = per entry of each side
        self.site = []
        for (na, wa, nb, wb) in G["side"]:
            self.site.append([np.zeros(na.size), np.zeros(na.size), np.zeros(nb.size), np.zeros(nb.size)])
        self.fb_mu = fallback[0] * self.mu0; self.fb_V = fallback[1] * (self.s0 ** 2 + self.beta2)
        self.p_first = np.full(D["n_matches"], np.nan)   # prediction made on the very first forward visit

    # --- messages along time
    def forward_msg(self, nodes):
        """fwd into node k = (fwd_prev * lik_prev), i.e. everything the previous node knows except the future,
        pushed through the drift (variance + drift)."""
        p = self.G["prev"][nodes]; has = p >= 0
        k, p = nodes[has], p[has]
        q_pi = self.fwd_pi[p] + self.lik_pi[p]; q_eta = self.fwd_eta[p] + self.lik_eta[p]
        var = 1.0 / q_pi + self.drift[k]
        self.fwd_pi[k] = 1.0 / var; self.fwd_eta[k] = (q_eta / q_pi) / var

    def backward_msg(self, nodes):
        """bwd into node k = (bwd_next * lik_next) pushed back through the drift on the link k -> next."""
        nx = self.G["nxt"][nodes]; has = nx >= 0
        k, nx = nodes[has], nx[has]
        q_pi = self.bwd_pi[nx] + self.lik_pi[nx]; q_eta = self.bwd_eta[nx] + self.lik_eta[nx]
        ok = q_pi > 0
        var = 1.0 / q_pi[ok] + self.drift[nx[ok]]
        self.bwd_pi[k[ok]] = 1.0 / var; self.bwd_eta[k[ok]] = (q_eta[ok] / q_pi[ok]) / var
        self.bwd_pi[k[~ok]] = 0.0; self.bwd_eta[k[~ok]] = 0.0

    def belief(self, nodes):
        pi = self.fwd_pi[nodes] + self.bwd_pi[nodes] + self.lik_pi[nodes]
        eta = self.fwd_eta[nodes] + self.bwd_eta[nodes] + self.lik_eta[nodes]
        with np.errstate(divide="ignore", invalid="ignore"):
            return eta / pi, 1.0 / pi

    # --- one EP site update for one match
    def match_step(self, i, first_visit=False):
        na, wa, nb, wb = self.G["side"][i]; s = self.site[i]
        cav = []
        for nodes, sp, se in ((na, s[0], s[1]), (nb, s[2], s[3])):
            pi = self.fwd_pi[nodes] + self.bwd_pi[nodes] + self.lik_pi[nodes] - sp     # cavity = belief / old site
            eta = self.fwd_eta[nodes] + self.bwd_eta[nodes] + self.lik_eta[nodes] - se
            cav.append((eta / pi, 1.0 / pi))
        (ma, va), (mb, vb) = cav
        if na.size: mA, VA = float(wa @ ma), float((wa * wa) @ (va + self.beta2))
        else:       mA, VA = self.fb_mu, self.fb_V
        if nb.size: mB, VB = float(wb @ mb), float((wb * wb) @ (vb + self.beta2))
        else:       mB, VB = self.fb_mu, self.fb_V
        c2 = VA + VB; c = sqrt(c2); md = mA - mB + self.h * self.D["Home"][i]
        if first_visit: self.p_first[i] = ndtr(md / c)
        a_won = self.D["A_won"][i] == 1
        v, w = v_w((md if a_won else -md) / c)
        out = 0.0
        for nodes, wt, m, var, sign, k in ((na, wa, ma, va, 1.0 if a_won else -1.0, 0),
                                           (nb, wb, mb, vb, -1.0 if a_won else 1.0, 2)):
            if not nodes.size: continue
            m_new = m + sign * (wt * var / c) * v                       # TrueSkill update of the CAVITY
            v_new = var * (1.0 - (wt * wt * var / c2) * w)
            site_pi = 1.0 / v_new - 1.0 / var                           # new site = posterior / cavity
            site_eta = m_new / v_new - m / var
            self.lik_pi[nodes] += site_pi - s[k]; self.lik_eta[nodes] += site_eta - s[k + 1]
            s[k][:] = site_pi; s[k + 1][:] = site_eta
            out = max(out, float(np.max(np.abs(m_new - m))) if nodes.size else 0.0)
        return out

    def ep_step(self, st, n_inner, tol, first_visit=False, order=None):
        """EP inside one time step: revisit its matches until no mean AND no standard deviation
        in this step moves by more than tol (or n_inner passes are used)."""
        ms = self.G["matches_in"][st] if order is None else order
        nodes = self.G["nodes_in"][st]
        if nodes.size == 0:                          # only teams with no known players this step
            n_inner = 1
        for it in range(n_inner):
            m0, v0 = self.belief(nodes)
            for i in ms:
                self.match_step(i, first_visit=(first_visit and it == 0))
            if nodes.size == 0: break
            m1, v1 = self.belief(nodes)
            if max(np.max(np.abs(m1 - m0)), np.max(np.abs(np.sqrt(v1) - np.sqrt(v0)))) < tol: break

    # --- the full schedule
    def check_finite(self):
        """Every stored natural parameter must be a finite number and every belief must have positive variance."""
        arrs = [self.fwd_pi, self.fwd_eta, self.bwd_pi, self.bwd_eta, self.lik_pi, self.lik_eta]
        if not all(np.all(np.isfinite(a)) for a in arrs):
            raise FloatingPointError("a message contains NaN or inf")
        pi = self.fwd_pi + self.bwd_pi + self.lik_pi
        if np.any(pi <= 0):
            raise FloatingPointError("a belief has non-positive precision")

    def run(self, n_sweeps=30, n_inner=10, tol=1e-4, forward_only=False, verbose=False, order_seed=None):
        """Forward pass, then backward pass, repeated.  Stops when, between two consecutive sweeps, no mean
        and no standard deviation changes by more than tol.  Returns a report that says whether the run
        stopped because of that threshold ('converged') or because it ran out of sweeps."""
        steps = self.G["steps"]; all_nodes = np.arange(self.G["N"])
        rng = np.random.default_rng(order_seed) if order_seed is not None else None
        orders = {st: (list(rng.permutation(self.G["matches_in"][st])) if rng is not None else None) for st in steps}
        rep = dict(d_mean=[], d_sd=[], converged=False, sweeps=0, stopped_by="max_sweeps")
        prev = None
        for sweep in range(n_sweeps):
            for st in steps:                                            # forward pass
                self.forward_msg(self.G["nodes_in"][st])
                self.ep_step(st, n_inner, tol, first_visit=(sweep == 0), order=orders[st])
            if forward_only:
                self.check_finite(); rep.update(sweeps=1, stopped_by="forward_only"); return rep
            for st in reversed(steps):                                  # backward pass
                self.backward_msg(self.G["nodes_in"][st])
                self.ep_step(st, n_inner, tol, order=orders[st])
            self.check_finite()
            m, v = self.belief(all_nodes); sd = np.sqrt(v)
            rep["sweeps"] = sweep + 1
            if prev is not None:
                dm = float(np.max(np.abs(m - prev[0]))); ds = float(np.max(np.abs(sd - prev[1])))
                rep["d_mean"].append(dm); rep["d_sd"].append(ds)
                if verbose: print(f"sweep {sweep + 1:2d}: largest change  mean {dm:.2e}   sd {ds:.2e}")
                if max(dm, ds) < tol:
                    rep.update(converged=True, stopped_by="threshold"); break
            prev = (m, sd)
        if verbose:
            print("stopped:", "change below tol" if rep["converged"] else f"reached the limit of {n_sweeps} sweeps (NOT converged)")
        return rep
