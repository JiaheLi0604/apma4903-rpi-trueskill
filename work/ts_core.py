"""Core engine for Nugget 2: player-level TrueSkill with partial-play weights, run walk-forward.
Written in notebook-ready style (heavily commented); the notebook builder pastes these functions in.
"""
import numpy as np
import pandas as pd
from math import sqrt
from scipy.special import log_ndtr, ndtr

LOG_SQRT_2PI = 0.9189385332046727   # log(sqrt(2*pi))


# ------------------------------------------------------------------ truncated-Gaussian helpers
def v_w(t):
    """Truncated-Gaussian correction terms for 'the performance difference d turned out > 0'.
    v(t) = phi(t)/Phi(t)        (how far the mean of d moves, in units of c)
    w(t) = v(t) * (v(t) + t)    (how much the variance of d shrinks, as a fraction of c^2)
    Computed in log-space so that very lopsided matches (t very negative) do not underflow."""
    v = float(np.exp(-0.5 * t * t - LOG_SQRT_2PI - log_ndtr(t)))
    w = v * (v + t)
    return v, min(max(w, 0.0), 1.0 - 1e-12)


def match_update(mu_d, c2, idx_w, w_w, idx_l, w_l, mu, var):
    """Update beliefs (in place) after a match, given the winner/loser rosters.
       mu_d : prior mean of (winner performance - loser performance)   (home advantage already included)
       c2   : prior variance of that difference
       idx_*: array positions of the players on each side;  w_* : their partial-play weights."""
    c = sqrt(c2)
    t = mu_d / c
    v, w = v_w(t)
    for idx, wt, sign in ((idx_w, w_w, +1.0), (idx_l, w_l, -1.0)):
        if idx.size:
            vi = var[idx]
            mu[idx] = mu[idx] + sign * (wt * vi / c) * v           # mean moves up (winner) / down (loser)
            var[idx] = vi * (1.0 - (wt * wt * vi / c2) * w)         # variance always shrinks
    return t


# ------------------------------------------------------------------ data preparation
def prepare_data(pm, matches):
    """Turn the cleaned box scores (pm) and the match table (matches) into integer-indexed arrays."""
    agg = pm.groupby(["Team", "Date", "Player"], as_index=False, dropna=False)["S"].sum()   # sets per player per day
    team_names = sorted(set(pm["Team"]) | set(matches["A"]) | set(matches["B"]))
    team_id = {t: i for i, t in enumerate(team_names)}
    agg["tid"] = agg["Team"].map(team_id).astype(int)
    agg["pid"], player_keys = pd.factorize(agg["tid"].astype(str) + "||" + agg["Player"].astype(str))
    date_list = sorted(set(agg["Date"]) | set(matches["Date"]))
    date_idx = {d: i for i, d in enumerate(date_list)}
    agg["di"] = agg["Date"].map(date_idx).astype(int)

    box = {}                                       # box[day][team] = (player ids, sets played)
    for (di, tid), g in agg.groupby(["di", "tid"]):
        box.setdefault(di, {})[tid] = (g["pid"].values.astype(int), g["S"].values.astype(float))

    season_of = pm.groupby("Date")["SeasonYear"].first()
    season_of_day = {date_idx[d]: int(s) for d, s in season_of.items() if d in date_idx}

    m_di = matches["Date"].map(date_idx).astype(int).values
    m_by_day = {}
    for i, di in enumerate(m_di):
        m_by_day.setdefault(di, []).append(i)

    return dict(
        n_players=len(player_keys), n_teams=len(team_names), team_names=team_names, team_id=team_id,
        player_keys=player_keys, date_list=date_list, box=box, season_of_day=season_of_day,
        days=sorted(set(box) | set(m_by_day)), m_by_day=m_by_day, n_matches=len(matches),
        A=matches["A"].map(team_id).astype(int).values, B=matches["B"].map(team_id).astype(int).values,
        A_won=matches["A_won"].values.astype(int), Home=matches["Home"].values.astype(float),
        m_season=matches["SeasonYear"].values.astype(int), m_di=m_di)


DEFAULT_PARAMS = dict(mu0=25.0, sigma0=25.0 / 3, beta=25.0 / 6, tau_season=25.0 / 300, home=0.0, decay=0.6, rho=1.0)

W_BAR = 6.0   # a volleyball team always has six players on the court


def on_court_weights(share):
    """Turn raw playing-time shares into partial-play weights that always add up to 6 (the six on-court spots).
    Why: a 'share of the team's sets' counts every player who touched a set, so deeper benches add up to more than 6.
    Without this step a team with a longer rotation would get a free head start (sum of weights x prior mean 25)."""
    return W_BAR * share / share.sum()


# ------------------------------------------------------------------ the walk-forward engine
def run_walk_forward(D, P, level="player", update_weights="expected", new_prior="global", end_season=None,
                     record=False, fallback=(6.0, 1.9)):
    """Walk through every day in order.  For each day:
         (0) register players seen for the first time,
         (1) predict that day's matches from beliefs built ONLY from earlier days,
         (2) update beliefs with that day's results,
         (3) add that day's box scores to the playing-time memory.
       level          : 'player' (team = weighted sum of its players)  or  'team' (each team is one 'player')
       update_weights : 'expected' (same past-playing-time weights as the prediction)
                        'actual'   (today's real participation, when the box score exists)
       fallback       : (typical sum of weights, typical sum of squared weights) for a team with no history yet"""
    mu0, sig0, beta = P["mu0"], P["sigma0"], P["beta"]
    tau, h, decay, rho = P["tau_season"], P["home"], P["decay"], P["rho"]
    tau_m2 = P.get("tau_match", 0.0) ** 2                                       # drift added per match played
    beta2 = beta * beta
    nU = D["n_players"] if level == "player" else D["n_teams"]
    mu = np.full(nU, float(mu0)); var = np.full(nU, float(sig0) ** 2); seen = np.zeros(nU, bool)
    sets = np.zeros(nU); team_sets = np.zeros(D["n_teams"])
    roster = [[] for _ in range(D["n_teams"])]; roster_np = [None] * D["n_teams"]
    fb_mu = fallback[0] * mu0; fb_V = fallback[1] * (sig0 ** 2 + beta2)      # an all-prior team

    n = D["n_matches"]
    p_out = np.full(n, np.nan); muA_out = np.full(n, np.nan); muB_out = np.full(n, np.nan); c_out = np.full(n, np.nan)
    hist = dict(di=[], pid=[], mu=[], var=[]) if record else None
    one = {}                                                                   # team-level: cached 1-element arrays

    def expected_lineup(tid):
        """(player positions, weights) = each known player's share of the team's past sets."""
        if level == "team":
            if tid not in one: one[tid] = (np.array([tid]), np.array([1.0]))
            return one[tid]
        r = roster_np[tid]
        if r is None:
            r = np.array(roster[tid], dtype=int); roster_np[tid] = r
        ts = team_sets[tid]
        if r.size == 0 or ts <= 0:
            return r[:0], np.zeros(0)
        w = np.minimum(sets[r] / ts, 1.0)
        keep = w > 0
        if not keep.any():
            return r[:0], np.zeros(0)
        return r[keep], on_court_weights(w[keep])

    def stats(idx, w):
        if idx.size == 0:
            return fb_mu, fb_V
        return float(w @ mu[idx]), float((w * w) @ (var[idx] + beta2))

    cur_season = None
    for di in D["days"]:
        season = D["season_of_day"].get(di, cur_season)
        if end_season is not None and season is not None and season > end_season:
            break
        if cur_season is not None and season != cur_season:                  # ---- new season: memory fades, uncertainty grows
            sets *= decay; team_sets *= decay
            mu[seen] = mu0 + rho * (mu[seen] - mu0)
            var[seen] += tau ** 2
        cur_season = season
        todays_box = D["box"].get(di, {})
        todays_matches = D["m_by_day"].get(di, [])

        # (0) first appearances
        if level == "player":
            for tid, (ids, s) in todays_box.items():
                new = ~seen[ids]
                if new.any():
                    nid = ids[new]; seen[nid] = True; var[nid] = sig0 ** 2
                    start = mu0                                              # prior mean of a brand-new player
                    if new_prior == "team":                                  # ...or: the current average of her teammates
                        li, lw = expected_lineup(tid)
                        if li.size: start = float(lw @ mu[li] / lw.sum())
                    mu[nid] = start
                    roster[tid].extend(nid.tolist()); roster_np[tid] = None
        else:
            for i in todays_matches:
                for tid in (D["A"][i], D["B"][i]):
                    if not seen[tid]: seen[tid] = True; mu[tid] = mu0; var[tid] = sig0 ** 2

        # (1) predict + (2) update, for every match played today
        for i in todays_matches:
            a, b = D["A"][i], D["B"][i]
            ia, wa = expected_lineup(a); ib, wb = expected_lineup(b)
            if tau_m2 > 0:                                                       # skill may drift a little between matches
                var[ia] += tau_m2; var[ib] += tau_m2
            mA, VA = stats(ia, wa); mB, VB = stats(ib, wb)
            c2 = VA + VB
            mu_d = mA - mB + h * D["Home"][i]                                   # mean of (perf A - perf B)
            p_out[i] = ndtr(mu_d / sqrt(c2)); muA_out[i] = mA; muB_out[i] = mB; c_out[i] = sqrt(c2)

            if update_weights == "actual" and level == "player":              # use who REALLY played, if we know
                for tid, which in ((a, "a"), (b, "b")):
                    if tid in todays_box:
                        ids, s = todays_box[tid]
                        if which == "a": ia, wa = ids, on_court_weights(s / s.max())
                        else:            ib, wb = ids, on_court_weights(s / s.max())
                mA, VA = stats(ia, wa); mB, VB = stats(ib, wb)
                c2 = VA + VB; mu_d = mA - mB + h * D["Home"][i]
            if D["A_won"][i] == 1:
                match_update(mu_d, c2, ia, wa, ib, wb, mu, var)
            else:
                match_update(-mu_d, c2, ib, wb, ia, wa, mu, var)
            if record:
                for idx in (ia, ib):
                    if idx.size:
                        hist["di"].append(np.full(idx.size, di)); hist["pid"].append(idx)
                        hist["mu"].append(mu[idx].copy()); hist["var"].append(var[idx].copy())

        # (3) memory update with today's box scores
        if level == "player":
            for tid, (ids, s) in todays_box.items():
                sets[ids] += s
                team_sets[tid] += s.max()

    out = dict(p=p_out, muA=muA_out, muB=muB_out, c=c_out, mu=mu, var=var, sets=sets, team_sets=team_sets,
               roster=roster, seen=seen)
    if record:
        out["hist"] = {k: np.concatenate(v) if v else np.array([]) for k, v in hist.items()}
    return out


# ------------------------------------------------------------------ metrics (same definitions as Nugget 3)
def log_loss(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p))))

def brier(p, y):
    return float(np.mean((p - y) ** 2))

def accuracy(p, y):
    return float(np.mean((p > 0.5) == (y == 1)))
