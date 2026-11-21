"""Season engine (Algorithm 2, SeasonScore): player-level TrueSkill with playing-time weights, walked forward day by day.
Written with plain for loops so every step can be read line by line.  Lines marked "Algorithm 2, line k" match the slide."""
import math
import numpy as np
import pandas as pd
from scipy.special import log_ndtr, ndtr
LOG_SQRT_2PI = 0.9189385332046727          # log of sqrt(2*pi)
W_BAR = 6.0                                 # a volleyball team always has six players on the court

DEFAULT_PARAMS = {"mu0": 25.0, "sigma0": 25.0 / 3, "beta": 25.0 / 6, "tau_season": 25.0 / 300,
                  "home": 0.0, "decay": 0.6, "rho": 1.0}


def v_w(t):
    """Nugget 2's V(t) = phi(t) / Phi(t) and W(t) = V(t) * (V(t) + t), computed in log space."""
    log_phi = -0.5 * t * t - LOG_SQRT_2PI
    v = math.exp(log_phi - log_ndtr(t))
    w = v * (v + t)
    if w < 0.0:                      # rounding guard: W is always between 0 and 1
        w = 0.0
    if w > 1.0 - 1e-12:
        w = 1.0 - 1e-12
    return v, w


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



def on_court_weights(share):
    """Rescale playing-time shares so that the team's weights add up to 6 (Algorithm 2, line 7)."""
    total = 0.0
    for x in share:
        total = total + x
    weights = []
    for x in share:
        weights.append(W_BAR * x / total)
    return np.array(weights)


def match_update(mu_d, c2, idx_w, w_w, idx_l, w_l, mu, var):
    """Algorithm 1 in closed form (the same numbers as message passing, to 1e-15): update every player of the
    winner (idx_w, weights w_w) and the loser (idx_l, weights w_l) in place.
    mu_d = predicted mean of (winner - loser),  c2 = its variance."""
    c = math.sqrt(c2)
    t = mu_d / c
    v, w = v_w(t)
    for k in range(len(idx_w)):                 # winners move up
        i = idx_w[k]
        wt = w_w[k]
        mu[i] = mu[i] + (wt * var[i] / c) * v
        var[i] = var[i] * (1.0 - (wt * wt * var[i] / c2) * w)
    for k in range(len(idx_l)):                 # losers move down
        i = idx_l[k]
        wt = w_l[k]
        mu[i] = mu[i] - (wt * var[i] / c) * v
        var[i] = var[i] * (1.0 - (wt * wt * var[i] / c2) * w)
    return t


def run_walk_forward(D, P, level="player", new_prior="team", lineup="active", end_season=None,
                     fallback=(6.0, 1.9), return_lineups=False):
    """Algorithm 2 (SeasonScore).  Walk through every day in order.  For each day:
         (0) register players seen for the first time (Algorithm 2, line 5),
         (1) predict that day's matches from beliefs built ONLY from earlier days (lines 7-9),
         (2) update beliefs with that day's results (line 10, Algorithm 1),
         (3) add that day's sets played to the playing-time memory (lines 11-12).
       level      : 'player' (team = weighted sum of its players)  or  'team' (each team is one 'player')
       new_prior  : 'team'   a new player starts at her team's current weighted mean rating (final model)
                    'global' a new player starts at 25, as in TrueSkill
       lineup     : 'active' only players who have played this season; before the team's first box score of a
                             season, the players who played last season (final model)
                    'all'    everyone who has ever played for the team
       Returns a dictionary; out['p'][m] = predicted probability that team A wins match m."""
    mu0 = P["mu0"]
    sig0 = P["sigma0"]
    beta = P["beta"]
    tau = P["tau_season"]
    h = P["home"]
    decay = P["decay"]
    rho = P["rho"]
    tau_m2 = P.get("tau_match", 0.0) ** 2           # optional drift before every match (only used by one baseline)
    beta2 = beta * beta

    if level == "player":
        n_units = D["n_players"]
    else:
        n_units = D["n_teams"]

    # ---- the state the engine keeps (Algorithm 2, line 1)
    mu = np.full(n_units, float(mu0))               # rating mean of every player
    var = np.full(n_units, float(sig0) ** 2)        # rating variance of every player
    seen = np.zeros(n_units, bool)                  # has this player appeared yet?
    sets = np.zeros(n_units)                        # n_i: sets she has played (older seasons discounted)
    team_sets = np.zeros(D["n_teams"])              # N_T: sets her team has played (older seasons discounted)
    last_season = np.full(n_units, -1)              # last season in which she played (before today)
    team_played = np.zeros(D["n_teams"], bool)      # has the team played this season (before today)?
    roster = []                                     # roster[team] = every player who has ever played for the team
    for team in range(D["n_teams"]):
        roster.append([])

    n = D["n_matches"]
    p_out = np.full(n, np.nan)
    muA_out = np.full(n, np.nan)
    muB_out = np.full(n, np.nan)
    c_out = np.full(n, np.nan)
    lineups = None
    if return_lineups:
        lineups = [None] * n
    fb_mu = fallback[0] * mu0                       # a team with no history: six average players
    fb_V = fallback[1] * (sig0 ** 2 + beta2)

    cur_season = None

    def expected_lineup(team):
        """Algorithm 2, line 7: the players of the team's lineup and their weights, from earlier days only."""
        if level == "team":
            return [team], [1.0]
        if team_sets[team] <= 0:
            return [], []
        ids = []
        shares = []
        for i in roster[team]:
            share = sets[i] / team_sets[team]               # r_i = n_i / N_T
            if share > 1.0:
                share = 1.0                                 # r_i = min(n_i / N_T, 1)
            if share > 0:
                ids.append(i)
                shares.append(share)
        if lineup == "active":                              # keep only this season's players
            if team_played[team]:
                oldest_ok = cur_season                      # the team has played this season
            else:
                oldest_ok = cur_season - 1                  # before its first match: last season's players
            ids_active = []
            shares_active = []
            for k in range(len(ids)):
                if last_season[ids[k]] >= oldest_ok:
                    ids_active.append(ids[k])
                    shares_active.append(shares[k])
            if len(ids_active) > 0:                         # (if nobody qualifies, keep everyone)
                ids = ids_active
                shares = shares_active
        if len(ids) == 0:
            return [], []
        return ids, list(on_court_weights(shares))         # w_i = 6 r_i / sum of r_j

    def team_mean_var(ids, weights):
        """Algorithm 1, line 1: team mean = sum w_i mu_i,  team variance = sum w_i^2 (sigma_i^2 + beta^2)."""
        if len(ids) == 0:
            return fb_mu, fb_V
        m = 0.0
        v = 0.0
        for k in range(len(ids)):
            m = m + weights[k] * mu[ids[k]]
            v = v + weights[k] ** 2 * (var[ids[k]] + beta2)
        return m, v

    for di in D["days"]:                                    # Algorithm 2, line 2: for every day d
        season = D["season_of_day"].get(di, cur_season)
        if end_season is not None and season is not None and season > end_season:
            break

        # ---- Algorithm 2, lines 3-4: a new season starts
        if cur_season is not None and season != cur_season:
            for i in range(n_units):
                sets[i] = sets[i] * decay                   # n_i <- decay * n_i
                if seen[i]:
                    mu[i] = mu0 + rho * (mu[i] - mu0)       # pull toward 25 by rho
                    var[i] = var[i] + tau ** 2              # add tau^2
            for team in range(D["n_teams"]):
                team_sets[team] = team_sets[team] * decay   # N_T <- decay * N_T
                team_played[team] = False
        cur_season = season
        todays_box = D["box"].get(di, {})                   # box score: who played how many sets today
        todays_matches = D["m_by_day"].get(di, [])

        # ---- Algorithm 2, line 5: first appearances
        if level == "player":
            for team in todays_box:
                ids_today = todays_box[team][0]
                new_players = []
                for i in ids_today:
                    if not seen[i]:
                        new_players.append(i)
                if len(new_players) > 0:
                    start = mu0                             # 'global': start at 25
                    if new_prior == "team":                 # 'team': start at the team's current weighted mean
                        ids, weights = expected_lineup(team)
                        if len(ids) > 0:
                            total_w = 0.0
                            total_wmu = 0.0
                            for k in range(len(ids)):
                                total_w = total_w + weights[k]
                                total_wmu = total_wmu + weights[k] * mu[ids[k]]
                            start = total_wmu / total_w
                    for i in new_players:
                        seen[i] = True
                        mu[i] = start
                        var[i] = sig0 ** 2
                        roster[team].append(i)
        else:
            for m in todays_matches:
                for team in (D["A"][m], D["B"][m]):
                    if not seen[team]:
                        seen[team] = True
                        mu[team] = mu0
                        var[team] = sig0 ** 2

        # ---- Algorithm 2, lines 6-10: predict, then update, every match of the day
        for m in todays_matches:
            a = D["A"][m]
            b = D["B"][m]
            ids_a, w_a = expected_lineup(a)                 # line 7: weights from earlier days only
            ids_b, w_b = expected_lineup(b)
            if return_lineups:
                lineups[m] = ((np.array(ids_a, dtype=int), np.array(w_a)), (np.array(ids_b, dtype=int), np.array(w_b)))
            if tau_m2 > 0:
                for i in ids_a:
                    var[i] = var[i] + tau_m2
                for i in ids_b:
                    var[i] = var[i] + tau_m2
            mean_a, v_a = team_mean_var(ids_a, w_a)
            mean_b, v_b = team_mean_var(ids_b, w_b)
            c2 = v_a + v_b
            mu_d = mean_a - mean_b + h * D["Home"][m]       # line 8: predicted gap (A - B), with home advantage
            p_out[m] = ndtr(mu_d / math.sqrt(c2))           # line 9: P(A wins), scored BEFORE the update
            muA_out[m] = mean_a
            muB_out[m] = mean_b
            c_out[m] = math.sqrt(c2)
            if D["A_won"][m] == 1:                          # line 10: Algorithm 1 (closed form)
                match_update(mu_d, c2, ids_a, w_a, ids_b, w_b, mu, var)
            else:
                match_update(-mu_d, c2, ids_b, w_b, ids_a, w_a, mu, var)

        # ---- Algorithm 2, lines 11-12: add today's sets played, once per day
        if level == "player":
            for team in todays_box:
                ids_today, sets_today = todays_box[team]
                most = 0.0
                for k in range(len(ids_today)):
                    sets[ids_today[k]] = sets[ids_today[k]] + sets_today[k]
                    last_season[ids_today[k]] = cur_season
                    if sets_today[k] > most:
                        most = sets_today[k]
                team_sets[team] = team_sets[team] + most        # sets_T(d) = max over players of sets_i(d)
                team_played[team] = True

    return {"p": p_out, "muA": muA_out, "muB": muB_out, "c": c_out, "mu": mu, "var": var, "sets": sets,
            "team_sets": team_sets, "roster": roster, "seen": seen, "last_season": last_season,
            "team_played": team_played, "lineups": lineups}


# ------------------------------------------------------------------ scores
def log_loss(p, y):
    """Mean of -log(probability given to what happened).  Lower is better; a coin flip scores 0.693."""
    total = 0.0
    for k in range(len(p)):
        q = min(max(p[k], 1e-6), 1 - 1e-6)
        if y[k] == 1:
            total = total - math.log(q)
        else:
            total = total - math.log(1 - q)
    return total / len(p)


def accuracy(p, y):
    """Share of matches where the favourite (probability > 0.5) won."""
    right = 0
    for k in range(len(p)):
        if (p[k] > 0.5) == (y[k] == 1):
            right = right + 1
    return right / len(p)


def brier(p, y):
    total = 0.0
    for k in range(len(p)):
        total = total + (p[k] - y[k]) ** 2
    return total / len(p)
