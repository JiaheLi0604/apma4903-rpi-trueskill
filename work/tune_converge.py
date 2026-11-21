"""Maximum-likelihood learning of the settings theta (Algorithm 3, CoordinateSearch), line by line as on the slide.
Objective (Nugget 1): L(theta) = sum over the 2024 matches of log P(result | earlier results, theta),
computed by Algorithm 2 (SeasonScore = run_walk_forward(...)["L"]); each term is log Phi(t_m) from message passing.
Method: coordinate search on fixed grids G_k. One setting at a time, compute L for every grid value and move only on a
STRICT increase; repeat rounds while a round raises L. The current value is always one of the tries, so the computed
L never goes down, and the grids are finite, so the loop stops.  beta is fixed: it only sets the rating scale.
Final model: active lineup + new players start at their team's average (ts_core defaults)."""
import pickle, sys, time
import numpy as np
from ts_core import *
D = pickle.load(open("D.pkl", "rb"))
y = D["A_won"]
val = D["m_season"] == 2024          # True for every 2024 match (the scored season)
tst = D["m_season"] == 2025          # True for every 2025 match (test season, never used here)
n_val = int(val.sum())               # 4698

grids = {"sigma0": [2, 3, 4, 5, 6, 8.33, 10, 12],              # G_sigma0
         "tau_season": [0, 0.25, 0.5, 1, 1.5, 2, 3],           # G_tau
         "home": [0, 0.5, 1, 1.5, 2, 3, 4],                    # G_h
         "decay": [0.2, 0.4, 0.6, 0.8, 1.0],                   # G_decay
         "rho": [0.6, 0.8, 0.9, 1.0]}                          # G_rho

runs = []                            # every SeasonScore execution (one full walk through 2022-2024)


def SeasonScore(theta):
    """Algorithm 2: L(theta), the log-likelihood of the 2024 results."""
    setting = []
    for name in grids:
        setting.append(theta[name])
    runs.append(tuple(setting))
    return run_walk_forward(D, theta, end_season=2024)["L"]


def describe(theta):
    text = ""
    for name in grids:
        text = text + f"{name}={theta[name]:g}  "
    return text


t0 = time.time()
# Algorithm 3, line 1: theta <- theta(0), L <- SeasonScore(theta), L(0) <- L, L(-1) <- -infinity, r <- 0
theta = dict(DEFAULT_PARAMS)
L = SeasonScore(theta)
L_round = {0: L, -1: -float("inf")}
r = 0
print(f"start (TrueSkill defaults): L = {L:.2f}   (log-loss -L/{n_val} = {-L / n_val:.5f})")
while L_round[r] - L_round[r - 1] > 0:              # line 2: while L(r) - L(r-1) > epsilon = 0
    r = r + 1                                       # line 3
    for k in grids:                                 # line 4: for k in (sigma0, tau, h, decay, rho)
        L_v = []                                    # line 5: L_v <- SeasonScore(theta with k set to v), every v in G_k
        for v in grids[k]:
            trial = dict(theta)
            trial[k] = v
            L_v.append(SeasonScore(trial))
        best = 0                                    # line 6: v* <- argmax_v L_v  (first in G_k if tied)
        for j in range(len(L_v)):
            if L_v[j] > L_v[best]:
                best = j
        if L_v[best] > L:                           #         if L_v* > L: theta <- theta with k set to v*, L <- L_v*
            theta[k] = grids[k][best]               #         (strict increase only)
            L = L_v[best]
    L_round[r] = L                                  # line 7: L(r) <- L
    print(f"round {r}: L = {L:.2f}   (log-loss {-L / n_val:.5f})  {describe(theta)}")

distinct = []                                       # count the distinct settings tried
for s_ in runs:
    if s_ not in distinct:
        distinct.append(s_)
print(f"converged after {r} rounds: {len(runs)} SeasonScore runs ({len(distinct)} distinct settings), {time.time()-t0:.0f} s")

out = run_walk_forward(D, theta)
for label, mask in (("validation 2024", val), ("test 2025", tst)):
    p = out["p"][mask]
    print(f"{label}: log-loss {log_loss(p, y[mask]):.4f}  accuracy {accuracy(p, y[mask]):.1%}  (all {mask.sum()} matches)")
try:
    old = pickle.load(open("P_tuned.pkl", "rb"))
    same = True
    for name in grids:
        if abs(old[name] - theta[name]) > 1e-12:
            same = False
    print("same as P_tuned.pkl:", same)
except FileNotFoundError:
    pass
if "--save" in sys.argv:
    pickle.dump(theta, open("P_tuned.pkl", "wb"))
    print("saved P_tuned.pkl")
