"""Maximum-likelihood learning of the model's parameters, run until convergence (Algorithm 3, CoordinateSearch).
Objective: maximize the log-likelihood of the 2024 results, sum over matches of log P(result | earlier results, theta).
Each P(result | ...) comes from one pass of message passing (Phi(t) at the truncation factor).
Method: coordinate ascent on fixed grids. One parameter at a time, try every grid value (one SeasonScore run each),
and move only on a STRICT improvement; repeat full rounds until a round changes nothing. The current value is always
one of the tries, so the computed likelihood never goes down, and the grids are finite, so the loop stops.
beta is fixed: it only sets the rating scale.
Final model: active lineup + new players start at their team's average (ts_core defaults)."""
import pickle, sys, time
import numpy as np
from ts_core import *
D = pickle.load(open("D.pkl", "rb"))
y = D["A_won"]
val = D["m_season"] == 2024          # True for every 2024 match (validation season)
tst = D["m_season"] == 2025          # True for every 2025 match (test season)

grids = {"sigma0": [2, 3, 4, 5, 6, 8.33, 10, 12],
         "tau_season": [0, 0.25, 0.5, 1, 1.5, 2, 3],
         "home": [0, 0.5, 1, 1.5, 2, 3, 4],
         "decay": [0.2, 0.4, 0.6, 0.8, 1.0],
         "rho": [0.6, 0.8, 0.9, 1.0]}

runs = []                            # every SeasonScore execution (one full walk through 2022-2024)


def neg_ll(P):
    """SeasonScore: mean negative log-likelihood (log-loss) of the 2024 predictions under settings P."""
    setting = []
    for name in grids:
        setting.append(P[name])
    runs.append(tuple(setting))
    out = run_walk_forward(D, P, end_season=2024)
    return log_loss(out["p"][val], y[val])


def describe(P):
    text = ""
    for name in grids:
        text = text + f"{name}={P[name]:g}  "
    return text


P = dict(DEFAULT_PARAMS)
t0 = time.time()
cur = neg_ll(P)
print(f"start (TrueSkill defaults): log-loss {cur:.5f}")
it = 0
while True:                          # Algorithm 3: repeat full rounds until nothing changes
    it = it + 1
    changed = []
    for name in grids:               # one parameter at a time
        best_ll = None
        best_v = None
        for v in grids[name]:        # try every grid value
            Q = dict(P)
            Q[name] = v
            ll = neg_ll(Q)
            if best_ll is None or ll < best_ll:
                best_ll = ll
                best_v = v
        if best_ll < cur:            # strict improvement only (Algorithm 3, line 6)
            if best_v != P[name]:
                changed.append(f"{name} {P[name]:g}->{best_v:g}")
            P[name] = best_v
            cur = best_ll
    if len(changed) > 0:
        print(f"round {it}: log-loss {cur:.5f}  {describe(P)}  changed: {changed}")
    else:
        print(f"round {it}: log-loss {cur:.5f}  {describe(P)}  (no change: converged)")
        break

distinct = []                        # count the distinct settings tried
for r in runs:
    if r not in distinct:
        distinct.append(r)
print(f"converged after {it} rounds: {len(runs)} SeasonScore runs ({len(distinct)} distinct settings), {time.time()-t0:.0f} s")

out = run_walk_forward(D, P)
for label, mask in (("validation 2024", val), ("test 2025", tst)):
    p = out["p"][mask]
    print(f"{label}: log-loss {log_loss(p, y[mask]):.4f}  accuracy {accuracy(p, y[mask]):.1%}  (all {mask.sum()} matches)")
try:
    old = pickle.load(open("P_tuned.pkl", "rb"))
    same = True
    for name in grids:
        if abs(old[name] - P[name]) > 1e-12:
            same = False
    print("same as P_tuned.pkl:", same)
except FileNotFoundError:
    pass
if "--save" in sys.argv:
    pickle.dump(P, open("P_tuned.pkl", "wb"))
    print("saved P_tuned.pkl")
