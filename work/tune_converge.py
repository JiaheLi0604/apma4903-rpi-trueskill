"""Maximum-likelihood learning of the model's parameters, run until convergence (Algorithm 3, CoordinateSearch).
Objective: maximize the log-likelihood of the 2024 results, sum over matches of log P(result | earlier results, theta).
Each P(result | ...) comes from one pass of message passing (Phi(t) at the truncation factor).
Method: coordinate ascent on fixed grids. One parameter at a time, try every grid value (one SeasonScore run each),
and move only on a STRICT improvement; repeat full rounds until a round changes nothing. The current value is always
one of the tries, so the computed likelihood never goes down, and the grids are finite, so the loop stops.
beta is fixed: it only sets the rating scale.
Final model: active lineup + new players start at their team's average (ts_core defaults)."""
import pickle, time, numpy as np
from ts_core import *
D = pickle.load(open("D.pkl", "rb")); y = D["A_won"]; val = D["m_season"] == 2024; tst = D["m_season"] == 2025

runs = []                            # every SeasonScore execution (one full walk through 2022-2024)
def neg_ll(P):                       # mean negative log-likelihood (log-loss) of the 2024 predictions
    runs.append(tuple(P[k] for k in grids))
    out = run_walk_forward(D, P, end_season=2024)
    return log_loss(out["p"][val], y[val])

grids = dict(sigma0=[2, 3, 4, 5, 6, 8.33, 10, 12],
             tau_season=[0, 0.25, 0.5, 1, 1.5, 2, 3],
             home=[0, 0.5, 1, 1.5, 2, 3, 4],
             decay=[0.2, 0.4, 0.6, 0.8, 1.0],
             rho=[0.6, 0.8, 0.9, 1.0])
P = dict(DEFAULT_PARAMS); t0 = time.time()
cur = neg_ll(P); print(f"start (TrueSkill defaults): log-loss {cur:.5f}")
it = 0
while True:                          # loop to convergence
    it += 1; changed = []
    for name, vals in grids.items():
        res = []
        for v in vals:
            Q = dict(P); Q[name] = v; res.append((neg_ll(Q), v))
        ll, best = min(res)
        if ll < cur:                 # strict improvement only (Algorithm 3, line 6)
            if best != P[name]: changed.append(f"{name} {P[name]:g}->{best:g}")
            P[name] = best; cur = ll
    print(f"round {it}: log-loss {cur:.5f}  " + "  ".join(f"{k}={P[k]:g}" for k in grids) +
          ("   changed: " + ", ".join(changed) if changed else "   (no change: converged)"))
    if not changed: break
print(f"converged after {it} rounds: {len(runs)} SeasonScore runs ({len(set(runs))} distinct settings), {time.time()-t0:.0f} s")
out = run_walk_forward(D, P)
for nm, m in (("validation 2024", val), ("test 2025", tst)):
    p = out["p"][m]; print(f"{nm}: log-loss {log_loss(p, y[m]):.4f}  accuracy {accuracy(p, y[m]):.1%}  (all {m.sum()} matches)")
try:
    old = pickle.load(open("P_tuned.pkl", "rb"))
    print("same as P_tuned.pkl:", all(abs(old[k] - P[k]) < 1e-12 for k in grids), {k: old[k] for k in grids})
except FileNotFoundError:
    pass
import sys
if "--save" in sys.argv:
    pickle.dump(P, open("P_tuned.pkl", "wb")); print("saved P_tuned.pkl")
