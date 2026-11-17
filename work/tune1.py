import pickle, time, numpy as np
from ts_core import *
D = pickle.load(open("D.pkl", "rb")); y = D["A_won"]; val = D["m_season"] == 2024; tst = D["m_season"] == 2025

# scale-invariance check: multiply sigma0, beta, tau, home by k -> identical probabilities
P = dict(DEFAULT_PARAMS); P["home"] = 1.0




def val_ll(P, **kw):
    out = run_walk_forward(D, P, end_season=2024, **kw)
    return log_loss(out["p"][val], y[val])

grids = dict(sigma0=[2, 3, 4, 5, 6, 8.33, 10, 12],
             tau_season=[0, 0.25, 0.5, 1, 1.5, 2, 3],
             home=[0, 0.5, 1, 1.5, 2, 3, 4],
             decay=[0.2, 0.4, 0.6, 0.8, 1.0],
             rho=[0.6, 0.8, 0.9, 1.0])
P = dict(DEFAULT_PARAMS); t0 = time.time()
best = val_ll(P); print("start", round(best, 4))
for sweep in range(2):
    for name, vals in grids.items():
        res = []
        for v in vals:
            Q = dict(P); Q[name] = v; res.append((val_ll(Q), v))
        ll, v = min(res); P[name] = v
        print(f"sweep{sweep} {name:10s}", " ".join(f"{x:g}:{l:.4f}" for l, x in res), f"-> {v}")
print("time", round(time.time()-t0, 1), "s;  tuned:", {k: round(v, 3) for k, v in P.items()})
out = run_walk_forward(D, P)
for nm, m in (("val 2024", val), ("test 2025", tst)):
    p = out["p"][m]; print(nm, "ll %.4f brier %.4f acc %.4f" % (log_loss(p, y[m]), brier(p, y[m]), accuracy(p, y[m])))
pickle.dump(P, open("P_tuned.pkl", "wb"))
