"""Check of Algorithm 3: a finer grid of 540 combinations around the coordinate-search choice (final model).
Run from work/:  python ../lec/grid.py   (about 3 minutes on 2 cores)."""
import sys, pickle, itertools, time, numpy as np
sys.path.insert(0, '.')
from ts_core import run_walk_forward, log_loss
from multiprocessing import Pool
D = pickle.load(open('D.pkl', 'rb')); P0 = pickle.load(open('P_tuned.pkl', 'rb'))
mv = D['m_season'] == 2024; mt = D['m_season'] == 2025; y = D['A_won']
S = [3, 3.5, 4, 4.5, 5, 6]; H = [2, 2.5, 3, 3.5, 4]; DE = [0.6, 0.8, 1.0]; T = [0.5, 1, 1.5]; R = [0.9, 1.0]
combos = list(itertools.product(S, H, DE, T, R))           # 6 x 5 x 3 x 3 x 2 = 540
def f(c):
    s, h, de, t, r = c; P = dict(P0, sigma0=s, home=h, decay=de, tau_season=t, rho=r)
    p = run_walk_forward(D, P)["p"]
    return c, log_loss(p[mv], y[mv]), log_loss(p[mt], y[mt])
if __name__ == "__main__":
    t0 = time.time()
    with Pool() as pool: res = pool.map(f, combos)
    pickle.dump(res, open('../lec/grid_res.pkl', 'wb'))
    res.sort(key=lambda r: r[1])
    print(len(res), "combinations (sigma0, home, decay, tau, rho)", round(time.time() - t0), "s")
    for r in res[:5]: print("  ", r[0], "2024 %.5f  2025 %.5f" % (r[1], r[2]))
    pick = (P0['sigma0'], P0['home'], P0['decay'], P0['tau_season'], P0['rho'])
    cs = [r for r in res if r[0] == pick][0]
    print("coordinate-search pick", pick, "2024 %.5f  2025 %.5f" % (cs[1], cs[2]), "rank", res.index(cs) + 1, "of", len(res),
          " 2024 gap to #1: %.5f" % (cs[1] - res[0][1]))
