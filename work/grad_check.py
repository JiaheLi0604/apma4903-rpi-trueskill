"""Check of Algorithm 3 with a gradient method: L-BFGS-B (finite-difference gradients) on the same 2024 log-loss,
started from the TrueSkill defaults; then the finite-difference gradient at the coordinate-search choice.
Run from work/:  python grad_check.py   (about 1 minute)."""
import pickle, time, numpy as np
from ts_core import run_walk_forward, log_loss, DEFAULT_PARAMS
from scipy.optimize import minimize
D = pickle.load(open('D.pkl', 'rb')); y = D['A_won']; val = D['m_season'] == 2024; tst = D['m_season'] == 2025
P_grid = pickle.load(open('P_tuned.pkl', 'rb'))
names = ['sigma0', 'tau_season', 'home', 'decay', 'rho']; lo = [1, 0, 0, 0.05, 0.3]; hi = [15, 3, 5, 1, 1]
def P_of(x): P = dict(DEFAULT_PARAMS); P.update(dict(zip(names, x))); return P
n = [0]
def f(x):
    n[0] += 1; return log_loss(run_walk_forward(D, P_of(x), end_season=2024)['p'][val], y[val])
def test(x): o = run_walk_forward(D, P_of(x)); return log_loss(o['p'][tst], y[tst])
t0 = time.time()
x0 = np.array([DEFAULT_PARAMS[k] for k in names])
r = minimize(f, x0, method='L-BFGS-B', bounds=list(zip(lo, hi)), options=dict(eps=1e-3, maxiter=40))
print('L-BFGS-B from defaults:', dict(zip(names, np.round(r.x, 3))), ' 2024 log-loss %.5f  2025 log-loss %.5f' % (r.fun, test(r.x)),
      ' iterations', r.nit, 'evaluations', n[0], round(time.time() - t0), 's')
xg = np.array([P_grid[k] for k in names])
fg = f(xg); print('coordinate-search choice:', dict(zip(names, xg)), ' 2024 log-loss %.5f  2025 log-loss %.5f' % (fg, test(xg)))
g = []
for i in range(5):
    e = np.zeros(5); e[i] = 1e-2
    g.append((fg - f(xg - e)) / 1e-2 if xg[i] + 1e-2 > hi[i] else (f(xg + e) - f(xg - e)) / 2e-2)
print('finite-difference gradient of the 2024 log-loss at the choice:', dict(zip(names, np.round(g, 5))))
