"""Results with the player-level model (Nugget 3's algorithm, parameters work/P_tuned.pkl) in the 'full' column.
'Same info' stays the team-level model (only this season's wins and losses, no home court), as before.
Run from the repository root after cmp/run.py, cmp/tune_fair.py and cmp/rank.py:  python cmp/player_results.py"""
import sys, pickle, numpy as np, pandas as pd
sys.path.insert(0, 'work'); from ts_core import run_walk_forward, on_court_weights
from scipy.special import expit
from scipy.optimize import minimize_scalar
from scipy.stats import spearmanr
D = pickle.load(open('work/D.pkl', 'rb')); P = pickle.load(open('work/P_tuned.pkl', 'rb'))
R = pickle.load(open('cmp/rpi_wf.pkl', 'rb')); T = pickle.load(open('cmp/ts_wf.pkl', 'rb'))
y = D['A_won'].astype(float); S = D['m_season']; names = np.array(D['team_names'])
dates = pd.to_datetime(pd.Series(D['date_list']))[D['m_di']].values
k = np.minimum(R['nA'], R['nB']); dR = R['rpi_A'] - R['rpi_B']
def li(p): p = np.clip(p, 1e-12, 1 - 1e-12); return -(y * np.log(p) + (1 - y) * np.log(1 - p))
acc = lambda p: ((p > .5) == (y == 1)).astype(float)
fit = (S == 2024) & (k >= 5)
b = minimize_scalar(lambda b: li(expit(b * dR))[fit].mean(), bounds=(0, 200), method='bounded').x
player = run_walk_forward(D, P)['p']                      # player-level, full information
M = {'RPI': expit(b * dR), 'RPI+home': T['rpi_home'], 'TS same info (team)': T['fair_tuned'], 'TS full (player)': player,
     'TS full (team, old)': T['full']}
rng = np.random.default_rng(1)
def ci(d): return np.percentile([d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)], [2.5, 97.5])
base = (S == 2025) & (k >= 5) & ~np.isnan(dR)
out = {}
for name, m in [('2025 both>=5', base), ('before Oct 15', base & (dates < np.datetime64('2025-10-15'))),
                ('Oct 15 onward', base & (dates >= np.datetime64('2025-10-15'))),
                ('tournament', (S == 2025) & (dates >= np.datetime64('2025-12-04')) & ~np.isnan(dR))]:
    print(f'== {name}  n={m.sum()}')
    for nm, p in M.items(): print(f'   {nm:22s} acc={acc(p)[m].mean():.4f}  LL={li(p)[m].mean():.4f}')
    if name == '2025 both>=5':
        for a, c in [('TS same info (team)', 'RPI'), ('TS full (player)', 'RPI'), ('TS full (player)', 'RPI+home')]:
            d = li(M[a])[m] - li(M[c])[m]; e = acc(M[a])[m] - acc(M[c])[m]
            dis = m & ((M[a] > .5) != (M[c] > .5))
            print(f'   {a} - {c}: dLL {d.mean():+.4f} {np.round(ci(d), 4)}  dAcc {e.mean():+.4f} {np.round(ci(e), 4)}  '
                  f'disagree n={dis.sum()}, {a} right {acc(M[a])[dis].mean():.3f}')
# ---- rankings at the end of the regular season (games before Nov 30, 2025)
cut = pd.Timestamp('2025-11-30')
D2 = dict(D); D2['days'] = [di for di in D['days'] if D['date_list'][di] < cut]
o = run_walk_forward(D2, P)
def strength(tid):                                          # line 3 of the pseudocode: team mean = sum of w_i mu_i
    r = np.array(o['roster'][tid], dtype=int); ts = o['team_sets'][tid]
    if r.size == 0 or ts <= 0: return np.nan
    w = np.minimum(o['sets'][r] / ts, 1.0); keep = w > 0
    if not keep.any(): return np.nan
    return float(on_court_weights(w[keep]) @ o['mu'][r[keep]])
df = pickle.load(open('cmp/rank_df.pkl', 'rb')).copy()
tid = {n: i for i, n in enumerate(names)}
df['tsP'] = [strength(tid[t]) for t in df.team]
print('teams without box scores (no player-level rating):', int(df.tsP.isna().sum()), 'of', len(df))
df['tsP_rk'] = df.tsP.rank(ascending=False).astype('Int64')
ok = df.tsP.notna()
print('Spearman RPI vs TS full (player):', round(spearmanr(df.rpi[ok], df.tsP[ok]).correlation, 3),
      ' RPI vs TS same info:', round(spearmanr(df.rpi, df.tsS).correlation, 3))
show = ['UTEP', 'North Carolina', 'Missouri', 'Georgia', 'Mississippi St.', 'High Point', 'Eastern Ky.', 'Xavier']
print(df[df.team.isin(show)][['team', 'rpi_rk', 'tsS_rk', 'tsF_rk', 'tsP_rk']].to_string(index=False))
top = df[ok & ((df.rpi_rk <= 64) | (df.tsP_rk <= 64))].copy(); top['gap'] = top.tsP_rk.astype(float) - top.rpi_rk
print('RPI ranks much higher than TS full (player):'); print(top.sort_values('gap', ascending=False).head(6)[['team', 'rpi_rk', 'tsS_rk', 'tsP_rk']].to_string(index=False))
print('TS full (player) ranks much higher than RPI:'); print(top.sort_values('gap').head(6)[['team', 'rpi_rk', 'tsS_rk', 'tsP_rk']].to_string(index=False))
# ---- UTEP vs UNC
i = [j for j in range(len(y)) if S[j] == 2025 and {names[D['A'][j]], names[D['B'][j]]} == {'UTEP', 'North Carolina'}][0]
unc_is_A = names[D['A'][i]] == 'North Carolina'
for nm, p in M.items(): print(f'   P(UNC wins) {nm:22s} {p[i] if unc_is_A else 1 - p[i]:.3f}')
print('   note: walk-forward player-level uses the data Home flag; live.py uses h = 0 (neutral site)')
pickle.dump(dict(player=player, rank_df=df), open('cmp/player_results.pkl', 'wb'))
