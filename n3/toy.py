"""The 2 vs. 2 toy match from the slides. Change any number below and rerun:  python n3/toy.py"""
import sys, numpy as np
sys.path.insert(0, 'n3')
from mp import message_passing, closed_form, brute_force_grid

beta = 25/6          # match-day noise
h = 0.0              # home advantage for team A (0 = neutral site)
A_won = True

#         means      variances (sigma^2)   weights
muA, varA, wA = [28, 22], [3**2, 6**2], [1, 1]
muB, varB, wB = [26, 24], [4**2, 2**2], [1, 1]

log = []
(mA, vA), (mB, vB) = message_passing(muA, varA, wA, muB, varB, wB, beta, h, A_won, log)
print("Every message (mean, sd):")
for name, m, v in log:
    print(f"  {name:35s} {m:8.2f} {np.sqrt(v):6.2f}")
print("\nPlayer   before          after")
for lab, mu0, v0, m1, v1 in [("A1", muA[0], varA[0], mA[0], vA[0]), ("A2", muA[1], varA[1], mA[1], vA[1]),
                              ("B1", muB[0], varB[0], mB[0], vB[0]), ("B2", muB[1], varB[1], mB[1], vB[1])]:
    print(f"  {lab}   {mu0:5.2f} ± {np.sqrt(v0):4.2f}   {m1:5.2f} ± {np.sqrt(v1):4.2f}")
cf = closed_form(muA, varA, wA, muB, varB, wB, beta, h, A_won)
bf = brute_force_grid(muA, varA, wA, muB, varB, wB, beta, h, A_won)
d = lambda a, b: max(np.abs(np.r_[a[0][0]-b[0][0], a[1][0]-b[1][0]]).max(),
                     np.abs(np.sqrt(np.r_[a[0][1], a[1][1]]) - np.sqrt(np.r_[b[0][1], b[1][1]])).max())
mp = ((mA, vA), (mB, vB))
print(f"\nmax difference vs. Nugget 2 formula: {d(mp, cf):.1e}")
print(f"max difference vs. grid integration: {d(mp, bf):.1e}")
