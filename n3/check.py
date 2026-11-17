import sys; sys.path.insert(0,'n3')
import numpy as np
from mp import *
beta = 25/6
cases = {
 "1v1 two newcomers":      ([25],[8.333**2],[1],      [25],[8.333**2],[1], 0, True),
 "1v1 veteran vs newcomer":([25],[2.0**2],[1],        [25],[8.333**2],[1], 0, True),
 "1v1 upset (weak wins)":  ([20],[3.0**2],[1],        [30],[3.0**2],[1],  0, True),
 "2v2 weighted, home":     ([26,22],[3**2,6**2],[1.2,0.8], [27,24],[4**2,2**2],[1.0,1.0], 2.0, False),
}
for name,(mA,vA,wA,mB,vB,wB,h,won) in cases.items():
    mp_ = message_passing(mA,vA,wA,mB,vB,wB,beta,h,won)
    cf  = closed_form(mA,vA,wA,mB,vB,wB,beta,h,won)
    bf  = brute_force_grid(mA,vA,wA,mB,vB,wB,beta,h,won)
    print(f"\n== {name}")
    for side,k in (("A",0),("B",1)):
        for i in range(len(mp_[k][0])):
            print(f"  {side}{i+1}: MP mu={mp_[k][0][i]:.4f} sd={np.sqrt(mp_[k][1][i]):.4f} | CF mu={cf[k][0][i]:.4f} sd={np.sqrt(cf[k][1][i]):.4f} | GRID mu={bf[k][0][i]:.4f} sd={np.sqrt(bf[k][1][i]):.4f}")
    mpcf = max(np.max(np.abs(np.r_[mp_[0][0]-cf[0][0], mp_[1][0]-cf[1][0]])), np.max(np.abs(np.sqrt(np.r_[mp_[0][1],mp_[1][1]])-np.sqrt(np.r_[cf[0][1],cf[1][1]]))))   # mean or sd, same metric as the grid check
    print(f"  max |MP-CF| = {mpcf:.2e}")
print()
for name,(mA,vA,wA,mB,vB,wB,h,won) in cases.items():
    mp_=message_passing(mA,vA,wA,mB,vB,wB,beta,h,won); bf=brute_force_grid(mA,vA,wA,mB,vB,wB,beta,h,won)
    e=max(np.abs(np.r_[mp_[0][0]-bf[0][0],mp_[1][0]-bf[1][0]]).max(), np.abs(np.sqrt(np.r_[mp_[0][1],mp_[1][1]])-np.sqrt(np.r_[bf[0][1],bf[1][1]])).max())
    print(f"{name:26s} max |MP - grid| = {e:.1e}")
