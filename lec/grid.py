import sys, pickle, itertools, time, numpy as np
sys.path.insert(0,'.')
from ts_core import run_walk_forward, log_loss
from multiprocessing import Pool
D=pickle.load(open('D.pkl','rb')); P0=pickle.load(open('P_tuned.pkl','rb'))
mv=D['m_season']==2024; mt=D['m_season']==2025; y=D['A_won']
S=[3,3.5,4,4.5,5,6]; H=[1,1.5,2,2.5,3]; DE=[0.6,0.8,1.0]; T=[0,0.25,1]; R=[0.9,1.0]
combos=list(itertools.product(S,H,DE,T,R))
def f(c):
    s,h,de,t,r=c; P=dict(P0,sigma0=s,home=h,decay=de,tau_season=t,rho=r)
    p=run_walk_forward(D,P)["p"]
    return c, log_loss(p[mv],y[mv]), log_loss(p[mt],y[mt])
if __name__=="__main__":
    t0=time.time()
    with Pool() as pool: res=pool.map(f,combos)
    pickle.dump(res,open('../lec/grid_res.pkl','wb'))
    res.sort(key=lambda r:r[1])
    print(len(res),"runs",round(time.time()-t0),"s")
    for r in res[:10]: print(r)
    cs=[r for r in res if r[0]==(4,2,1.0,0.25,1.0)]; print("coord-search pick:",cs, "rank", res.index(cs[0])+1)
