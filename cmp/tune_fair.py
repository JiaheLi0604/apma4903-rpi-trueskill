import sys,pickle,numpy as np
sys.path.insert(0,'work'); from ts_core import run_walk_forward
from scipy.special import expit
from scipy.optimize import minimize
D=pickle.load(open('work/D.pkl','rb')); P=pickle.load(open('work/P_team.pkl','rb')); R=pickle.load(open('cmp/rpi_wf.pkl','rb'))
y=D['A_won'].astype(float); S=D['m_season']; k=np.minimum(R['nA'],R['nB']); H=D['Home']
def ll(p,yy): p=np.clip(p,1e-12,1-1e-12); return -np.mean(yy*np.log(p)+(1-yy)*np.log(1-p))
def fair(Pf,season):
    D2=dict(D); D2['days']=[di for di in D['days'] if D['season_of_day'][di]==season]
    return run_walk_forward(D2,Pf,level='team')['p']
m24=(S==2024)&(k>=5)
res=[]
for s0 in [2,3,4,5,6,8.33,12]:
  for tm in [0,0.5,1]:
    Pf=dict(P,home=0.0,sigma0=s0,tau_match=tm); p=fair(Pf,2024); res.append((ll(p[m24],y[m24]),s0,tm))
res.sort(); print(res[:5])
best=res[0]; Pf=dict(P,home=0.0,sigma0=best[1],tau_match=best[2])
p25=fair(Pf,2025); p24=fair(Pf,2024)
# RPI + home logistic fitted on 2024
dR=R['rpi_A']-R['rpi_B']
f=lambda w: ll(expit(w[0]*dR[m24]+w[1]*H[m24]),y[m24])
w=minimize(f,[16,0.2]).x; print('RPI+home weights',w)
pRH=expit(w[0]*dR+w[1]*H)
T=pickle.load(open('cmp/ts_wf.pkl','rb')); T['fair_tuned']=np.where(S==2025,p25,p24); T['rpi_home']=pRH; T['P_fair']=Pf
pickle.dump(T,open('cmp/ts_wf.pkl','wb'))
