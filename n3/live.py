import sys,pickle,numpy as np,pandas as pd
sys.path[:0]=['n3','work']
from mp import *; from ts_core import run_walk_forward; from ttt import record_lineups
D=pickle.load(open('work/D.pkl','rb')); P=pickle.load(open('work/P_tuned.pkl','rb'))
names=D['team_names']; tid=D['team_id']
day=pd.Timestamp('2025-12-04')
cand=[i for i in range(D['n_matches']) if D['date_list'][D['m_di'][i]]==day and {names[D['A'][i]],names[D['B'][i]]}=={'UTEP','North Carolina'}]
i=cand[0]; di=D['m_di'][i]
print('match',i,'A=',names[D['A'][i]],'B=',names[D['B'][i]],'A_won',D['A_won'][i],'Home',D['Home'][i])
D2=dict(D); D2['days']=[d for d in D['days'] if d<di]
o=run_walk_forward(D2,P)
lu=record_lineups(D,P)[i]
(ia,wa),(ib,wb)=lu
mu,var=o['mu'],o['var']
keys=D['player_keys']
def show(ids,w,label):
    print(label)
    for p,ww in sorted(zip(ids,w),key=lambda x:-x[1]): print(f"   {keys[p].split('||')[1]:22s} w={ww:.2f} mu={mu[p]:6.2f} sd={np.sqrt(var[p]):.2f}")
show(ia,wa,names[D['A'][i]]); show(ib,wb,names[D['B'][i]])
beta=P["beta"]; h=0.0   # neutral site (played in Madison); the data wrongly marks it as a UTEP home game
won=bool(D['A_won'][i])
mA=wa@mu[ia]; mB=wb@mu[ib]; c=np.sqrt((wa**2)@(var[ia]+beta**2)+(wb**2)@(var[ib]+beta**2))
from scipy.special import ndtr
print(f"team means {mA:.1f} vs {mB:.1f}, c={c:.2f}, P(A wins)={ndtr((mA-mB+h)/c):.3f}, h={h}")
log=[]
mpA,mpB=message_passing(mu[ia],var[ia],wa,mu[ib],var[ib],wb,beta,h,won,log)
cfA,cfB=closed_form(mu[ia],var[ia],wa,mu[ib],var[ib],wb,beta,h,won)
mcA,mcB=monte_carlo(mu[ia],var[ia],wa,mu[ib],var[ib],wb,beta,h,won,N=3_000_000)
d=lambda a,b: max(np.abs(a[0]-b[0]).max(),np.abs(np.sqrt(a[1])-np.sqrt(b[1])).max())
print('max|MP-CF|',d(mpA,cfA),d(mpB,cfB)); print('max|MP-MC| (mu or sd)',d(mpA,mcA),d(mpB,mcB))
for nm,m,v in log[:3]+[l for l in log if l[0].startswith(('sum->t','diff->d','d marginal','trunc'))]: print(f"  {nm:35s} mean={m:8.3f} sd={np.sqrt(v):.3f}")
pickle.dump(dict(i=i,ia=ia,wa=wa,ib=ib,wb=wb,mu=mu[np.r_[ia,ib]],var=var[np.r_[ia,ib]]),open('n3/live_state.pkl','wb'))
