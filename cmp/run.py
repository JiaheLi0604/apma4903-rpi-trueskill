import sys,pickle,numpy as np,pandas as pd
sys.path.insert(0,'work'); from ts_core import run_walk_forward, log_loss
D=pickle.load(open('work/D.pkl','rb')); P=pickle.load(open('work/P_team.pkl','rb'))
A,B,y,S,DI,H=D['A'],D['B'],D['A_won'].astype(bool),D['m_season'],D['m_di'],D['Home']
dates=D['date_list']; nT=D['n_teams']
print('Home values',np.unique(H,return_counts=True))
# ---------- walk-forward RPI (season to date, D1 vs D1, before the match day) ----------
rpi_A=np.full(len(A),np.nan); rpi_B=np.full(len(A),np.nan); nA=np.zeros(len(A)); nB=np.zeros(len(A))
def rpi_from(games):
    W=np.zeros(nT); N=np.zeros(nT); pw={}; pn={}
    for a,b,aw in games:
        W[a]+=aw; W[b]+=1-aw; N[a]+=1; N[b]+=1
        pw[(a,b)]=pw.get((a,b),0)+aw; pw[(b,a)]=pw.get((b,a),0)+1-aw
        pn[(a,b)]=pn.get((a,b),0)+1; pn[(b,a)]=pn.get((b,a),0)+1
    with np.errstate(invalid='ignore',divide='ignore'): wp=W/N
    os=np.zeros(nT); oc=np.zeros(nT)
    for a,b,aw in games:
        for t,o in ((a,b),(b,a)):
            n=N[o]-pn[(o,t)]
            if n>0: os[t]+=(W[o]-pw[(o,t)])/n; oc[t]+=1
    with np.errstate(invalid='ignore',divide='ignore'): owp=os/oc
    s2=np.zeros(nT); c2=np.zeros(nT)
    for a,b,aw in games:
        for t,o in ((a,b),(b,a)):
            if not np.isnan(owp[o]): s2[t]+=owp[o]; c2[t]+=1
    with np.errstate(invalid='ignore',divide='ignore'): oowp=s2/c2
    return 0.25*wp+0.5*owp+0.25*oowp, N, (wp,owp,oowp)
for season in (2024,2025):
    days=sorted(set(DI[S==season]))
    games=[]
    for di in days:
        idx=np.where(DI==di)[0]
        if games:
            r,N,_=rpi_from(games)
            rpi_A[idx]=r[A[idx]]; rpi_B[idx]=r[B[idx]]; nA[idx]=N[A[idx]]; nB[idx]=N[B[idx]]
        games+= [(A[i],B[i],int(y[i])) for i in idx]
    if season==2025:
        final_rpi,_,comps=rpi_from([g for g in games if dates[DI[0]]<=dates[DI[0]]])
pickle.dump(dict(rpi_A=rpi_A,rpi_B=rpi_B,nA=nA,nB=nB),open('cmp/rpi_wf.pkl','wb'))
# ---------- TrueSkill team-level: full and "fair" (current season only, no home) ----------
full=run_walk_forward(D,P,level='team')['p']
Pf=dict(P,home=0.0)
fair=np.full(len(A),np.nan)
for season in (2024,2025):
    D2=dict(D); D2['days']=[di for di in D['days'] if D['season_of_day'][di]==season]
    fair_s=run_walk_forward(D2,Pf,level='team')['p']; m=S==season; fair[m]=fair_s[m]
pickle.dump(dict(full=full,fair=fair),open('cmp/ts_wf.pkl','wb'))
print('done')
