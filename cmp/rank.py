import sys,pickle,numpy as np,pandas as pd
sys.path.insert(0,'work'); from ts_core import run_walk_forward
from scipy.special import expit
from scipy.stats import spearmanr
D=pickle.load(open('work/D.pkl','rb')); P=pickle.load(open('work/P_team.pkl','rb')); T=pickle.load(open('cmp/ts_wf.pkl','rb')); R=pickle.load(open('cmp/rpi_wf.pkl','rb'))
exec(open('cmp/run.py').read().split('# ---------- walk-forward')[0].split('print(')[0])
A,B,y,S,DI=D['A'],D['B'],D['A_won'],D['m_season'],D['m_di']; dates=D['date_list']; names=D['team_names']; nT=D['n_teams']
src=open('cmp/run.py').read(); s=src.index('def rpi_from'); e=src.index('for season in (2024,2025):'); exec(src[s:e])
cut=pd.Timestamp('2025-11-30')
idx=[i for i in range(len(A)) if S[i]==2025 and dates[DI[i]]<cut]
r,N,(wp,owp,oowp)=rpi_from([(A[i],B[i],int(y[i])) for i in idx])
D2=dict(D); D2['days']=[di for di in D['days'] if dates[di]<cut]
o=run_walk_forward(D2,P,level='team'); muF=o['mu']
D3=dict(D); D3['days']=[di for di in D['days'] if dates[di]<cut and D['season_of_day'][di]==2025]
o3=run_walk_forward(D3,T['P_fair'],level='team'); muS=o3['mu']
played=np.zeros(nT,bool); played[A[idx]]=True; played[B[idx]]=True
df=pd.DataFrame({'team':np.array(names)[played],'rpi':r[played],'wp':wp[played],'owp':owp[played],'oowp':oowp[played],'tsF':muF[played],'tsS':muS[played],'n':N[played]})
for c in ['rpi','tsF','tsS']: df[c+'_rk']=df[c].rank(ascending=False).astype(int)
df=df.sort_values('rpi_rk')
print('Spearman all teams RPI vs TS full',round(spearmanr(df.rpi,df.tsF).correlation,3),' vs TS same-info',round(spearmanr(df.rpi,df.tsS).correlation,3))
top=df[(df.rpi_rk<=64)|(df.tsF_rk<=64)]
print('top64 overlap RPI vs TSfull:',len(set(df[df.rpi_rk<=64].team)&set(df[df.tsF_rk<=64].team)))
for t in ['UTEP','North Carolina','Texas A&M','Nebraska','Princeton','Columbia','Kentucky','Pittsburgh']:
    print(df[df.team==t].to_string(header=False,index=False,float_format='%.3f'))
top['gap']=top.tsF_rk-top.rpi_rk
pd.set_option('display.width',250)
print('RPI much higher than TS (RPI likes):'); print(top.sort_values('gap',ascending=False).head(10).to_string(index=False,float_format='%.3f'))
print('TS much higher than RPI:'); print(top.sort_values('gap').head(10).to_string(index=False,float_format='%.3f'))
pickle.dump(df,open('cmp/rank_df.pkl','wb'))
# UTEP-UNC match
i=[j for j in range(len(A)) if S[j]==2025 and {names[A[j]],names[B[j]]}=={'UTEP','North Carolina'}]
for j in i: print('match',j,dates[DI[j]],names[A[j]],'vs',names[B[j]],'A_won',y[j],'pRPI(A)',expit(16.73*(R['rpi_A'][j]-R['rpi_B'][j])).round(3),'TSfull',T['full'][j].round(3),'TSsame',T['fair_tuned'][j].round(3),'Home',D['Home'][j])
