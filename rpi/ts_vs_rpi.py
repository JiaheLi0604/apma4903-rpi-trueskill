import sys,pickle,numpy as np,pandas as pd
sys.path.insert(0,'work'); from ts_core import run_walk_forward
D=pickle.load(open('work/D.pkl','rb')); P=pickle.load(open('work/P_team.pkl','rb'))
cut=pd.Timestamp('2025-11-24')
D2=dict(D); D2['days']=[di for di in D['days'] if D['date_list'][di]<=cut]
o=run_walk_forward(D2,P,level='team')
names=list(D['team_names']); mu=o['mu']; sd=np.sqrt(o['var'])
# teams that played in 2025
played=set()
for di in D2['days']:
    if D['date_list'][di].year==2025:
        for i in D['m_by_day'].get(di,[]): played|={D['A'][i],D['B'][i]}
ts=pd.DataFrame({'team':[names[t] for t in played],'mu':[mu[t] for t in played],'sd':[sd[t] for t in played]}).sort_values('mu',ascending=False).reset_index(drop=True)
ts['ts_rank']=ts.index+1
off="""Nebraska,Texas,Arizona St.,Kentucky,Creighton,Pittsburgh,Louisville,SMU,Texas A&M,Stanford,Purdue,Wisconsin,Indiana,Kansas,Minnesota,TCU,Southern California,Miami (FL),UTEP,Iowa St.,BYU,Colorado,Rice,Tennessee,Xavier,UNI,Western Ky.,Baylor,San Diego,South Dakota St.,Penn St.,Kansas St.,Utah St.,Marquette,Utah,Florida,Tulsa,Michigan,Villanova,South Fla.,Georgia Tech,UCLA,Cincinnati,Arizona,Texas St.,Washington,Pepperdine,James Madison,North Carolina,Illinois,Dayton,Missouri,Arkansas St.,Auburn""".split(',')
rec="""28-0,22-2,24-3,23-2,25-5,24-4,24-4,23-5,23-3,25-4,23-5,22-4,22-6,20-10,21-8,19-9,22-6,24-5,25-4,22-5,20-8,22-7,21-9,20-6,26-4,24-5,27-5,16-9,23-4,22-3,17-11,16-8,21-7,17-10,14-13,15-11,25-6,21-8,20-8,17-12,16-11,16-12,15-11,14-12,17-12,12-16,19-8,22-9,20-7,13-14,23-7,17-11,22-8,17-11""".split(',')
ts=ts.set_index('team')
print('missing in TS:',[t for t in off if t not in ts.index])
rows=[(i+1,t,rec[i],int(ts.loc[t,'ts_rank']),ts.loc[t,'mu'],ts.loc[t,'sd']) for i,t in enumerate(off) if t in ts.index]
df=pd.DataFrame(rows,columns=['rpi','team','rec','ts','mu','sd']); df['gap']=df.ts-df.rpi
pd.set_option('display.width',200)
print(df.to_string(index=False,float_format='%.2f'))
from scipy.stats import spearmanr
print('Spearman RPI vs TS (top54):',round(spearmanr(df.rpi,df.ts).correlation,3))
print('TS top 25:'); print(ts.head(25)[['ts_rank','mu','sd']].to_string(float_format='%.2f'))
print('TS top 54 not in RPI top 54:',[(t,int(ts.loc[t,'ts_rank'])) for t in ts.index[:54] if t not in off])
