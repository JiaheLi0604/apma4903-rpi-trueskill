import pickle,numpy as np,pandas as pd
from scipy.special import expit
from scipy.optimize import minimize_scalar
D=pickle.load(open('work/D.pkl','rb')); R=pickle.load(open('cmp/rpi_wf.pkl','rb')); T=pickle.load(open('cmp/ts_wf.pkl','rb'))
y=D['A_won'].astype(float); S=D['m_season']; dates=np.array(D['date_list'])[D['m_di']]
dR=R['rpi_A']-R['rpi_B']; k=np.minimum(R['nA'],R['nB'])
def li(p): p=np.clip(p,1e-12,1-1e-12); return -(y*np.log(p)+(1-y)*np.log(1-p))
fit=(S==2024)&(k>=5)
b=minimize_scalar(lambda b: li(expit(b*dR))[fit].mean(),bounds=(0,200),method='bounded').x
M={'RPI':expit(b*dR),'RPI+home':T['rpi_home'],'TS same info':T['fair_tuned'],'TS full':T['full']}
rng=np.random.default_rng(1)
def ci(d):
    bs=[d[rng.integers(0,len(d),len(d))].mean() for _ in range(2000)]; return np.percentile(bs,[2.5,97.5])
for name,mask in [('2025 both>=5',(S==2025)&(k>=5)),('2025 Oct15+',(S==2025)&(k>=5)&(dates>=pd.Timestamp('2025-10-15'))),('Tourney',(S==2025)&(dates>=pd.Timestamp('2025-12-04')))]:
    m=mask&~np.isnan(dR)
    print('==',name,'n=',m.sum())
    for nm,p in M.items():
        print(f'  {nm:14s} LL={li(p)[m].mean():.4f} acc={np.mean((p[m]>.5)==(y[m]==1)):.4f}')
    for a,c in [('TS same info','RPI'),('TS full','RPI'),('TS full','RPI+home')]:
        d=li(M[a])[m]-li(M[c])[m]; acc=((M[a][m]>.5)==(y[m]==1)).astype(float)-((M[c][m]>.5)==(y[m]==1))
        dis=m&((M[a]>.5)!=(M[c]>.5))
        print(f'  {a} - {c}: dLL={d.mean():+.4f} CI={np.round(ci(d),4)} dAcc={acc.mean():+.4f} CI={np.round(ci(acc),4)}; disagree n={dis.sum()} {a} right {np.mean((M[a][dis]>.5)==(y[dis]==1)):.3f}')
