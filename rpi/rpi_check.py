import pandas as pd, numpy as np, re
d=pd.read_csv('data/wvb_teammatch_div1_2025.csv')
d['Date']=pd.to_datetime(d.Date.astype(str).str.extract(r'(\d+/\d+/\d+)')[0],format='%m/%d/%Y')
def opp(s):
    s=str(s).strip()
    if s.startswith('@ '): return s[2:].strip(),'away'
    if ' @' in s: return s.split(' @')[0].strip(),'neutral'
    return s,'home'
d[['Opp','Loc']]=d.Opponent.apply(lambda s:pd.Series(opp(s)))
d['W']=d.Result.astype(str).str.startswith('W')
vc=d.Opp.value_counts(); teams=set(d.Team)|set(vc[vc>=10].index)
print('teams',len(teams))
print('rows',len(d),'opp matched to D1 team:',d.Opp.isin(teams).mean().round(3))
cut=pd.Timestamp('2025-11-24')
a=d[(d.Date<=cut)&d.Opp.isin(teams)].copy()
a['key']=a.apply(lambda r:(r.Date,)+tuple(sorted([r.Team,r.Opp])),axis=1)
m=a.drop_duplicates('key')[['Team','Opp','W']]
# games list both directions
g=pd.concat([m.rename(columns={'Team':'t','Opp':'o'}),m.rename(columns={'Opp':'t','Team':'o'}).assign(W=lambda x:~x.W)])
W=g.groupby('t').W.sum(); N=g.groupby('t').W.count()
wp=W/N
# OWP: for each game of t vs o, WP of o excluding games vs t
pair=g.groupby(['t','o']).W.agg(['sum','count'])
def wp_excl(o,t):
    w,n=W[o],N[o]
    if (o,t) in pair.index: w-=pair.loc[(o,t),'sum']; n-=pair.loc[(o,t),'count']
    return w/n if n>0 else np.nan
g['owp_o']=[wp_excl(o,t) for t,o in zip(g.t,g.o)]
owp=g.groupby('t').owp_o.mean()
g['oo']=g.o.map(owp)
oowp=g.groupby('t').oo.mean()
rpi=(0.25*wp+0.5*owp+0.25*oowp).sort_values(ascending=False)
ours=pd.Series(range(1,len(rpi)+1),index=rpi.index)
off="""Nebraska,Texas,Arizona St.,Kentucky,Creighton,Pittsburgh,Louisville,SMU,Texas A&M,Stanford,Purdue,Wisconsin,Indiana,Kansas,Minnesota,TCU,Southern California,Miami (FL),UTEP,Iowa St.,BYU,Colorado,Rice,Tennessee,Xavier,UNI,Western Ky.,Baylor,San Diego,South Dakota St.,Penn St.,Kansas St.,Utah St.,Marquette,Utah,Florida,Tulsa,Michigan,Villanova,South Fla.,Georgia Tech,UCLA,Cincinnati,Arizona,Texas St.,Washington,Pepperdine,James Madison,North Carolina,Illinois,Dayton,Missouri,Arkansas St.,Auburn""".split(',')
miss=[t for t in off if t not in ours.index]
print('unmatched names:',miss)
import difflib
for t in miss: print(t,'->',difflib.get_close_matches(t,list(ours.index),3,0.5))
from scipy.stats import spearmanr
ok=[t for t in off if t in ours.index]
r_off=[off.index(t)+1 for t in ok]; r_our=[ours[t] for t in ok]
print('Spearman (official top-54 vs ours):',round(spearmanr(r_off,r_our).correlation,3))
print('mean |rank diff|:',np.mean(np.abs(np.array(r_off)-np.array(r_our))).round(1))
print('our top54 overlap with official top54:',len(set(ours.index[:54])&set(off)))
for t in ok[:25]: print(f'{off.index(t)+1:3d} {ours[t]:4d} {t:20s} RPI={rpi[t]:.4f}')
for t in ["UTEP","North Carolina","Xavier","Michigan"]: print("COMP",t,"WP=%.3f OWP=%.3f OOWP=%.3f RPI=%.4f ourrank=%d"%(wp[t],owp[t],oowp[t],rpi[t],ours[t]))
big=sorted(ok,key=lambda t:-abs(ours[t]-(off.index(t)+1)))[:8]
print('largest gaps:',[(t,off.index(t)+1,ours[t]) for t in big])
print('in our top 54 but not official:',[ (t,ours[t]) for t in ours.index[:54] if t not in off])

