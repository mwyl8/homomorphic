"""Reproduces the Section 5.6 security numbers on the recovered FSDD YAMNet embeddings.
Labels are reconstructed exactly from FSDD filenames (staging was sorted(recordings)[:1000]).
Requires: recovered_embeddings/fsdd_1024.npy  and  a clone of the FSDD repo (for filenames).
  git clone --depth 1 https://github.com/Jakobovski/free-spoken-digit-dataset fsdd
"""
import numpy as np, os, csv, sys
FSDD_REC = sys.argv[1] if len(sys.argv)>1 else "fsdd/recordings"
recs=sorted(os.listdir(FSDD_REC))[:1000]
digit=np.array([int(f.split("_")[0]) for f in recs])
speaker=np.array([f.split("_")[1] for f in recs])
E=np.load("recovered_embeddings/fsdd_1024.npy").astype(np.float64)
E=E/(np.linalg.norm(E,axis=1,keepdims=True)+1e-12)
sp={s:i for i,s in enumerate(np.unique(speaker))}; creator=np.array([sp[s] for s in speaker])
def auc(sc,lb):
    o=np.argsort(sc); r=np.empty(len(sc)); r[o]=np.arange(1,len(sc)+1)
    p=lb==1; return float((r[p].sum()-p.sum()*(p.sum()+1)/2)/(p.sum()*(~p).sum()))
def probe(E,pat,t=64):
    d=E[pat==1].mean(0)-E[pat==0].mean(0); k=np.argsort(-np.abs(d))[:t]; p=np.zeros_like(d); p[k]=d[k]; return p/(np.linalg.norm(p)+1e-12)
def e1(E,pat,sig,rng): s=E@probe(E,pat); s=s+(rng.normal(0,sig,len(s)) if sig>0 else 0); return auc(s,pat)
def e2(E,cr,sig,rng,nq=400):
    ids=np.unique(cr); qs=rng.choice(len(E),min(nq,len(E)),False); c=0
    for qi in qs:
        best,bv=-1,-1e18
        for cid in ids:
            m=(cr==cid).copy(); m[qi]=False
            if m.sum()==0: continue
            v=(E[m]@E[qi]+(rng.normal(0,sig,int(m.sum())) if sig>0 else 0)).mean()
            if v>bv: bv,best=v,cid
        c+=(best==cr[qi])
    return c/len(qs)
def rec(E,sig,rng,k=10,nq=400):
    qs=rng.choice(len(E),min(nq,len(E)),False); t=0
    for qi in qs:
        s=E@E[qi]; s[qi]=-1e18; cl=set(np.argsort(-s)[:k])
        sn=s+(rng.normal(0,sig,len(s)) if sig>0 else 0); sn[qi]=-1e18
        t+=len(cl&set(np.argsort(-sn)[:k]))/k
    return t/len(qs)
sig=lambda e,d=1e-5,R=1.0: R*R*np.sqrt(2*np.log(1.25/d))/e
pat=(digit==0).astype(int)
rows=[["epsilon","sigma","attack_auc","attrib_acc","recall_at_10"]]
for e in [0.1,0.3,1.0,3.0,10.0]:
    g=sig(e); rows.append([e,round(g,2),round(e1(E,pat,g,np.random.default_rng(10)),3),round(e2(E,creator,g,np.random.default_rng(11)),3),round(rec(E,g,np.random.default_rng(12)),3)])
rows.append(["plain",0.0,round(e1(E,pat,0,np.random.default_rng(0)),3),round(e2(E,creator,0,np.random.default_rng(1)),3),1.0])
[print(r) for r in rows]
csv.writer(open("security_results_fsdd.csv","w",newline="")).writerows(rows)
