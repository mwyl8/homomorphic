"""
esc50_security.py  --  Section 5.6 security study on ESC-50 (run on the t3.xlarge).

Embeds ESC-50 with YAMNet (same pipeline as milvus_1024_esc50.py), reconstructs
ground-truth labels from ESC-50 filenames, and runs the two attacks + the DP sweep.
ESC-50 filename format: {fold}-{clip}-{take}-{target}.wav  (target = class 0..49).
We use: pattern = a target class; group = the 5 ESC-50 major categories (class//10).

USAGE on the instance (ESC-50 audio in ./ESC-50/audio):
    git clone --depth 1 https://github.com/karoldvl/ESC-50
    pip install tensorflow tensorflow_hub librosa soundfile numpy
    python esc50_security.py --audio_dir ESC-50/audio --limit 1000
Outputs esc50_security_results.csv and prints the table to paste into Table 4.
Reuses the exact attack/DP logic already validated on the FSDD embeddings.
"""
import argparse, os, csv, numpy as np

def embed_folder(audio_dir, limit):
    import tensorflow as tf, tensorflow_hub as hub, librosa
    m=hub.load("https://tfhub.dev/google/yamnet/1")
    files=sorted(f for f in os.listdir(audio_dir) if f.lower().endswith(".wav"))[:limit]
    E,cls=[],[]
    for i,f in enumerate(files):
        w,_=librosa.load(os.path.join(audio_dir,f),sr=16000,mono=True)
        _,emb,_=m(w.astype(np.float32))
        E.append(tf.reduce_mean(emb,axis=0).numpy())
        cls.append(int(f.split("-")[-1].split(".")[0]))
        if i%200==0: print(f"  embedded {i}/{len(files)}")
    return np.array(E,dtype=np.float64), np.array(cls)

def auc(sc,lb):
    o=np.argsort(sc); r=np.empty(len(sc)); r[o]=np.arange(1,len(sc)+1)
    p=lb==1; return float((r[p].sum()-p.sum()*(p.sum()+1)/2)/(p.sum()*(~p).sum()))
def probe(E,pat,t=64):
    d=E[pat==1].mean(0)-E[pat==0].mean(0); k=np.argsort(-np.abs(d))[:t]; p=np.zeros_like(d); p[k]=d[k]; return p/(np.linalg.norm(p)+1e-12)
def e1(E,pat,sig,rng): s=E@probe(E,pat); s=s+(rng.normal(0,sig,len(s)) if sig>0 else 0); return auc(s,pat)
def e2(E,grp,sig,rng,nq=400):
    ids=np.unique(grp); qs=rng.choice(len(E),min(nq,len(E)),False); c=0
    for qi in qs:
        best,bv=-1,-1e18
        for g in ids:
            msk=(grp==g).copy(); msk[qi]=False
            if msk.sum()==0: continue
            v=(E[msk]@E[qi]+(rng.normal(0,sig,int(msk.sum())) if sig>0 else 0)).mean()
            if v>bv: bv,best=v,g
        c+=(best==grp[qi])
    return c/len(qs)
def recall(E,sig,rng,k=10,nq=400):
    qs=rng.choice(len(E),min(nq,len(E)),False); t=0
    for qi in qs:
        s=E@E[qi]; s[qi]=-1e18; cl=set(np.argsort(-s)[:k])
        sn=s+(rng.normal(0,sig,len(s)) if sig>0 else 0); sn[qi]=-1e18
        t+=len(cl&set(np.argsort(-sn)[:k]))/k
    return t/len(qs)
def m1_attack_precision(E,pat,k=10,nq=400,rng=None):
    # M1: return only top-k identities (no raw scores). Attack = probe query, measure
    # fraction of returned top-k that are pattern-positive vs. base rate.
    rng=rng or np.random.default_rng(7); p=probe(E,pat); s=E@p
    topk=np.argsort(-s)[:k]; return float(pat[topk].mean()), float(pat.mean())
def sig(e,d=1e-5,R=1.0): return R*R*np.sqrt(2*np.log(1.25/d))/e

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--audio_dir"); ap.add_argument("--limit",type=int,default=1000)
    ap.add_argument("--emb"); ap.add_argument("--labels"); ap.add_argument("--target_class",type=int,default=0)
    a=ap.parse_args()
    if a.emb:
        E=np.load(a.emb).astype(np.float64); cls=np.load(a.labels)
    else:
        E,cls=embed_folder(a.audio_dir,a.limit); np.save("esc50_emb.npy",E); np.save("esc50_labels.npy",cls)
    E=E/(np.linalg.norm(E,axis=1,keepdims=True)+1e-12)
    grp=cls//10  # 5 ESC-50 major categories
    pat=(cls==a.target_class).astype(int)
    print(f"N={len(E)} classes={len(np.unique(cls))} groups={len(np.unique(grp))} target_class={a.target_class} (+{int(pat.sum())})")
    a1=e1(E,pat,0,np.random.default_rng(0)); a2=e2(E,grp,0,np.random.default_rng(1))
    prec,base=m1_attack_precision(E,pat)
    print(f"No mitigation: attack AUC={a1:.3f} (chance .5) | group attr acc={a2:.3f} (chance {1/len(np.unique(grp)):.3f})")
    print(f"M1 (top-k only): attack precision@10={prec:.3f} vs base rate {base:.3f}; retrieval utility stays 1.000")
    rows=[["epsilon","sigma","attack_auc","group_attr_acc","recall_at_10"]]
    for e in [0.1,0.3,1.0,3.0,10.0]:
        g=sig(e); rows.append([e,round(g,2),round(e1(E,pat,g,np.random.default_rng(10)),3),round(e2(E,grp,g,np.random.default_rng(11)),3),round(recall(E,g,np.random.default_rng(12)),3)])
    rows.append(["plain",0.0,round(a1,3),round(a2,3),1.0])
    for r in rows: print(r)
    csv.writer(open("esc50_security_results.csv","w",newline="")).writerows(rows)
    print("saved esc50_security_results.csv")
