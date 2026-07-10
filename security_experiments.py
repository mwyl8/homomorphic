"""
security_experiments.py
Empirical security study for the AHE music-IR paper (KDD revision, Contribution 1).

Runs three experiments on decrypted similarity scores:
  (E1) Melodic-pattern inference  -> attack ROC-AUC
  (E2) Creator-identity inference -> top-1 attribution accuracy
  (E3) Mitigations M1 (top-k) + M2 (Gaussian DP noise), swept over epsilon
       -> attack success AND retrieval utility (Recall@10) vs epsilon

USAGE (on the paper's real setup):
  python security_experiments.py --emb embeddings.npy --pattern pattern.npy --creator creator.npy
    embeddings.npy : (N, d) float array of YAMNet embeddings (same as the paper)
    pattern.npy    : (N,)  0/1 array marking a target musical pattern/tag
    creator.npy    : (N,)  int array of creator/group id per track
If no files are given, a synthetic dataset is generated ONLY to validate the code
and illustrate the expected shape (clearly labeled; NOT for the paper).
"""
import argparse, numpy as np

def normalize(X):
    n=np.linalg.norm(X,axis=1,keepdims=True); n[n==0]=1.0; return X/n

def auc(scores,labels):
    # rank-based ROC-AUC (Mann-Whitney), no sklearn dependency
    order=np.argsort(scores); ranks=np.empty_like(order,dtype=float); ranks[order]=np.arange(1,len(scores)+1)
    pos=labels==1; nneg=(~pos).sum(); npos=pos.sum()
    if npos==0 or nneg==0: return float('nan')
    return (ranks[pos].sum()-npos*(npos+1)/2)/(npos*nneg)

def build_probe(E,pattern,top_t=32):
    pos=E[pattern==1].mean(0); neg=E[pattern==0].mean(0); diff=pos-neg
    keep=np.argsort(-np.abs(diff))[:top_t]; probe=np.zeros_like(diff); probe[keep]=diff[keep]
    return probe/ (np.linalg.norm(probe)+1e-12)

def e1_melodic(E,pattern,noise_sigma=0.0,rng=None):
    probe=build_probe(E,pattern); s=E@probe
    if noise_sigma>0: s=s+ (rng or np.random).normal(0,noise_sigma,size=s.shape)
    return auc(s,pattern)

def e2_creator(E,creator,noise_sigma=0.0,rng=None,n_query=300):
    rng=rng or np.random.default_rng(0); ids=np.unique(creator); correct=0; tot=0
    q_idx=rng.choice(len(E),size=min(n_query,len(E)),replace=False)
    for qi in q_idx:
        q=E[qi]; true=creator[qi]; best=None; bestval=-1e18
        for c in ids:
            mask=(creator==c); mask[qi]=False
            if mask.sum()==0: continue
            sc=E[mask]@q
            if noise_sigma>0: sc=sc+rng.normal(0,noise_sigma,size=sc.shape)
            val=sc.mean()
            if val>bestval: bestval=val; best=c
        correct+= (best==true); tot+=1
    return correct/max(tot,1)

def recall_at_k(E,noise_sigma,rng,k=10,n_query=300):
    rng=rng or np.random.default_rng(1); q_idx=rng.choice(len(E),size=min(n_query,len(E)),replace=False); tot=0.0
    for qi in q_idx:
        s=E@E[qi]; s[qi]=-1e18; clean=set(np.argsort(-s)[:k])
        sn=s+ (rng.normal(0,noise_sigma,size=s.shape) if noise_sigma>0 else 0.0); sn[qi]=-1e18
        noisy=set(np.argsort(-sn)[:k]); tot+=len(clean&noisy)/k
    return tot/len(q_idx)

def dp_sigma(eps,delta=1e-5,R=1.0):
    return (R*R)*np.sqrt(2*np.log(1.25/delta))/eps

def synth(N=1000,d=128,C=10,seed=0):
    rng=np.random.default_rng(seed); centers=rng.normal(0,1,size=(C,d))
    creator=rng.integers(0,C,size=N); E=centers[creator]+rng.normal(0,0.6,size=(N,d))
    pat_dir=rng.normal(0,1,size=d); pat_dir[32:]=0; pattern=(rng.random(N)<0.3).astype(int)
    E=E+ pattern[:,None]*pat_dir[None,:]*1.2
    return normalize(E),pattern,creator

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--emb"); ap.add_argument("--pattern"); ap.add_argument("--creator")
    a=ap.parse_args(); rng=np.random.default_rng(42)
    if a.emb:
        E=normalize(np.load(a.emb)); pattern=np.load(a.pattern).astype(int); creator=np.load(a.creator).astype(int)
        tag="REAL"
    else:
        E,pattern,creator=synth(); tag="SYNTHETIC (validation only, not for paper)"
    print(f"[data={tag}] N={len(E)} d={E.shape[1]} creators={len(np.unique(creator))} pattern+={int(pattern.sum())}")
    baseC=1.0/len(np.unique(creator))
    print(f"\n== No mitigation ==")
    print(f"E1 melodic-pattern attack AUC     : {e1_melodic(E,pattern):.3f}  (chance 0.500)")
    print(f"E2 creator attribution accuracy   : {e2_creator(E,creator,rng=rng):.3f}  (chance {baseC:.3f})")
    print(f"\n== E3 privacy-utility vs epsilon (M2 Gaussian noise, delta=1e-5, R=1) ==")
    print(f"{'eps':>6} {'sigma':>8} {'attackAUC':>10} {'attrAcc':>9} {'Recall@10':>10}")
    for eps in [0.1,0.3,1.0,3.0,10.0]:
        sig=dp_sigma(eps)
        a1=e1_melodic(E,pattern,noise_sigma=sig,rng=np.random.default_rng(1))
        a2=e2_creator(E,creator,noise_sigma=sig,rng=np.random.default_rng(2))
        r=recall_at_k(E,sig,np.random.default_rng(3))
        print(f"{eps:6.1f} {sig:8.2f} {a1:10.3f} {a2:9.3f} {r:10.3f}")
    print(f"{'plain':>6} {0.0:8.2f} {e1_melodic(E,pattern):10.3f} {e2_creator(E,creator,rng=np.random.default_rng(2)):9.3f} {1.000:10.3f}")

if __name__=="__main__": main()
