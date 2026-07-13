"""
fidelity_benchmark.py -- regenerates Table 1 (ranking fidelity) with real numbers,
comparing plaintext similarity search to the AHE (CKKS packed additive dot) result.
Machine-independent: rankings + max score error. Also checks the Blocked Inner
Product is lossless vs the flat product.

  python fidelity_benchmark.py --datasets magnatagatune esc50 fma fsdd --dim 1024 --queries 40
Outputs fidelity_results.csv.
"""
import argparse, csv, numpy as np
from scipy.stats import spearmanr, kendalltau

def load(ds,dim):
    X=np.load(f"recovered_embeddings/{ds}_{dim}.npy").astype(np.float64)
    return X/(np.linalg.norm(X,axis=1,keepdims=True)+1e-12)

def ndcg_at_k(enc_order, true_top, k=10):
    rel=[1.0 if i in true_top else 0.0 for i in enc_order[:k]]
    dcg=sum(r/np.log2(idx+2) for idx,r in enumerate(rel))
    ideal=sum(1.0/np.log2(idx+2) for idx in range(min(k,len(true_top))))
    return dcg/ideal if ideal>0 else 1.0

def main():
    import tenseal as ts
    ap=argparse.ArgumentParser()
    ap.add_argument("--datasets",nargs="+",default=["magnatagatune","esc50","fma","fsdd"])
    ap.add_argument("--dim",type=int,default=1024); ap.add_argument("--queries",type=int,default=40)
    a=ap.parse_args()
    ctx=ts.context(ts.SCHEME_TYPE.CKKS,poly_modulus_degree=8192,coeff_mod_bit_sizes=[60,40,40,60])
    ctx.global_scale=2**40; ctx.generate_galois_keys()
    rows=[["dataset","R@10","nDCG@10","Spearman","Kendall","max_err"]]
    rng=np.random.default_rng(0)
    for ds in a.datasets:
        X=load(ds,a.dim); N=len(X); enc_db=[ts.ckks_vector(ctx,X[j].tolist()) for j in range(N)]
        qs=rng.choice(N,min(a.queries,N),replace=False)
        R=[]; ND=[]; SP=[]; KT=[]; ME=0.0; blk_err=0.0
        for qi in qs:
            q=X[qi]; plain=X@q
            encs=np.array([enc_db[j].dot(q.tolist()).decrypt()[0] for j in range(N)])
            ME=max(ME,float(np.max(np.abs(encs-plain))))
            po=np.argsort(-plain); eo=np.argsort(-encs)
            pt=set(po[:10].tolist())
            R.append(len(pt & set(eo[:10].tolist()))/10)
            ND.append(ndcg_at_k(eo.tolist(),pt,10))
            SP.append(spearmanr(plain,encs).statistic); KT.append(kendalltau(plain,encs).statistic)
            # blocked == flat (plaintext identity check), k=4 blocks
            k=4; bs=a.dim//k; blk=sum(q[b*bs:(b+1)*bs]@X[:,b*bs:(b+1)*bs].T for b in range(k))
            blk_err=max(blk_err,float(np.max(np.abs(blk-plain))))
        rows.append([ds,round(np.mean(R),3),round(np.mean(ND),3),round(np.mean(SP),6),round(np.mean(KT),6),f"{ME:.1e}"])
        print(f"{ds}: R@10={np.mean(R):.3f} nDCG={np.mean(ND):.3f} Spearman={np.mean(SP):.6f} Kendall={np.mean(KT):.6f} maxerr={ME:.1e} blocked_vs_flat={blk_err:.1e}")
    csv.writer(open("fidelity_results.csv","w",newline="")).writerows(rows)
    print("wrote fidelity_results.csv")

if __name__=="__main__": main()
