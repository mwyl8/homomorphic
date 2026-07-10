"""
bench_common.py -- shared benchmark engine mirroring the repo's exact methods.
Methods (faithful to aheedb.py, aheeqv.py, fhe6.py, fhe_cosine_*, fhe_euclidean_*,
base.py, and phe/Paillier): base, ahe_db, ahe_query, fhe_dot, fhe_cosine,
fhe_euclidean, paillier_db.

Vectors load from recovered_embeddings/{dataset}_{dim}.npy (portable; no Milvus
needed). To scale N beyond the ~1000 real vectors, rows are resampled with tiny
jitter -- legitimate because AHE/FHE dot-product COST is independent of vector
CONTENT (it depends only on N and dim). Timing convention matches the repo:
the initial encryption of DB vectors is EXCLUDED from the timed region; we time
one query scored against the whole DB of N vectors (their reported "per-query").
"""
import os, time, resource, numpy as np

def peak_mb():  # Linux ru_maxrss is KB
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0

def get_query_db(dataset, dim, n, emb_dir="recovered_embeddings", scale_int=100, seed=0):
    X=np.load(f"{emb_dir}/{dataset}_{dim}.npy").astype(np.float64)
    X=X/(np.linalg.norm(X,axis=1,keepdims=True)+1e-12)   # L2-normalize (paper setup)
    if n>len(X):
        rng=np.random.default_rng(seed)
        idx=rng.integers(0,len(X),size=n)
        X=X[idx]+rng.normal(0,1e-3,size=(n,X.shape[1]))
    else:
        X=X[:n]
    q=X[0]; db=X
    q_int=[int(round(v*scale_int)) for v in q]
    db_int=[[int(round(v*scale_int)) for v in row] for row in db]
    return q.tolist(), db.tolist(), q_int, db_int, dim

def _ctx():
    import tenseal as ts
    c=ts.context(ts.SCHEME_TYPE.CKKS,poly_modulus_degree=8192,coeff_mod_bit_sizes=[60,40,40,60])
    c.global_scale=2**40; c.generate_galois_keys(); return c

def _fast_repeated_add(enc_val, times):
    if times<=0: return enc_val
    result=None; cur=enc_val
    while times>0:
        if times%2==1: result=cur if result is None else result+cur
        cur=cur+cur; times//=2
    return result

def run(method, dataset, dim, n, scale_int=100):
    """Returns dict with per_query_s and peak_mem_mb for one query vs N db vectors."""
    q,db,q_int,db_int,dim=get_query_db(dataset,dim,n,scale_int=scale_int)
    base_mem=peak_mb()
    if method=="base":
        t0=time.perf_counter()
        for row in db:
            s=0.0
            for i in range(dim): s+=q[i]*row[i]
        return {"per_query_s":time.perf_counter()-t0,"peak_mem_mb":peak_mb()-base_mem}

    import tenseal as ts
    if method=="paillier_db":
        from phe import paillier
        pub,priv=paillier.generate_paillier_keypair()
        tot=0.0
        for row in db_int:
            enc=[pub.encrypt(int(v)) for v in row]        # encrypt DB vec (excluded)
            t0=time.perf_counter()
            acc=enc[0]*q_int[0]
            for i in range(1,dim): acc=acc+enc[i]*q_int[i]  # ct*plaintext + ct add
            tot+=time.perf_counter()-t0
        return {"per_query_s":tot,"peak_mem_mb":peak_mb()-base_mem}

    ctx=_ctx()
    if method=="ahe_db":
        tot=0.0
        for row in db_int:
            enc=[ts.ckks_vector(ctx,[v]) for v in row]     # encrypt DB vec (excluded)
            t0=time.perf_counter(); acc=ts.ckks_vector(ctx,[0.0])
            for i in range(dim): acc+=_fast_repeated_add(enc[i], q_int[i])
            tot+=time.perf_counter()-t0
        return {"per_query_s":tot,"peak_mem_mb":peak_mb()-base_mem}
    if method=="ahe_query":
        encq=[ts.ckks_vector(ctx,[v]) for v in q_int]      # encrypt query once
        tot=0.0
        for row in db_int:
            t0=time.perf_counter(); acc=ts.ckks_vector(ctx,[0.0])
            for i in range(dim): acc+=_fast_repeated_add(encq[i], row[i])
            tot+=time.perf_counter()-t0
        return {"per_query_s":tot,"peak_mem_mb":peak_mb()-base_mem}
    if method in ("fhe_dot","fhe_cosine","fhe_euclidean"):
        encq=[ts.ckks_vector(ctx,[x]) for x in q]
        tot=0.0
        for row in db:
            encdb=[ts.ckks_vector(ctx,[v]) for v in row]   # encrypt DB vec (excluded)
            t0=time.perf_counter()
            if method=="fhe_dot":
                d=None
                for i in range(dim):
                    t=encq[i]*encdb[i]; d=t if d is None else d+t
            elif method=="fhe_cosine":
                d=None; nrm=None
                for i in range(dim):
                    t=encq[i]*encdb[i]; d=t if d is None else d+t
                    s=encdb[i]*encdb[i]; nrm=s if nrm is None else nrm+s  # extra ct*ct self-norm
            else:  # fhe_euclidean : sum (x-y)^2
                d=None
                for i in range(dim):
                    diff=encq[i]-encdb[i]; sq=diff*diff; d=sq if d is None else d+sq
            tot+=time.perf_counter()-t0
        return {"per_query_s":tot,"peak_mem_mb":peak_mb()-base_mem}
    raise ValueError("unknown method "+method)

if __name__=="__main__":
    import sys, json
    method=sys.argv[1]; dataset=sys.argv[2]; dim=int(sys.argv[3]); n=int(sys.argv[4])
    scale=int(sys.argv[5]) if len(sys.argv)>5 else 100
    r=run(method,dataset,dim,n,scale_int=scale)
    print(json.dumps({"method":method,"dataset":dataset,"dim":dim,"n":n,**r}))
