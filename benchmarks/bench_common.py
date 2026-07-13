"""
bench_common.py -- PACKED CKKS benchmark engine (matches the paper's methodology:
each vector packed into ONE ciphertext, one plaintext-weighted dot product; no
per-coordinate repeated addition). Methods: base, ahe_query, ahe_db, fhe_dot,
fhe_cosine, fhe_euclidean, paillier_db.

Memory: for DB-encrypting methods (ahe_db, fhe_*), all N ciphertexts are held so
peak memory grows with N (matches the paper's memory-vs-N story); ahe_query holds
only the encrypted query. Timing excludes the initial encryption (as in the repo).
Vectors load from recovered_embeddings/{dataset}_{dim}.npy (L2-normalized). N above
the ~1000 real vectors is reached by resampling with jitter (cost is content-free).
"""
import time, resource, numpy as np
def peak_mb(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0
def get_query_db(dataset,dim,n,emb_dir="recovered_embeddings",seed=0):
    X=np.load(f"{emb_dir}/{dataset}_{dim}.npy").astype(np.float64)
    X=X/(np.linalg.norm(X,axis=1,keepdims=True)+1e-12)
    if n>len(X):
        rng=np.random.default_rng(seed); idx=rng.integers(0,len(X),n)
        X=X[idx]+rng.normal(0,1e-3,size=(n,X.shape[1]))
    else: X=X[:n]
    return X[0].tolist(), [r.tolist() for r in X], dim
def _ctx():
    import tenseal as ts
    c=ts.context(ts.SCHEME_TYPE.CKKS,poly_modulus_degree=8192,coeff_mod_bit_sizes=[60,40,40,60])
    c.global_scale=2**40; c.generate_galois_keys(); return c

def run(method,dataset,dim,n,scale_int=None):
    q,db,dim=get_query_db(dataset,dim,n); base=peak_mb()
    if method=="base":
        t0=time.perf_counter()
        for row in db:
            s=0.0
            for i in range(dim): s+=q[i]*row[i]
        return {"per_query_s":time.perf_counter()-t0,"peak_mem_mb":round(peak_mb()-base,1)}
    if method=="paillier_db":
        from phe import paillier
        pub,priv=paillier.generate_paillier_keypair(); SC=1000
        q_int=[int(round(v*SC)) for v in q]
        enc_db=[[pub.encrypt(int(round(v*SC))) for v in row] for row in db]  # held: memory grows w/ N
        m=peak_mb()-base; t0=time.perf_counter()
        for enc in enc_db:
            acc=enc[0]*q_int[0]
            for i in range(1,dim): acc=acc+enc[i]*q_int[i]
        return {"per_query_s":time.perf_counter()-t0,"peak_mem_mb":round(m,1)}
    import tenseal as ts; ctx=_ctx()
    if method=="ahe_query":
        encq=ts.ckks_vector(ctx,q); m=peak_mb()-base; t0=time.perf_counter()
        for row in db: s=encq.dot(row)                # ct x plaintext packed dot
        return {"per_query_s":time.perf_counter()-t0,"peak_mem_mb":round(m,1)}
    if method=="ahe_db":
        enc_db=[ts.ckks_vector(ctx,row) for row in db]  # N ciphertexts held
        m=peak_mb()-base; t0=time.perf_counter()
        for enc in enc_db: s=enc.dot(q)                # ct x plaintext packed dot
        return {"per_query_s":time.perf_counter()-t0,"peak_mem_mb":round(m,1)}
    if method in ("fhe_dot","fhe_cosine","fhe_euclidean"):
        encq=ts.ckks_vector(ctx,q); enc_db=[ts.ckks_vector(ctx,row) for row in db]
        m=peak_mb()-base; t0=time.perf_counter()
        for enc in enc_db:
            if method=="fhe_dot": s=encq.dot(enc)                 # ct x ct
            elif method=="fhe_cosine": s=encq.dot(enc); nrm=enc.dot(enc)  # 2x ct x ct (self-norm)
            else: diff=encq-enc; s=diff.dot(diff)                 # ct x ct
        return {"per_query_s":time.perf_counter()-t0,"peak_mem_mb":round(m,1)}
    raise ValueError("unknown method "+method)

if __name__=="__main__":
    import sys,json
    method=sys.argv[1]; dataset=sys.argv[2]; dim=int(sys.argv[3]); n=int(sys.argv[4])
    print(json.dumps({"method":method,"dataset":dataset,"dim":dim,"n":n,**run(method,dataset,dim,n)}))
