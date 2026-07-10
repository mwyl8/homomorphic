"""
scale_benchmark.py -- sweeps method x dim x N and records per-query runtime and
peak memory into a CSV (the real scaling data for Figures 1-2). Each config runs
in a fresh subprocess so peak memory is clean.

Examples:
  python scale_benchmark.py --quick
  python scale_benchmark.py --methods base ahe_query ahe_db fhe_dot fhe_cosine fhe_euclidean paillier_db \
      --dims 128 256 512 1024 --Ns 250 500 1000 2000 3000 --dataset magnatagatune --out scale_results.csv
Notes: FHE/Paillier at large N x dim are slow; use --quick first, then widen.
"""
import argparse, csv, json, subprocess, sys, os, time
ap=argparse.ArgumentParser()
ap.add_argument("--methods",nargs="+",default=["base","ahe_query","ahe_db","fhe_dot","fhe_cosine","fhe_euclidean","paillier_db"])
ap.add_argument("--dims",nargs="+",type=int,default=[128,256,512,1024])
ap.add_argument("--Ns",nargs="+",type=int,default=[250,500,1000,2000,3000])
ap.add_argument("--dataset",default="magnatagatune")
ap.add_argument("--out",default="scale_results.csv")
ap.add_argument("--quick",action="store_true")
a=ap.parse_args()
if a.quick:
    a.methods=["base","ahe_db","fhe_dot"]; a.dims=[128,1024]; a.Ns=[250,500,1000]
rows=[]
with open(a.out,"w",newline="") as fh:
    w=csv.writer(fh); w.writerow(["method","dataset","dim","N","per_query_s","peak_mem_mb"]); fh.flush()
    for m in a.methods:
        for d in a.dims:
            for n in a.Ns:
                t0=time.time()
                try:
                    out=subprocess.run([sys.executable,"bench_common.py",m,a.dataset,str(d),str(n)],
                                       capture_output=True,text=True,timeout=3600)
                    r=json.loads(out.stdout.strip().splitlines()[-1])
                    row=[m,a.dataset,d,n,round(r["per_query_s"],4),round(r["peak_mem_mb"],1)]
                except Exception as e:
                    row=[m,a.dataset,d,n,"ERR",str(e)[:40]]
                w.writerow(row); fh.flush()
                print(f"{m:14s} dim={d:4d} N={n:5d} -> {row[4]} s, {row[5]} MB  ({time.time()-t0:.0f}s)")
print("wrote",a.out)
