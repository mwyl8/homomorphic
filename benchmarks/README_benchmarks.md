# Benchmark suite (run on a clean t3.xlarge)

Regenerates and EXTENDS the paper's real data. All scripts read the recovered
embeddings in `recovered_embeddings/` (portable; no Milvus needed).

## Setup
```
pip install tenseal phe pymilvus numpy scipy psutil tensorflow tensorflow_hub librosa soundfile
# recovered_embeddings/ must be in the working directory
```

## 1. Scaling data for runtime + memory (fixes the "padding" figures)
```
python scale_benchmark.py --quick            # sanity check first
python scale_benchmark.py --dims 128 256 512 1024 --Ns 250 500 1000 2000 3000
```
-> `scale_results.csv` (method, dim, N, per_query_s, peak_mem_mb). Gives multi-point
runtime-vs-N and memory-vs-N curves for AHE Query, AHE DB, FHE Dot/Cos/Euc, Paillier, base.
FHE/Paillier at large N x dim are slow; widen N gradually.

## 2. Ranking fidelity (Table 1, real numbers)
```
python fidelity_benchmark.py --datasets magnatagatune esc50 fma fsdd --dim 1024
```
-> `fidelity_results.csv` (R@10, nDCG@10, Spearman, Kendall, max_err) + blocked-vs-flat check.

## 3. Security study (Section 5.6)
```
# FSDD (labels reconstructed from filenames):
git clone --depth 1 https://github.com/Jakobovski/free-spoken-digit-dataset fsdd
python run_fsdd_security.py fsdd/recordings           # -> security_results_fsdd.csv
# ESC-50 (music-relevant; embeds fresh with YAMNet):
git clone --depth 1 https://github.com/karoldvl/ESC-50
python esc50_security.py --audio_dir ESC-50/audio --limit 1000   # -> esc50_security_results.csv
```

Send me the CSVs and I fold the numbers + rebuilt figures into main_kdd.tex.
