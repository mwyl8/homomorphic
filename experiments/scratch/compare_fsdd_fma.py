from pymilvus import MilvusClient
import numpy as np

DB = "milvus_demo.db"
FSDD = "fsdd_audio_vector_1024"
FMA  = "fma_1000_audio_vectors"   # your 1024-d one

client = MilvusClient(DB)

def stats(col, n=200):
    rows = client.query(
        collection_name=col,
        filter="id >= 0",
        output_fields=["vector"],
        limit=n
    )
    V = np.array([r["vector"] for r in rows], dtype=np.float32)
    nonzero = (V != 0).mean()          # fraction nonzero
    return {
        "count": V.shape[0],
        "dim": V.shape[1],
        "min": float(V.min()),
        "max": float(V.max()),
        "mean": float(V.mean()),
        "std": float(V.std()),
        "nonzero_frac": float(nonzero),
    }

print("FSDD:", stats(FSDD))
print("FMA :", stats(FMA))
