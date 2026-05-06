from pymilvus import MilvusClient
import numpy as np

# -------- CHANGE THIS IF YOU WANT --------
DB_FILE = "milvus_demo.db"
COLLECTION = "fma_1000_audio_vectors"  # change to 128/256/512 version if needed
# -----------------------------------------

client = MilvusClient(DB_FILE)

print("\nCollection info:")
print(client.describe_collection(COLLECTION))

rows = client.query(
    collection_name=COLLECTION,
    filter="id >= 0",
    output_fields=["id", "vector"],
    limit=5
)

print("\nShowing first 5 vectors:\n")

for r in rows:
    v = np.array(r["vector"], dtype=np.float32)

    print("ID:", r["id"])
    print("Length:", len(v))
    print("First 20 values:", v[:20])
    print("Min/Max:", float(v.min()), float(v.max()))
    print("-" * 60)
