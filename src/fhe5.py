from pymilvus import MilvusClient
import tenseal as ts
from time import perf_counter
import psutil
import os

# === 1. Connect to Milvus ===
client = MilvusClient("milvus_demo.db")
collection_name = "yamnet_embeddings"

# === 2. Fetch vectors ===
results = client.query(
    collection_name=collection_name,
    output_fields=["vector"],
    limit=10
)

# Extract vectors
database_vectors = [r["vector"] for r in results]

# === 3. Use first as query vector ===
query_vector = database_vectors[0]
dim = len(query_vector)

# === 4. TenSEAL CKKS context ===
context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=8192,
    coeff_mod_bit_sizes=[60, 40, 40, 60]
)
context.global_scale = 2**40
context.generate_galois_keys()

# === 5. Encrypt query vector (element-wise) ===
enc_query = [ts.ckks_vector(context, [x]) for x in query_vector]

# === 6. Performance wrapper ===
def track_performance(fn):
    def wrapper(*args, **kwargs):
        process = psutil.Process(os.getpid())
        cpu_start = process.cpu_times()
        mem_before = process.memory_info().rss / (1024 * 1024)  # MB
        start_time = perf_counter()

        result = fn(*args, **kwargs)

        end_time = perf_counter()
        mem_after = process.memory_info().rss / (1024 * 1024)
        cpu_end = process.cpu_times()

        print("\n--- Performance Report ---")
        print(f"CPU time (user): {cpu_end.user - cpu_start.user:.2f}s")
        print(f"CPU time (system): {cpu_end.system - cpu_start.system:.2f}s")
        print(f"Memory used: {mem_after - mem_before:.2f} MB")
        print(f"Wall-clock time: {end_time - start_time:.4f} seconds")
        print("--------------------------")

        return result
    return wrapper

# === 7. Main FHE dot product loop ===
@track_performance
def fhe_manual_dot():
    enc_dot_products = []
    for db_vec in database_vectors:
        dot_product = None
        for i in range(dim):
            enc_xi = enc_query[i]
            enc_yi = ts.ckks_vector(context, [db_vec[i]])
            term = enc_xi * enc_yi
            dot_product = term if dot_product is None else dot_product + term
        enc_dot_products.append(dot_product)
    print(f"\nEncrypted dot products computed: {len(enc_dot_products)}")

# === 8. Run benchmark ===
fhe_manual_dot()
