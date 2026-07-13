from pymilvus import MilvusClient
import tenseal as ts
from time import perf_counter
import psutil
import os
import numpy as np

# === 1. Connect to Milvus ===
client = MilvusClient("milvus_demo.db")
collection_name = "audio_vector_128"

# === 2. Fetch vectors ===
results = client.query(
    collection_name=collection_name,
    output_fields=["vector"],
    limit=1000
)

# Extract vectors
database_vectors = [r["vector"] for r in results]
print("extract vectors done")

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

# === 5. Encrypt query vector (element-wise encryption) ===
enc_query = [ts.ckks_vector(context, [x]) for x in query_vector]

# === 6. Performance measurement decorator ===
def track_performance(fn):
    def wrapper(*args, **kwargs):
        process = psutil.Process(os.getpid())
        num_cpus = psutil.cpu_count(logical=True)
        mem_before = process.memory_info().rss / (1024 * 1024)  # MB
        psutil.cpu_percent(interval=None)
        process.cpu_percent(interval=None)
        start_time = perf_counter()

        result = fn(*args, **kwargs)
        end_time = perf_counter()

        cpu_percent_per_core = process.cpu_percent(interval=None)
        cpu_total_percent = cpu_percent_per_core * num_cpus
        mem_after = process.memory_info().rss / (1024 * 1024)

        print("\n--- Performance Report ---")
        print(f"Wall-clock time: {end_time - start_time:.4f} seconds")
        print(f"CPU utilization (total across {num_cpus} cores): {cpu_total_percent:.1f}%")
        print(f"Memory used: {mem_after - mem_before:.2f} MB")
        print("--------------------------")

        return result

    return wrapper

# === 7. Main FHE scalar-by-scalar dot product function ===
@track_performance
def fhe_manual_dot():
    enc_dot_products = []
    counter=0
    time = 0
    for db_vec in database_vectors:
        print(counter)
        if counter >= 1000:
            break
        counter += 1
        # --- EXCLUDE: Encryption step (outside timed section)
        encrypted_db_vector = [ts.ckks_vector(context, [v]) for v in db_vec]

        # --- BEGIN TIMED SECTION (everything else)
        dot_product = None
        start = perf_counter()
        for i in range(dim):
            enc_xi = enc_query[i]  # Already encrypted query vector element
            enc_yi = encrypted_db_vector[i]  # Encrypted DB vector element
            term = enc_xi * enc_yi
            dot_product = term if dot_product is None else dot_product + term

        enc_dot_products.append(dot_product)
        end = perf_counter()
        time += end - start
        # --- END TIMED SECTION

    print(f"\nEncrypted dot products computed: {len(enc_dot_products)}")
    return enc_dot_products, time

# === 8. Run the benchmark ===
enc_dot_products, fhe_time = fhe_manual_dot()

# === 9. Recall@10 calculation ===
# Ground truth (plaintext dot products)
plain_scores = [np.dot(query_vector, v) for v in database_vectors]
ground_truth_top10 = np.argsort(plain_scores)[::-1][:10]

# Decrypt FHE results
decrypted_scores = [dp.decrypt()[0] for dp in enc_dot_products]
fhe_top10 = np.argsort(decrypted_scores)[::-1][:10]

# Recall@10
overlap = len(set(fhe_top10) & set(ground_truth_top10))
recall_at_10 = overlap / 10.0

print(f"\nTop-10 neighbors (FHE ranking): {fhe_top10}")
print(f"Top-10 neighbors (Ground truth): {ground_truth_top10}")
print(f"Recall@10: {recall_at_10:.2f}")
