from pymilvus import MilvusClient
import tenseal as ts
from time import perf_counter
import psutil
import os

# === 1. Connect to Milvus ===
client = MilvusClient("milvus_demo.db")
collection_name = "audio_vector_1024"

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
    return time


# === 8. Run the benchmark ===
print(fhe_manual_dot())

