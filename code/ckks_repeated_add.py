import tenseal as ts
from pymilvus import MilvusClient
from time import perf_counter
import psutil
import os

# === Performance Tracker ===
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

# === Core AHE-style dot product logic ===
#@track_performance
def ahe_repeated_add():
    # === 1. Connect to Milvus and fetch vectors ===
    client = MilvusClient("milvus_demo.db")
    collection_name = "yamnet_embeddings"

    results = client.query(
        collection_name=collection_name,
        output_fields=["vector"],
        limit=6
    )

    all_vectors = [r["vector"] for r in results]
    dim = len(all_vectors[0])

    # === 2. Use first vector as query, encrypt the rest as DB ===
    query_vector = [int(round(v)) for v in all_vectors[0]]
    db_vectors = [ [int(round(val)) for val in vec] for vec in all_vectors[1:] ]

    # === 3. TenSEAL context ===
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60]
    )
    context.global_scale = 2**40
    context.generate_galois_keys()

    # Encrypt each database vector (element-wise encryption)
    encrypted_db_vectors = []
    for db_vec in db_vectors:
        encrypted_vector = [ts.ckks_vector(context, [v]) for v in db_vec]
        encrypted_db_vectors.append(encrypted_vector)

    # === 4. Fast repeated add (log-time) ===
    def fast_repeated_add(enc_val, times):
        if times <= 0:
            return ts.ckks_vector(context, [0.0])
        result = None
        current = enc_val
        while times > 0:
            if times % 2 == 1:
                result = current if result is None else result + current
            current += current
            times //= 2
        return result

    # === 5. Compute encrypted dot products ===
    encrypted_dot_products = []
    start = perf_counter()

    @track_performance
    def doing_stuff():
        for enc_db_vec in encrypted_db_vectors:
            enc_dot = ts.ckks_vector(context, [0.0])
            for i in range(dim):
                x_i = query_vector[i]
                partial = fast_repeated_add(enc_db_vec[i], x_i)
                enc_dot += partial
            encrypted_dot_products.append(enc_dot)

    doing_stuff()

    end = perf_counter()

    # === 6. Output timing info ===
    print(f"Encrypted dot products computed: {len(encrypted_dot_products)}")
    print(f"Total time (enc_db + pt_query via repeated add): {end - start:.4f} seconds")

# === Run benchmark ===
ahe_repeated_add()
