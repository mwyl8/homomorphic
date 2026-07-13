import tenseal as ts
from pymilvus import MilvusClient
from time import perf_counter
import psutil
import os


# === Performance Tracker ===
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


# === Core AHE Logic ===
def ahe_query_encrypted():
    # === 1. Connect to Milvus and fetch vectors ===
    client = MilvusClient("milvus_demo.db")
    collection_name = "fsdd_audio_vector_1024"

    results = client.query(
        collection_name=collection_name,
        output_fields=["vector"],
        limit=1000,
    )

    # === 2. Extract vectors ===
    all_vectors = [r["vector"] for r in results]
    if not all_vectors:
        raise RuntimeError(f"No vectors found in {collection_name}")
    dim = len(all_vectors[0])
    print("extract vectors done")

    # Use the first vector as the query, rest as database
    query_vector = all_vectors[0]
    database_vectors = all_vectors

    # Convert all values to integers for repeated addition
    query_vector_int = [int(round(v)) for v in query_vector]
    for i in range(len(database_vectors)):
        database_vectors[i] = [int(round(v)) for v in database_vectors[i]]

    # === 3. TenSEAL context and encrypted query ===
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60],
    )
    context.global_scale = 2**40
    context.generate_galois_keys()

    # Encrypt each element of the query vector individually
    enc_query_vector = [ts.ckks_vector(context, [v]) for v in query_vector_int]

    # === 4. Log-time repeated addition ===
    def fast_repeated_add(enc_val, times):
        if times <= 0:
            return enc_val
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

    @track_performance
    def doing_stuff():
        total = min(1000, len(database_vectors))
        time_spent = 0.0
        for idx, db_vec in enumerate(database_vectors[:total], start=1):
            enc_dot = ts.ckks_vector(context, [0.0])
            start = perf_counter()
            for i in range(dim):
                y_i = db_vec[i]
                partial = fast_repeated_add(enc_query_vector[i], y_i)
                enc_dot += partial
            encrypted_dot_products.append(enc_dot)
            end = perf_counter()
            time_spent += end - start
            if idx % 25 == 0 or idx == total:
                print(f"Progress: {idx}/{total} encrypted dot products")
        return time_spent

    print(doing_stuff())

    # === 6. Output ===
    print(f"Encrypted dot products computed: {len(encrypted_dot_products)}")


# === Run the program ===
ahe_query_encrypted()
