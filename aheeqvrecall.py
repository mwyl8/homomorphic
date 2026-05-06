import tenseal as ts
from pymilvus import MilvusClient
from time import perf_counter
import psutil
import os
import numpy as np

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


# === Core AHE Logic with Recall@10 ===
def ahe_query_encrypted():
    # === 1. Connect to Milvus and fetch vectors ===
    client = MilvusClient("milvus_demo.db")
    collection_name = "audio_vector_1024"

    results = client.query(
        collection_name=collection_name,
        output_fields=["vector"],
        limit=1000
    )

    # === 2. Extract vectors ===
    all_vectors = [r["vector"] for r in results]
    dim = len(all_vectors[0])
    print("extract vectors done")
    
    query_vector = all_vectors[0]
    database_vectors = all_vectors

    # Round values to integers for AHE repeated addition
    query_vector_int = [int(round(v)) for v in query_vector]
    for i in range(len(database_vectors)):
        database_vectors[i] = [int(round(v)) for v in database_vectors[i]]

    # === 3. TenSEAL context and encrypted query ===
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60]
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
        counter = 0
        total_time = 0
        for db_vec in database_vectors:
            enc_dot = ts.ckks_vector(context, [0.0]) 
            print(counter)  # <-- progress monitor
            counter += 1
            if counter > 999:  # stop after 1000 for now
                break
            start = perf_counter()
            for i in range(dim):
                y_i = db_vec[i]
                partial = fast_repeated_add(enc_query_vector[i], y_i)
                enc_dot += partial
            encrypted_dot_products.append(enc_dot)
            end = perf_counter()
            total_time += end - start
        return total_time

    print(doing_stuff())
    print(f"Encrypted dot products computed: {len(encrypted_dot_products)}")

    # === 6. Decrypt and get Top-10 neighbors ===
    decrypted_scores = [enc_dot.decrypt()[0] for enc_dot in encrypted_dot_products]
    encrypted_top10 = np.argsort(decrypted_scores)[-10:][::-1].tolist()
    print("Top-10 neighbors (encrypted ranking):", encrypted_top10)

    # === 7. Compute ground truth with plaintext dot products ===
    ground_truth_scores = [np.dot(query_vector, db_vec) for db_vec in database_vectors]
    ground_truth_top10 = np.argsort(ground_truth_scores)[-10:][::-1].tolist()
    print("Top-10 neighbors (ground truth):", ground_truth_top10)

    # === 8. Recall@10 ===
    recall_at_10 = len(set(encrypted_top10).intersection(set(ground_truth_top10))) / 10
    print(f"Recall@10: {recall_at_10:.2f}")


# === Run the program ===
ahe_query_encrypted()
