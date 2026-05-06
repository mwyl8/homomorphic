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


# === Core AHE-style dot product logic ===
#@track_performance
def ahe_repeated_add():
    # === 1. Connect to Milvus and fetch vectors ===
    client = MilvusClient("milvus_demo.db")
    collection_name = "audio_vector_128"

    results = client.query(
        collection_name=collection_name,
        output_fields=["vector"],
        limit=1000
    )

    #print(results)

    all_vectors = [r["vector"] for r in results]
    dim = len(all_vectors[0])
    print("extract vectors done")
    # === 2. Use first vector as query, encrypt the rest as DB ===
    query_vector = [int(round(v*100)) for v in all_vectors[0]]
    db_vectors = [ [int(round(val*100)) for val in vec] for vec in all_vectors ]

    # === 3. TenSEAL context ===
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60]
    )
    context.global_scale = 2**40
    context.generate_galois_keys()

    # === 4. Fast repeated add (log-time) ===
    def fast_repeated_add(enc_val, times):
        if times <= 0:
            #return ts.ckks_vector(context, [0.0])
            return enc_val #return itself instead of doing one encryption
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
        counter = 0
        time = 0
        for db_vec in db_vectors:
            encrypted_vector = [ts.ckks_vector(context, [v]) for v in db_vec]
            if counter > 999:
                break
            counter = counter + 1
            print(counter)
            enc_dot = ts.ckks_vector(context, [0.0])
            start = perf_counter()
            #encrypted_vector = [ts.ckks_vector(context, [v]) for v in db_vec]
            for i in range(dim):
                x_i = query_vector[i]
                partial = fast_repeated_add(encrypted_vector[i], x_i)
                enc_dot += partial
            encrypted_dot_products.append(enc_dot)
            end = perf_counter()
            time += end - start
        return time

    print(doing_stuff())


    # === 6. Output timing info ===
    print(f"Encrypted dot products computed: {len(encrypted_dot_products)}")

# === Run benchmark ===
ahe_repeated_add()
