import tenseal as ts
from pymilvus import MilvusClient
from time import perf_counter
import psutil
import os

def track_performance(fn):
    def wrapper(*args, **kwargs):
        process = psutil.Process(os.getpid())
        num_cpus = psutil.cpu_count(logical=True)
        mem_before = process.memory_info().rss / (1024 * 1024)  # MB
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


def ahe_repeated_add(collection_name):
    print(f"\n=== Running encrypted-database AHE for collection: {collection_name} ===")

    client = MilvusClient("milvus_demo.db")
    if not client.has_collection(collection_name=collection_name):
        print(f"Collection {collection_name} not found, skipping.")
        return

    results = client.query(
        collection_name=collection_name,
        output_fields=["vector"],
        limit=1000
    )

    all_vectors = [r["vector"] for r in results if "vector" in r]
    if len(all_vectors) < 2:
        print(f"Too few vectors in {collection_name}, skipping.")
        return

    dim = len(all_vectors[0])
    query_vector = [int(round(v * 100)) for v in all_vectors[0]]
    db_vectors = [[int(round(val * 100)) for val in vec] for vec in all_vectors]

    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60]
    )
    context.global_scale = 2**40
    context.generate_galois_keys()

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

    encrypted_dot_products = []

    @track_performance
    def doing_stuff():
        counter = 0
        for db_vec in db_vectors:
            if counter >= 1000:
                break
            encrypted_vector = [ts.ckks_vector(context, [v]) for v in db_vec]
            enc_dot = ts.ckks_vector(context, [0.0])
            for i in range(dim):
                x_i = query_vector[i]
                partial = fast_repeated_add(encrypted_vector[i], x_i)
                enc_dot += partial
            encrypted_dot_products.append(enc_dot)
            counter += 1
        return f"Processed {counter} vectors"

    print(doing_stuff())
    print(f"Encrypted dot products computed: {len(encrypted_dot_products)}")


if __name__ == "__main__":
    collections = [
        "fma_1000_audio_vectors",
        "fma_1000_audio_vectors_512",
        "fma_1000_audio_vectors_256",
        "fma_1000_audio_vectors_128"
    ]
    for cname in collections:
        ahe_repeated_add(cname)
