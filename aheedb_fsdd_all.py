import tenseal as ts
from pymilvus import MilvusClient
from time import perf_counter
import psutil
import os


def ahe_repeated_add(collection_name, limit=1000, scale_int=100):
    print(f"\n=== Running encrypted-database AHE for collection: {collection_name} ===")

    process = psutil.Process(os.getpid())
    num_cpus = psutil.cpu_count(logical=True)
    client = MilvusClient("milvus_demo.db")

    if not client.has_collection(collection_name=collection_name):
        print(f"Collection {collection_name} not found, skipping.")
        return

    try:
        results = client.query(
            collection_name=collection_name,
            output_fields=["vector"],
            limit=limit
        )
    except Exception as e:
        print(f"Failed to query {collection_name}: {e}")
        return

    all_vectors = [r.get("vector") for r in results if r.get("vector") is not None]
    if len(all_vectors) < 1:
        print("No vectors found.")
        return

    dim = len(all_vectors[0])

    # query = first vector, DB = all vectors (includes query vs itself)
    query_vector = [int(round(v * scale_int)) for v in all_vectors[0]]
    db_vectors = [[int(round(val * scale_int)) for val in vec] for vec in all_vectors]

    # TenSEAL context
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60]
    )
    context.global_scale = 2**40
    context.generate_galois_keys()

    def fast_repeated_add(enc_val, times):
        if times <= 0:
            return ts.ckks_vector(context, [0.0])
        if times == 1:
            return enc_val

        result = None
        current = enc_val
        while times > 0:
            if times & 1:
                result = current if result is None else result + current
            current = current + current
            times >>= 1
        return result

    total_vectors = min(limit, len(db_vectors))
    encrypted_dot_products = []

    # ---- collection-level tracking (like your FHE summary) ----
    cpu_start = process.cpu_times()
    mem_start = process.memory_info().rss / (1024 * 1024)
    psutil.cpu_percent(interval=None)
    process.cpu_percent(interval=None)
    wall_start = perf_counter()

    sum_per_vector_time = 0.0
    processed = 0

    for idx in range(total_vectors):
        db_vec = db_vectors[idx]
        if db_vec is None or len(db_vec) != dim:
            continue

        mem_before = process.memory_info().rss / (1024 * 1024)
        t0 = perf_counter()

        # encrypt ONE db vector at a time (keeps memory sane)
        encrypted_vector = [ts.ckks_vector(context, [v]) for v in db_vec]

        enc_dot = ts.ckks_vector(context, [0.0])

        for i in range(dim):
            x_i = query_vector[i]
            if x_i == 0:
                continue
            partial = fast_repeated_add(encrypted_vector[i], x_i)
            enc_dot += partial

        encrypted_dot_products.append(enc_dot)

        t1 = perf_counter()
        mem_after = process.memory_info().rss / (1024 * 1024)

        dt = t1 - t0
        sum_per_vector_time += dt
        processed += 1

        print(f"[{collection_name}] Vector {processed}/{total_vectors} | "
              f"Time: {dt:.4f}s | Mem Δ: {mem_after - mem_before:.2f} MB")

    wall_end = perf_counter()
    mem_end = process.memory_info().rss / (1024 * 1024)
    cpu_end = process.cpu_times()

    cpu_percent_per_core = process.cpu_percent(interval=None)
    cpu_total_percent = cpu_percent_per_core * num_cpus

    print(f"\n=== Summary for {collection_name} ===")
    print(f"Vectors processed: {processed}")
    print(f"Encrypted dot products computed: {len(encrypted_dot_products)}")
    print(f"Sum of per-vector runtimes: {sum_per_vector_time:.4f} seconds")
    print(f"Wall-clock runtime: {wall_end - wall_start:.4f} seconds")
    print(f"CPU time (user): {cpu_end.user - cpu_start.user:.2f}s")
    print(f"CPU time (system): {cpu_end.system - cpu_start.system:.2f}s")
    print(f"CPU utilization (total across {num_cpus} cores): {cpu_total_percent:.1f}%")
    print(f"Memory change (start->end): {mem_end - mem_start:.2f} MB")


if __name__ == "__main__":
    collections = [
        "fsdd_audio_vector_1024",
        "fsdd_audio_vector_512",
        "fsdd_audio_vector_256",
        "fsdd_audio_vector_128",
    ]

    for cname in collections:
        ahe_repeated_add(cname, limit=1000, scale_int=100)

    print("\nAll FSDD AHE-EDB dimensions completed.")
