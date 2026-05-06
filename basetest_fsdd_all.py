from pymilvus import MilvusClient
from time import perf_counter
import psutil
import os


def basetest_collection(client, collection_name, limit=1000):
    print(f"\n=== Running base test for collection: {collection_name} ===")

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

    if len(all_vectors) < limit:
        print(f"Warning: Only {len(all_vectors)} vectors available.")

    if not all_vectors:
        print(f"No vectors found in {collection_name}")
        return

    # --- INCLUDE QUERY VS ITSELF ---
    query_vector = all_vectors[0]
    database_vectors = all_vectors  # include self

    dim = len(query_vector)

    process = psutil.Process(os.getpid())
    num_cpus = psutil.cpu_count(logical=True)

    # Collection-level tracking
    cpu_start = process.cpu_times()
    mem_start = process.memory_info().rss / (1024 * 1024)
    psutil.cpu_percent(interval=None)
    process.cpu_percent(interval=None)
    wall_start = perf_counter()

    per_vector_time_sum = 0.0
    dot_products = []
    processed = 0
    total = len(database_vectors)

    for idx, db_vec in enumerate(database_vectors, start=1):
        if db_vec is None or len(db_vec) != dim:
            continue

        mem_before = process.memory_info().rss / (1024 * 1024)
        t0 = perf_counter()

        dot = 0.0
        for i in range(dim):
            dot += query_vector[i] * db_vec[i]

        t1 = perf_counter()
        mem_after = process.memory_info().rss / (1024 * 1024)

        dt = t1 - t0
        per_vector_time_sum += dt
        dot_products.append(dot)
        processed += 1

        print(f"[{collection_name}] Vector {processed}/{total} | "
              f"Time: {dt:.6f}s | Mem Δ: {mem_after - mem_before:.2f} MB")

    wall_end = perf_counter()
    mem_end = process.memory_info().rss / (1024 * 1024)
    cpu_end = process.cpu_times()

    cpu_percent_per_core = process.cpu_percent(interval=None)
    cpu_total_percent = cpu_percent_per_core * num_cpus

    print(f"\n=== Summary for {collection_name} ===")
    print(f"Dot products computed: {len(dot_products)}")
    print(f"Sum of per-vector runtimes: {per_vector_time_sum:.6f} seconds")
    print(f"Wall-clock runtime: {wall_end - wall_start:.6f} seconds")
    print(f"CPU time (user): {cpu_end.user - cpu_start.user:.2f}s")
    print(f"CPU time (system): {cpu_end.system - cpu_start.system:.2f}s")
    print(f"CPU utilization (total across {num_cpus} cores): {cpu_total_percent:.1f}%")
    print(f"Memory change (start->end): {mem_end - mem_start:.2f} MB")


def main():
    client = MilvusClient("milvus_demo.db")

    collections = [
        "fsdd_audio_vector_1024",
        "fsdd_audio_vector_512",
        "fsdd_audio_vector_256",
        "fsdd_audio_vector_128",
    ]

    for cname in collections:
        basetest_collection(client, cname, limit=1000)

    print("\nAll FSDD base tests completed.")


if __name__ == "__main__":
    main()
