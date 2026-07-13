import psutil
import os
from pymilvus import MilvusClient
from time import perf_counter

# === Performance Tracker (CPU %) ===
def track_performance(fn):
    def wrapper(*args, **kwargs):
        process = psutil.Process(os.getpid())
        num_cpus = psutil.cpu_count(logical=True)

        mem_before = process.memory_info().rss / (1024 * 1024)  # MB

        # reset counters
        psutil.cpu_percent(interval=None)
        process.cpu_percent(interval=None)

        wall_start = perf_counter()
        result = fn(*args, **kwargs)
        wall_end = perf_counter()

        # average CPU usage for this process (per core)
        cpu_percent_per_core = process.cpu_percent(interval=None)
        # convert to total CPU usage across all cores
        cpu_total_percent = cpu_percent_per_core * num_cpus

        mem_after = process.memory_info().rss / (1024 * 1024)

        print("\n--- Performance Report ---")
        print(f"Wall-clock time: {wall_end - wall_start:.4f} seconds")
        print(f"CPU utilization (total across {num_cpus} cores): {cpu_total_percent:.1f}%")
        print(f"Memory delta: {mem_after - mem_before:.2f} MB")
        print("--------------------------")

        return result
    return wrapper


# === Main Baseline Logic ===
@track_performance
def plaintext_sum():

    # ---- Connect to Milvus ----
    client = MilvusClient("milvus_demo.db")
    collection_name = "yamnet_embeddings"

    # ---- Fetch vectors from Milvus ----
    results = client.query(
        collection_name=collection_name,
        output_fields=["vector"],
        limit=10
    )

    # ---- Extract vectors ----
    all_vectors = [r["vector"] for r in results]
    query_vector = all_vectors[0]             # Use the first as query
    database_vectors = all_vectors[1:]        # Use the rest as DB

    # ---- Totals list ----
    totals = []

    # ---- Start computation timing ----
    start = perf_counter()

    for db_vec in database_vectors:
        z = 0
        for a in range(len(db_vec)):
            z += db_vec[a] + query_vector[a]
        totals.append(z)

    end = perf_counter()

    # ---- Output ----
    print("Duration (inner loop only):", end - start)
    print("Totals:", totals)


# === Run it ===
if __name__ == "__main__":
    plaintext_sum()
