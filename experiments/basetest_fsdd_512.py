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


# === 1. Setup: Fetch vectors from Milvus ===
client = MilvusClient("milvus_demo.db")
collection_name = "fsdd_audio_vector_512"

results = client.query(
    collection_name=collection_name,
    output_fields=["vector"],
    limit=1000,
)

all_vectors = [r["vector"] for r in results]
if not all_vectors:
    raise RuntimeError(f"No vectors found in {collection_name}")

query_vector = all_vectors[0]
database_vectors = all_vectors[1:]
dim = len(query_vector)


# === 2. Core computation wrapped in performance tracker ===
@track_performance
def compute_dot_products():
    dot_products = []
    total = len(database_vectors)
    for idx, db_vec in enumerate(database_vectors, start=1):
        dot = 0.0
        for i in range(dim):
            dot += query_vector[i] * db_vec[i]
        dot_products.append(dot)
        if idx % 25 == 0 or idx == total:
            print(f"Progress: {idx}/{total} dot products computed")
    print(f"\nDot products computed: {len(dot_products)}")


# === 3. Run benchmark ===
compute_dot_products()
