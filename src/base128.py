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

# === 1. Setup: Fetch vectors from Milvus ===
client = MilvusClient("milvus_demo.db")
collection_name = "audio_vector_128"

results = client.query(
    collection_name=collection_name,
    output_fields=["vector"],
    limit=1000  # Fetch more than 1 so we can use one as query
)

all_vectors = [r["vector"] for r in results]
query_vector = all_vectors[0]
database_vectors = all_vectors
dim = len(query_vector)

# === 2. Core computation wrapped in performance tracker ===
@track_performance
def compute_dot_products():
    dot_products = []
    for db_vec in database_vectors:
        dot = 0.0
        for i in range(dim):
            dot += query_vector[i] * db_vec[i]
        dot_products.append(dot)
    print(f"\nDot products computed: {len(dot_products)}")

# === 3. Run benchmark ===
compute_dot_products()
