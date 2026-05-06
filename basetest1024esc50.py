from pymilvus import MilvusClient
from time import perf_counter
import psutil
import os

def track_performance(fn):
    def wrapper(*args, **kwargs):
        process = psutil.Process(os.getpid())
        cpu_start = process.cpu_times()
        mem_before = process.memory_info().rss / (1024 * 1024)
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

client = MilvusClient("milvus_demo.db")
collection_name = "esc50_audio_vectors_1024"

results = client.query(
    collection_name=collection_name,
    output_fields=["vector"],
    limit=1000
)

if len(results) < 2:
    raise RuntimeError("Not enough vectors available in 1024-dim collection.")

all_vectors = [r["vector"] for r in results if "vector" in r]
query_vector = all_vectors[0]
database_vectors = all_vectors[1:]
dim = len(query_vector)

@track_performance
def compute_dot_products():
    dot_products = []
    for db_vec in database_vectors:
        if len(db_vec) != dim:
            continue
        dot = sum(query_vector[i] * db_vec[i] for i in range(dim))
        dot_products.append(dot)
    print(f"\nDot products computed: {len(dot_products)}")

compute_dot_products()
