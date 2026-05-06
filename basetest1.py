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

# === Plaintext Dot Product Logic ===

def plaintext_dot_product():
    # ---- Connect to Milvus ----
    client = MilvusClient("milvus_demo.db")
    collection_name = "yamnet_embeddings"

    # ---- Query vectors ----
    results = client.query(
        collection_name=collection_name,
        output_fields=["vector"],
        limit=6  
    )

    all_vectors = [r["vector"] for r in results]
    query_vector = all_vectors[0]
    database_vectors = all_vectors[1:]
    dim = len(query_vector)

    dot_products = []
    @track_performance
    def repeated_prod():
        for db_vec in database_vectors:
            dot = 0.0
            for i in range(dim):
                dot += query_vector[i] * db_vec[i]
            dot_products.append(dot)
    repeated_prod()
    print(f"\nDot products computed: {len(dot_products)}")
    # Uncomment to print actual dot values:
    # print("Dot products:", dot_products)

# === Run the benchmark ===
plaintext_dot_product()
