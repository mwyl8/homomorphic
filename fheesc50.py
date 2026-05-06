from pymilvus import MilvusClient
import tenseal as ts
from time import perf_counter
import psutil

# === Performance Tracker ===
def track_performance(fn):
    def wrapper(*args, **kwargs):
        process = psutil.Process()
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

# === Main FHE encrypted-query function ===
def fhe_query_encrypted_all_sizes():
    sizes = [1024, 512, 256, 128]
    client = MilvusClient("milvus_demo.db")

    for size in sizes:
        collection_name = f"esc50_audio_vectors_{size}"
        print(f"\n=== Processing collection: {collection_name} ===")
        try:
            results = client.query(
                collection_name=collection_name,
                output_fields=["vector"],
                limit=1000
            )
        except Exception as e:
            print(f"Failed to query collection {collection_name}: {e}")
            continue

        if not results:
            print(f"No vectors found in collection {collection_name}, skipping...")
            continue

        # Map from ID to vector to allow skipping missing IDs
        vector_dict = {i: r.get("vector") for i, r in enumerate(results)}
        if 0 not in vector_dict or not vector_dict[0]:
            print(f"No valid query vector found in {collection_name}, skipping...")
            continue

        query_vector = vector_dict[0]
        dim = len(query_vector)

        # TenSEAL CKKS context
        context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 40, 40, 60]
        )
        context.global_scale = 2**40
        context.generate_galois_keys()

        # Encrypt query vector element-wise
        enc_query = [ts.ckks_vector(context, [int(round(x))]) for x in query_vector]

        @track_performance
        def encrypted_dot_products():
            enc_dot_products = []
            counter = 0

            for i in range(1000):  # IDs 0 to 999
                db_vec = vector_dict.get(i)
                if not db_vec or len(db_vec) != dim:
                    continue  # skip missing or invalid vectors

                enc_dot = ts.ckks_vector(context, [0.0])
                encrypted_db_vector = [ts.ckks_vector(context, [int(round(v))]) for v in db_vec]

                for j in range(dim):
                    enc_dot += enc_query[j] * encrypted_db_vector[j]

                enc_dot_products.append(enc_dot)
                counter += 1

            print(f"Encrypted dot products computed: {len(enc_dot_products)}")
            return enc_dot_products

        encrypted_dot_products()

# === Run for all 4 sizes ===
fhe_query_encrypted_all_sizes()
