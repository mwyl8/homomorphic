from pymilvus import MilvusClient
import tenseal as ts
from time import perf_counter
import psutil
import os


def fhe_euclidean_collection(client, collection_name, limit=1000):
    print(f"\n=== Processing collection (FHE euclidean): {collection_name} ===")

    try:
        results = client.query(
            collection_name=collection_name,
            output_fields=["vector"],
            limit=limit
        )
    except Exception as e:
        print(f"Failed to query collection {collection_name}: {e}")
        return

    if not results:
        print(f"No vectors found in collection {collection_name}, skipping...")
        return

    vector_dict = {i: r.get("vector") for i, r in enumerate(results)}
    if 0 not in vector_dict or not vector_dict[0]:
        print(f"No valid query vector found in {collection_name}, skipping...")
        return

    query_vector = vector_dict[0]
    dim = len(query_vector)

    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60]
    )
    context.global_scale = 2**40
    context.generate_galois_keys()

    enc_query = [ts.ckks_vector(context, [int(round(x))]) for x in query_vector]

    process = psutil.Process(os.getpid())
    num_cpus = psutil.cpu_count(logical=True)

    total_time_sum = 0.0
    processed = 0

    mem_start = process.memory_info().rss / (1024 * 1024)
    psutil.cpu_percent(interval=None)
    process.cpu_percent(interval=None)
    wall_start = perf_counter()

    enc_l2_scores = []

    for i in range(limit):
        db_vec = vector_dict.get(i)
        if not db_vec or len(db_vec) != dim:
            continue

        mem_before = process.memory_info().rss / (1024 * 1024)
        t0 = perf_counter()

        enc_dist2 = ts.ckks_vector(context, [0.0])
        encrypted_db_vector = [ts.ckks_vector(context, [int(round(v))]) for v in db_vec]

        for j in range(dim):
            diff = enc_query[j] - encrypted_db_vector[j]
            enc_dist2 += diff * diff

        enc_l2_scores.append(enc_dist2)

        t1 = perf_counter()
        mem_after = process.memory_info().rss / (1024 * 1024)

        dt = t1 - t0
        total_time_sum += dt
        processed += 1

        print(f"[{collection_name}] Vector {processed} | Time: {dt:.4f}s | Mem Δ: {mem_after - mem_before:.2f} MB")

    wall_end = perf_counter()
    mem_end = process.memory_info().rss / (1024 * 1024)

    cpu_percent_per_core = process.cpu_percent(interval=None)
    cpu_total_percent = cpu_percent_per_core * num_cpus

    print(f"\n=== Summary for {collection_name} (FHE euclidean) ===")
    print(f"Vectors processed: {processed}")
    print(f"Sum of per-vector runtimes: {total_time_sum:.4f} seconds")
    print(f"Wall-clock runtime: {wall_end - wall_start:.4f} seconds")
    print(f"CPU utilization (total across {num_cpus} cores): {cpu_total_percent:.1f}%")
    print(f"Memory change (start->end): {mem_end - mem_start:.2f} MB")
    print(f"Encrypted squared-euclidean scores computed: {len(enc_l2_scores)}")


def main():
    client = MilvusClient("milvus_demo.db")
    sizes = [1024, 512, 256, 128]

    for size in sizes:
        collection_name = f"audio_vector_{size}"
        fhe_euclidean_collection(client, collection_name, limit=1000)


if __name__ == "__main__":
    main()
