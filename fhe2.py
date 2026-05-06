from pymilvus import MilvusClient
import tenseal as ts
from time import perf_counter

# === 1. Milvus connection ===
client = MilvusClient("milvus_demo.db")
collection_name = "yamnet_embeddings"

# === 2. Fetch vectors from Milvus ===
results = client.query(
    collection_name=collection_name,
    output_fields=["vector"],
    limit=10
)

# Extract vectors from result
database_vectors = [r["vector"] for r in results]

# === 3. Define or select your query vector ===
# Example: use the first DB vector as query (or make your own)
query_vector = database_vectors[0]

# === 4. Create TenSEAL CKKS context ===
context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=8192,
    coeff_mod_bit_sizes=[60, 40, 40, 60]
)
context.global_scale = 2**40
context.generate_galois_keys()

# === 5. Encrypt the query vector ===
enc_query = ts.ckks_vector(context, query_vector)

# === 6. Compute encrypted dot products ===
enc_dot_products = []
start = perf_counter()

for db_vec in database_vectors:
    enc_db_vec = ts.ckks_vector(context, db_vec)
    enc_product = enc_query * enc_db_vec
    dot = enc_product.sum()
    enc_dot_products.append(dot)

end = perf_counter()

# === 7. Output results ===
# print("Homomorphic dot products with query vector:")
# for i, score in enumerate(enc_dot_products):
#     print(f"Vector {i}: {score:.5f}")

print(f"\nTotal FHE computation time: {end - start:.4f} seconds")
