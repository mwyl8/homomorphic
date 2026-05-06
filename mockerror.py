
import tenseal as ts
import numpy as np

# === CONFIG ===
VECTOR_DIM = 1024
MAX_INT = 100
np.random.seed(42)

# === 1. Create mock database and query vector ===
db_vector = np.random.randint(1, MAX_INT+1, VECTOR_DIM).tolist()
query_vector = np.random.randint(1, MAX_INT+1, VECTOR_DIM).tolist()

# === 2. Base dot product ===
def base_dot(query, db):
    return sum(q*v for q, v in zip(query, db))

base_value = base_dot(query_vector, db_vector)
print(f"Base dot product: {base_value}")

# === 3. Repeated addition for AHE ===
def repeated_addition(val, times):
    result = 0
    for _ in range(times):
        result += val
    return result

# === 4. TenSEAL CKKS context ===
context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=8192,
    coeff_mod_bit_sizes=[60, 40, 40, 60]
)
context.global_scale = 2**40
context.generate_galois_keys()

# === 5. FHE: encrypt both query and database vector ===
enc_query = ts.ckks_vector(context, query_vector)
enc_db = ts.ckks_vector(context, db_vector)
enc_dot = enc_query.dot(enc_db)
fhe_value = enc_dot.decrypt()[0]
fhe_error = abs(fhe_value - base_value) / abs(base_value) * 100
print(f"FHE dot product: {fhe_value}, Percent error: {fhe_error:.12f}%")

# === 6. AHE: encrypt query only, repeated addition ===
enc_query_vals = [ts.ckks_vector(context, [q]) for q in query_vector]
ahe_query_dot = ts.ckks_vector(context, [0.0])
for q_enc, v in zip(enc_query_vals, db_vector):
    ahe_query_dot += repeated_addition(q_enc, v)
ahe_query_value = ahe_query_dot.decrypt()[0]
ahe_query_error = abs(ahe_query_value - base_value) / abs(base_value) * 100
print(f"AHE encrypted query dot product: {ahe_query_value}, Percent error: {ahe_query_error:.12f}%")

# === 7. AHE: encrypt database only, repeated addition ===
enc_db_vals = [ts.ckks_vector(context, [v]) for v in db_vector]
ahe_db_dot = ts.ckks_vector(context, [0.0])
for q, v_enc in zip(query_vector, enc_db_vals):
    ahe_db_dot += repeated_addition(v_enc, q)
ahe_db_value = ahe_db_dot.decrypt()[0]
ahe_db_error = abs(ahe_db_value - base_value) / abs(base_value) * 100
print(f"AHE encrypted database dot product: {ahe_db_value}, Percent error: {ahe_db_error:.12f}%")
