import tenseal as ts
import numpy as np

# === Setup ===
dim = 128
np.random.seed(42)

# Whole number vectors
query_vector = np.random.randint(0, 10, size=dim).tolist()
db_vector = np.random.randint(0, 10, size=dim).tolist()

# Plaintext dot product
plaintext_dot = sum(q * d for q, d in zip(query_vector, db_vector))
print(f"Plaintext dot product: {plaintext_dot}")

# === CKKS context ===
context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=8192,
    coeff_mod_bit_sizes=[40, 20, 40]
)
context.global_scale = 2**40
context.generate_galois_keys()

# Encrypt query vector (simulating AHE additive ops)
enc_query = [ts.ckks_vector(context, [q]) for q in query_vector]

# === Emulate dot product with additive HE ===
encrypted_sum = ts.ckks_vector(context, [0])

for i in range(dim):
    # emulate multiplication via repeated addition
    # (q * d = q added d times)
    temp = ts.ckks_vector(context, [0])
    for _ in range(db_vector[i]):
        temp = temp + enc_query[i]
    encrypted_sum = encrypted_sum + temp

# Decrypt
decrypted_dot = encrypted_sum.decrypt()[0]

print(f"Decrypted (AHE-style) dot product: {decrypted_dot}")
print(f"Absolute error: {abs(plaintext_dot - decrypted_dot)}")
