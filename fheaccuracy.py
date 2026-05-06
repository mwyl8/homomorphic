import tenseal as ts
import numpy as np

dims = [128, 256, 512, 1024]

for dim in dims:
    print(f"\n=== Dimension {dim} ===")

    # --- 1. Create own vectors ---
    query_vector = np.random.randint(0, 10, size=dim).astype(float)
    db_vector = np.random.randint(0, 10, size=dim).astype(float)

    # --- 2. Plain dot product ---
    plain_dot = np.dot(query_vector, db_vector)

    # --- 3. CKKS context ---
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60]
    )
    context.global_scale = 2**40
    context.generate_galois_keys()

    # --- 4. Encrypt vectors ---
    enc_query = ts.ckks_vector(context, query_vector)
    enc_db = ts.ckks_vector(context, db_vector)

    # --- 5. FHE dot product ---
    enc_prod = enc_query * enc_db
    fhe_dot = sum(enc_prod.decrypt())  # decrypt -> vector, sum all elements

    # --- 6. Display comparison ---
    print(f"Plain dot product:     {plain_dot:.6f}")
    print(f"Decrypted FHE dot:    {fhe_dot:.6f}")
    print(f"Absolute error:       {abs(plain_dot - fhe_dot):.6f}")
