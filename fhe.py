import tenseal as ts
from datetime import datetime

# Create a context with encryption parameters (CKKS scheme supports FHE for real numbers)
context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=8192,
    coeff_mod_bit_sizes=[60, 40, 40, 60]
)
context.generate_galois_keys()
context.global_scale = 2**40

# Encrypt two values
vec1 = [1.5, 2.0, 3.5]
vec2 = [2.0, 1.0, 4.5]




start =datetime.now()
enc_vec1 = ts.ckks_vector(context, vec1)
enc_vec2 = ts.ckks_vector(context, vec2)
enc_product = enc_vec1 * enc_vec2
end = datetime.now() 
print('Duration for mul: {}'.format(end - start))

start =datetime.now()
enc_vec1 = ts.ckks_vector(context, vec1)
enc_vec1 = enc_vec1+enc_vec1 
for i in range(0,3):
    enc_vec1 = enc_vec1+enc_vec1 
end = datetime.now() 
end = datetime.now() 
print('Duration for plus: {}'.format(end - start))

# Homomorphic addition and multiplication
start =datetime.now()
for i in range(0, 1000):
    enc_sum = enc_vec1 + enc_vec2
end = datetime.now() 
print('Duration for plus: {}'.format(end - start))


start=datetime.now()
for i in range(0, 1000):
    enc_product = enc_vec1 * enc_vec2
end=datetime.now()
print('Duration for mul: {}'.format(end - start))



# Decrypt results
dec_sum = enc_sum.decrypt()
dec_product = enc_product.decrypt()

print("Original vectors:")
print("vec1:", vec1)
print("vec2:", vec2)
print("\nDecrypted sum:", dec_sum)
print("Decrypted product:", dec_product)
