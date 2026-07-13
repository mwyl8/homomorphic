from phe import paillier
from time import perf_counter

# Key generation
public_key, private_key = paillier.generate_paillier_keypair()

# Messages to encrypt
m1 = 4
m2 = 3

# Encrypt the messages
c1 = public_key.encrypt(m1)
c2 = public_key.encrypt(m2)

start = perf_counter()
# Homomorphic addition (on ciphertexts)
c_sum = c1 + c2  # This performs homomorphic addition
end = perf_counter()
print('Duration for add: {}'.format(end - start))

# Decrypt the result
decrypted_sum = private_key.decrypt(c_sum)

# Output
print(f"Original messages: {m1}, {m2}")
print(f"Decrypted sum: {decrypted_sum}")

