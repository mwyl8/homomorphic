from phe import paillier

# Key generation
public_key, private_key = paillier.generate_paillier_keypair()

# Messages to encrypt
m1 = 4
m2 = 3

# Encrypt the messages
c1 = public_key.encrypt(m1)
c2 = public_key.encrypt(m2)

# Homomorphic addition (on ciphertexts)
c_sum = c1 + c2  # This performs homomorphic addition

# Decrypt the result
decrypted_sum = private_key.decrypt(c_sum)

# Output
print(f"Original messages: {m1}, {m2}")
print(f"Decrypted sum: {decrypted_sum}")
