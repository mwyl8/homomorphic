import faiss
import numpy as np

# Step 1: Create some sample vectors (e.g., 5 vectors, 4 dimensions each)
dimension = 4
vectors = np.random.random((5, dimension)).astype('float32')

# Step 2: Create a FAISS index (L2 = Euclidean distance)
index = faiss.IndexFlatL2(dimension)

# Step 3: Add vectors to the index
index.add(vectors)

# Step 4: Create a query vector
query = np.random.random((1, dimension)).astype('float32')

# Step 5: Search for the 3 most similar vectors
k = 3
distances, indices = index.search(query, k)

# Output results
print("Query vector:")
print(query)
print("\nTop 3 nearest vectors:")
print("Indices:", indices)
print("Distances:", distances)
