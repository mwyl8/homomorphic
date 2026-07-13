from pymilvus import connections, utility, Collection

# 1. Connect to Milvus
connections.connect("default", host="localhost", port="19530")

# 2. Print collections
print("Available collections:", utility.list_collections())

# 3. Try to access audio_vectors
try:
    col = Collection("audio_vectors")
    col.load()
    print(f"Collection 'audio_vectors' has {col.num_entities} vectors.")
except Exception as e:
    print("ERROR when loading collection:", e)

