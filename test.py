import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import librosa
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
import time

# Load YAMNet from TensorFlow Hub
yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
yamnet_model = hub.load(yamnet_model_handle)

# Load your audio file
waveform, sr = librosa.load('./1.mp3', sr=16000, mono=True)
waveform = waveform.astype(np.float32)

# Run inference
scores, embeddings, spectrogram = yamnet_model(waveform)

# embeddings: shape [time_frames, 1024]
# average to get one global embedding for the entire clip:
global_embedding = tf.reduce_mean(embeddings, axis=0).numpy()
# Prevent NumPy from truncating array output
np.set_printoptions(threshold=np.inf)

print(global_embedding)

# try milvus
# ---- Step 1: Connect to Milvus ----
connections.connect(alias="default", host="localhost", port="19530")

# ---- Step 2: Define collection schema ----
collection_name = "yamnet_embeddings"

fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024)
]
schema = CollectionSchema(fields, description="YAMNet embedding storage")

# ---- Step 3: Create collection (drop if exists) ----
if Collection.exists(collection_name):
    Collection(collection_name).drop()
collection = Collection(name=collection_name, schema=schema)

# ---- Step 4: Insert your embedding vector ----
# Example: random vector simulating YAMNet output
vector = np.random.rand(1024).astype("float32").tolist()

collection.insert([[vector]])

# ---- Step 5: Load collection and search ----
collection.load()

# Search by cosine similarity (Milvus uses inner product; normalize vectors)
query_vector = np.array(vector)
query_vector = query_vector / np.linalg.norm(query_vector)
query_vector = query_vector.tolist()

results = collection.search(
    data=[query_vector],
    anns_field="embedding",
    param={"metric_type": "IP", "params": {"nprobe": 10}},
    limit=1,
    output_fields=["id"]
)

# ---- Step 6: Print result ----
for hits in results:
    for hit in hits:
        print(f"Matched ID: {hit.id}, Distance: {hit.distance}")
