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
from pymilvus import MilvusClient

client = MilvusClient("milvus_demo.db")

#Create a Collection

if client.has_collection(collection_name="yamnet_embeddings"):
    client.drop_collection(collection_name="yamnet_embeddings")
client.create_collection(
    collection_name="yamnet_embeddings",
    dimension=102,  # The vectors we will use in this demo has 1024 dimensions
)
res = client.list_collections()
print(res)

res = client.describe_collection(
    collection_name="yamnet_embeddings"
)
print(res)

# ---- Step 4: Insert your embedding vector ----
# Example: random vector simulating YAMNet output

vector = np.random.rand(102).astype("float32").tolist()
print (vector)
data = {"id": 0, "vector": vector, "text": "test", "subject": "history"}
vector = np.random.rand(102).astype("float32").tolist()
data1 = {"id": 1, "vector": vector, "text": "test", "subject": "history"}
vector = np.random.rand(102).astype("float32").tolist()
data2 = {"id": 2, "vector": vector, "text": "test", "subject": "history"}
print (data)
print (data1)
print (data2)

#res = client.insert(collection_name="yamnet_embeddings", data=[[global_embedding]])
res = client.insert(collection_name="yamnet_embeddings", data=data)
print(res)
res = client.insert(collection_name="yamnet_embeddings", data=data1)
print(res)
res = client.insert(collection_name="yamnet_embeddings", data=data2)
print(res)

query_vector=vector

res = client.search(
    collection_name="yamnet_embeddings",
    data=[query_vector],
    limit=5,
)

for hits in res:
    print("TopK results:")
    for hit in hits:
        print(hit) 
