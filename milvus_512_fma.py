import os
import numpy as np
import librosa
import tensorflow as tf
import tensorflow_hub as hub
from pymilvus import MilvusClient

# 1. Load YAMNet model from TensorFlow Hub
yamnet_model_handle = "https://tfhub.dev/google/yamnet/1"
yamnet_model = hub.load(yamnet_model_handle)

# 2. Connect to Milvus
client = MilvusClient("milvus_demo.db")

# 3. Create collection for 512-dimensional embeddings
collection_name = "fma_1000_audio_vectors_512"

if client.has_collection(collection_name=collection_name):
    client.drop_collection(collection_name=collection_name)

client.create_collection(
    collection_name=collection_name,
    dimension=512
)

# 4. Prepare list of audio files
folder = "/home/ec2-user/datasets/fma_1000"
mp3_files = sorted([f for f in os.listdir(folder) if f.endswith(".mp3")])

num_files_to_process = min(1000, len(mp3_files))

# 5. Process audio files and insert embeddings into Milvus
for i in range(num_files_to_process):
    file_path = os.path.join(folder, mp3_files[i])
    try:
        # Load audio
        waveform, sr = librosa.load(file_path, sr=16000, mono=True)
        waveform = waveform.astype(np.float32)

        if waveform.ndim != 1 or waveform.size == 0:
            raise ValueError(f"Invalid waveform shape or empty file: {file_path}")

        # Run YAMNet inference
        scores, embeddings, spectrogram = yamnet_model(waveform)
        global_embedding = tf.reduce_mean(embeddings, axis=0).numpy()  # 1024-dim

        # Downsample to 512-dim
        embedding_512d = global_embedding[::2]  # take every other element

        # Prepare data for Milvus
        data = {
            "id": i,
            "vector": embedding_512d.tolist(),
            "text": f"fma_{i}",
            "subject": "fma"
        }

        # Insert into Milvus
        res = client.insert(collection_name=collection_name, data=data)
        print(f"Inserted 512-dim vector for file {mp3_files[i]} with id {i}")

    except Exception as e:
        print(f"Failed to process file {file_path}: {e}")
