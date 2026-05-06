import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import librosa
from pymilvus import MilvusClient
import os

# ---- Load YAMNet from TensorFlow Hub ----
yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
yamnet_model = hub.load(yamnet_model_handle)

# ---- Connect to Milvus ----
client = MilvusClient("milvus_demo.db")

# ---- Create Collection (with 1024-dim vectors) ----
collection_name = "yamnet_embeddings"
if client.has_collection(collection_name=collection_name):
    client.drop_collection(collection_name=collection_name)

client.create_collection(
    collection_name=collection_name,
    dimension=1024,  # Still using 1024 dimensions like your original setup
)
# ---- Prepare list of audio files ----
folder = "/home/ec2-user/audio_files"  # change this to your audio files folder
mp3_files = sorted([f for f in os.listdir(folder) if f.endswith(".mp3")])

# Only process up to 10 files or fewer if less available
num_files_to_process = min(1000, len(mp3_files))

# ---- Process Audio Files ----
for i in range(num_files_to_process):
    file_path = os.path.join(folder, mp3_files[i])  # Define file_path here

    try:
        waveform, sr = librosa.load(file_path, sr=16000, mono=True)
        waveform = waveform.astype(np.float32)

        if waveform.ndim != 1 or waveform.size == 0:
            raise ValueError(f"Invalid waveform shape or empty file: {file_path}")

        # Run inference
        scores, embeddings, spectrogram = yamnet_model(waveform)
        global_embedding = tf.reduce_mean(embeddings, axis=0).numpy()
        # Downsample 1024-dim embedding to 102-dim by selecting every 10th value (or adjust as needed)
        embedding_1024d = global_embedding

        data = {
            "id": i,
            "vector": embedding_1024d.tolist(),
            "text": f"test_{i}",
            "subject": "history"
        }

        res = client.insert(collection_name=collection_name, data=data)
        print(f"Inserted vector for file {mp3_files[i]} with id {i}")

    except Exception as e:
        print(f"Failed to process file {file_path}: {e}")

