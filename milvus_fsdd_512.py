import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import librosa
from pymilvus import MilvusClient
import os

# ---- Load YAMNet from TensorFlow Hub ----
yamnet_model_handle = "https://tfhub.dev/google/yamnet/1"
yamnet_model = hub.load(yamnet_model_handle)

# ---- Connect to Milvus ----
client = MilvusClient("milvus_demo.db")

# ---- Create Collection (with 512-dim vectors) ----
collection_name = "fsdd_audio_vector_512"
if client.has_collection(collection_name=collection_name):
    client.drop_collection(collection_name=collection_name)

client.create_collection(
    collection_name=collection_name,
    dimension=512,
)

# ---- Prepare list of audio files ----
folder = "/home/ec2-user/dataset/fsdd_flat"
audio_exts = (".mp3", ".wav", ".flac", ".ogg", ".m4a")
audio_files = sorted(
    [f for f in os.listdir(folder) if f.lower().endswith(audio_exts)]
)

num_files_to_process = min(1000, len(audio_files))

# ---- Process Audio Files ----
for i in range(num_files_to_process):
    file_path = os.path.join(folder, audio_files[i])

    try:
        waveform, sr = librosa.load(file_path, sr=16000, mono=True)
        waveform = waveform.astype(np.float32)

        if waveform.ndim != 1 or waveform.size == 0:
            raise ValueError(f"Invalid waveform shape or empty file: {file_path}")

        scores, embeddings, spectrogram = yamnet_model(waveform)
        global_embedding = tf.reduce_mean(embeddings, axis=0).numpy()
        embedding_512d = global_embedding[::2]

        data = {
            "id": i,
            "vector": embedding_512d.tolist(),
            "text": f"fsdd_{i}",
            "subject": "fsdd",
        }

        client.insert(collection_name=collection_name, data=data)
        print(f"Inserted vector for file {audio_files[i]} with id {i}")

    except Exception as e:
        print(f"Failed to process file {file_path}: {e}")
