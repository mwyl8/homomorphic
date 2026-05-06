import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import librosa

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

