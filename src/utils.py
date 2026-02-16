import numpy as np

# IMPORTANT: must match training folder order (alphabetical)
emotion_labels = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]

def preprocess_face(face):
    face = face / 255.0
    face = np.reshape(face, (1,48,48,1))
    return face
