import librosa
import numpy as np
from audio.recorder import record_audio

def extract_features(file):
    y, sr = librosa.load(file, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfcc.T, axis=0)

def enroll_admin():
    features = []
    
    for i in range(3):
        filename = f"sample_{i}.wav"
        record_audio(filename)
        features.append(extract_features(filename))
    
    admin_voice = np.mean(features, axis=0)
    np.save("data/admin_voice.npy", admin_voice)
    print("Admin voice enrolled successfully.")

if __name__ == "__main__":
    enroll_admin()