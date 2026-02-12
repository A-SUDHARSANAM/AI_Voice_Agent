import librosa
import numpy as np
import os
from audio.recorder import record_audio


SAMPLE_RATE = 16000
NUM_SAMPLES = 3


def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    combined = np.vstack([mfcc, delta, delta2])

    return combined.T  # Return frame-level features


def enroll_admin():
    """
    Records multiple voice samples and creates admin fingerprint
    """

    # Ensure data directory exists
    if not os.path.exists("data"):
        os.makedirs("data")

    features = []

    print("=== ADMIN VOICE ENROLLMENT ===")
    print(f"You will record {NUM_SAMPLES} samples.")
    print("--------------------------------")

    for i in range(NUM_SAMPLES):
        input(f"\nPress ENTER to record sample {i+1}...")

        filename = f"temp_sample_{i}.wav"
        record_audio(filename)

        feature = extract_features(filename)
        features.append(feature)

        # Delete temporary audio
        os.remove(filename)

        print(f"Sample {i+1} recorded successfully.")

    # Create average fingerprint
    admin_voice = np.vstack(features)
    np.save("data/admin_voice.npy", admin_voice)

    print("\n✅ Enrollment completed successfully.")
    print("Voice fingerprint saved in data/admin_voice.npy")


if __name__ == "__main__":
    enroll_admin()