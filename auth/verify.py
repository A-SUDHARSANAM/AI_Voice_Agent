import numpy as np
import librosa
import os
from sklearn.metrics.pairwise import cosine_similarity
from audio.recorder import record_audio

SAMPLE_RATE = 16000
THRESHOLD = 0.75   # You can tune after testing


def extract_features(file_path):
    """
    Extract MFCC + delta + delta2 features (frame-level)
    """
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    combined = np.vstack([mfcc, delta, delta2])

    return combined.T  # shape: (frames, features)


def verify_admin():
    """
    Record voice and verify against enrolled admin fingerprint
    """

    if not os.path.exists("data/admin_voice.npy"):
        print("❌ Admin voice not enrolled yet.")
        print("Run enrollment first.")
        return False

    print("Recording for verification...")

    test_file = "temp_test.wav"
    record_audio(test_file)

    admin_voice = np.load("data/admin_voice.npy")
    test_features = extract_features(test_file)

    # Delete temporary file
    os.remove(test_file)

    # Compute similarity matrix (frame-level)
    similarity_matrix = cosine_similarity(test_features, admin_voice)

    # For each test frame, find best matching enrolled frame
    similarity = np.mean(np.max(similarity_matrix, axis=1))

    print(f"Similarity Score: {similarity:.4f}")

    if similarity >= THRESHOLD:
        print("✅ Admin Verified")
        return True
    else:
        print("❌ Access Denied")
        return False