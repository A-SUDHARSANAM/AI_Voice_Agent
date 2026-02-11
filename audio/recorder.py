import sounddevice as sd
from scipy.io.wavfile import write

def record_audio(filename, duration=3, fs=16000):
    print("Recording...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    write(filename, fs, recording)
    print("Saved:", filename)