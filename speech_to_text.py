import os
import sys
import logging
import warnings
import librosa
import contextlib
from transformers import pipeline

# 1. Suppress Python warnings
warnings.filterwarnings("ignore")

# 2. Suppress Huggingface's internal logging
logging.getLogger("transformers").setLevel(logging.ERROR)

# 3. Suppress all stderr output (decoder/torch/kenlm messages)
@contextlib.contextmanager
def suppress_stderr():
    devnull = open(os.devnull, 'w')
    old_stderr = sys.stderr
    sys.stderr = devnull
    try:
        yield
    finally:
        sys.stderr = old_stderr
        devnull.close()

def transcribe_audio(audio_path):
    print("Transcribing...")

    # Load audio with librosa
    audio, rate = librosa.load(audio_path, sr=16000)

    # Suppress logs and initialize model
    with suppress_stderr():
        asr = pipeline("automatic-speech-recognition", model="jonatasgrosman/wav2vec2-large-xlsr-53-english")

    # Run transcription (this won't print any warnings now)
    result = asr(audio)
    print("Transcription:", result["text"])

if __name__ == "__main__":
    transcribe_audio("male.wav")
