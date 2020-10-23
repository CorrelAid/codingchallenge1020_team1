import numpy as np
import librosa
import random


def time_stretch(audio_sample, low=1.1, high=1.5):

    if isinstance(audio_sample, str) and audio_sample.split(".")[-1] == ".wav":
        audio_sample = librosa.load(audio_sample, sr=48000)

    stretch_factor = np.random.uniform(low, high)

    if random.choice([0, 1]):
        stretch_factor = 1/stretch_factor

    return librosa.effects.time_stretch(audio_sample, stretch_factor)