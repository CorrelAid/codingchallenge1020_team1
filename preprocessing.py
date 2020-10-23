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


def change_pitch(audio_sample, low=1.5, high=4):

    if isinstance(audio_sample, str) and audio_sample.split(".")[-1] == ".wav":
        audio_sample = librosa.load(audio_sample, sr=48000)

    pitch_factor = np.random.uniform(low, high) * random.choice([1, -1])

    return librosa.effects.pitch_shift(audio_sample, sr=48000, n_steps=pitch_factor)


def noise_injection(audio_sample, low=0.001, high=0.01):

    if isinstance(audio_sample, str) and audio_sample.split(".")[-1] == ".wav":
        audio_sample = librosa.load(audio_sample, sr=48000)

    noise_level = np.random.uniform(low, high)
    noise = np.random.normal(0, noise_level, len(audio_sample))

    return audio_sample + noise