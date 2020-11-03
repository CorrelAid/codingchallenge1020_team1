import numpy as np
import librosa
import random
import sys
try:
    from auditok import split
except:
    e = sys.exc_info()[0]
    print("Couldn't import auditok. You can still use all data augmentation methods, but not the 'augment_sample'"
          "function. \n Error message: {}".format(e))


def augment_sample(audio_sample_path, target_seconds=3, stretch_low=1.1, stretch_high=1.5, pitch_low=1.5, pitch_high=4,
                   noise_low=0.001, noise_high=0.01, sr=48000):

    target_len = target_seconds * sr
    audio_regions = split(audio_sample_path, sampling_rate=sr)

    audio_regions_arrays = []
    audio_regions_info = []

    for audio_region in audio_regions:
        region_ar = np.asarray(audio_region)
        audio_regions_arrays.append(region_ar)
        audio_regions_info.append((audio_region.meta.start, audio_region.meta.end))

    audio_regions_arrays, audio_regions_info = merge_regions(audio_regions_arrays, audio_regions_info, sr=sr)

    total_len = np.sum([max(a.shape) for a in audio_regions_arrays])
    max_stretch = int(100 * target_len/total_len)/100

    stretch_high = min(max_stretch, stretch_high)

    audio_regions_arrays = [time_stretch(ar, stretch_low, stretch_high, sr=sr) for ar in audio_regions_arrays]
    audio_regions_arrays = [change_pitch(ar, pitch_low, pitch_high, sr=sr) for ar in audio_regions_arrays]

    total_len_after_stretch = np.sum([max(a.shape) for a in audio_regions_arrays])
    n_missing = int(target_len - total_len_after_stretch)
    n_gaps = len(audio_regions_arrays) - 1

    filler_distribution = np.random.uniform(0, 1, 2+n_gaps)
    filler_distribution = filler_distribution/np.sum(filler_distribution)
    filler_distribution_ints = [int(i * n_missing) for i in filler_distribution]
    filler_distribution_ints[-1] = int(n_missing - np.sum(filler_distribution_ints[:-1]))

    noise_at_start = noise_period(filler_distribution_ints[0], 0.001, 0.01)
    noise_at_end = noise_period(filler_distribution_ints[-1], 0.001, 0.01)

    ar = audio_regions_arrays[0]

    for i, audio_region_ar in enumerate(audio_regions_arrays[1:]):
        noise_filler = noise_period(filler_distribution_ints[i+1], 0.001, 0.01)
        ar = np.concatenate([ar, noise_filler, audio_region_ar])

    final_ar = np.concatenate([noise_at_start, ar, noise_at_end])
    final_ar = noise_injection(final_ar, noise_low, noise_high, sr=sr)

    if len(final_ar) > target_len:
        final_ar = final_ar[:target_len]

    if len(final_ar) < target_len:
        missing_n = target_len - len(final_ar)
        final_ar = np.concatenate([final_ar, noise_period(missing_n, 0.001, 0.01)])

    return final_ar


def merge_regions(audio_regions_arrays, audio_regions_info, time_threshold=0.3, sr=48000):

    flag = False

    if len(audio_regions_arrays) == 1:
        return audio_regions_arrays, audio_regions_info

    for i, audio_region_array in enumerate(audio_regions_arrays[:-1]):

        time_between_regions = audio_regions_info[i+1][0] - audio_regions_info[i][1]
        if time_between_regions <= time_threshold:
            len_between = int(time_between_regions * sr)
            noise_filler = noise_period(len_between)
            new_region = np.concatenate([audio_region_array, noise_filler, audio_regions_arrays[i+1]])
            new_info = (audio_regions_info[i][0], audio_regions_info[i+1][1])

            del audio_regions_arrays[i+1]
            del audio_regions_info[i+1]

            audio_regions_arrays[i] = new_region
            audio_regions_info[i] = new_info
            flag = True
            break

    if flag:
        return merge_regions(audio_regions_arrays, audio_regions_info, time_threshold, sr)

    return audio_regions_arrays, audio_regions_info


def noise_period(noise_len, low=0.001, high=0.01):

    noise_level = np.random.uniform(low, high)
    return np.random.normal(0, noise_level, noise_len)


def time_stretch(audio_sample, low=1.1, high=1.5, sr=48000):

    if isinstance(audio_sample, str) and audio_sample.split(".")[-1] == ".wav":
        audio_sample = librosa.load(audio_sample, sr=sr)

    stretch_factor = np.random.uniform(low, high)

    if random.choice([0, 1]):
        stretch_factor = 1/stretch_factor

    return librosa.effects.time_stretch(audio_sample, stretch_factor)


def change_pitch(audio_sample, low=1.5, high=4, sr=48000):

    if isinstance(audio_sample, str) and audio_sample.split(".")[-1] == ".wav":
        audio_sample = librosa.load(audio_sample, sr=sr)

    pitch_factor = np.random.uniform(low, high) * random.choice([1, -1])

    return librosa.effects.pitch_shift(audio_sample, sr=sr, n_steps=pitch_factor)


def noise_injection(audio_sample, low=0.001, high=0.01, sr=48000):

    if isinstance(audio_sample, str) and audio_sample.split(".")[-1] == ".wav":
        audio_sample = librosa.load(audio_sample, sr=sr)

    noise_level = np.random.uniform(low, high)
    noise = np.random.normal(0, noise_level, len(audio_sample))

    return audio_sample + noise
