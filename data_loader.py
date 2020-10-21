import numpy as np
import librosa
import librosa.feature
import json
import os
import csv


def correct_additional_utterances(path, subfolder_name="corrected"):
    ''' A function to convert the additional utterances of the Zindi-Challenge to proper WAV files

    The function uses FFMPEG to convert the additional given .wav files (which all actually use OGG
    codec) to proper .wav files to avoid handling problems. It assumes the original given folder
    structure in which every class has a separate folder with some .wav files in there. It will create
    a new subfolder and store the converted versions (with the same name as the original) in there.

    :param path: Path to the directory in which the class directories are found
    :param subfolder_name: Name of the newly created subfolder for the new converted audio
    '''

    # Create list of all class folders
    class_folders = os.listdir(path)

    # Catch some pathing mistakes (though not all)
    if len(class_folders) == 0:
        print("The given directory is empty.")
        return

    elif len(class_folders[0].split(".")) != 1:
        print("The first entry in given directory is not a folder. You should"
              "give the path to the directory in which the class-subfolders are.")
        return

    for class_folder in class_folders:

        # Path to distinctive class subfolders
        class_path = os.path.join(path, class_folder)
        # Path to new subfolder for converted audio files
        path_new_subfolder = os.path.join(class_path, subfolder_name)

        # Create directory if not existing yet
        if not os.path.isdir(path_new_subfolder):
            os.mkdir(path_new_subfolder)

        # List of all audio files in respective class directory
        audio_files = os.listdir(class_path)

        for audio_file_name in audio_files:
            audio_file_path = os.path.join(class_path, audio_file_name)
            corrected_audio_file_path = os.path.join(path_new_subfolder, audio_file_name)

            # Run FFMPEG command on CMD to convert audio file to target location (in new subfolder)
            command_to_run = "ffmpeg -i {} {}".format(audio_file_path, corrected_audio_file_path)
            os.system(command_to_run)


def load_original_utterances(path_original="data\\audio_files",
                             train_csv_path="Train.csv", label_to_num="label_to_num.txt",
                             sample_rate=44100, energy_threshold=0.18):
    """Function to load and return all original utterances

    :param path_original: Path to directory where .wav files are located
    :param train_csv_path: Path to the Train.csv file
    :param label_to_num: Path to the label_to_num.txt file or the already loaded dict
    :param sample_rate: sample rate for loading audio data
    :param energy_threshold: Energy threshold under which an audio sample gets discarded
    :return: signals (list of np arrays of audio data), labels (list of corresponding labels)
    """

    # Load dict from txt if dict wasn't directly provided
    if isinstance(label_to_num, str):
        with open(label_to_num, 'r') as file:
            label_to_num = json.load(file)

    if not isinstance(label_to_num, dict):
        print("label_to_num seems to be neither a dict nor a path to a json dict.")
        return

    # Load Train.csv file
    csv_file = open(train_csv_path, newline='')
    train_csv = csv.reader(csv_file, delimiter=',')
    # Call iterator one time to get rid of first row that only contains descriptions
    train_csv.__next__()

    signals = []
    labels = []
    ignored_audio = []

    for row in train_csv:
        filename, label = row
        # Get only name of wav file without rest of path in Train.csv
        filename = filename.split('/')[-1]
        # Convert string label to number label
        label_num = label_to_num[label]

        # Using os.path.join for all path operations to make it OS agnostic
        file_path = os.path.join(path_original, filename)

        # Load signal and calculate total energy
        signal, sr = librosa.load(file_path, sr=sample_rate)
        energy = np.sum(librosa.feature.rms(signal))

        # Skip if energy is below threshold
        if energy <= energy_threshold:
            ignored_audio.append(filename)
            continue

        signals.append(signal)
        labels.append(label_num)

    print("Following files where under the energy threshold of {} and "
          "were not loaded: {}".format(energy_threshold, ignored_audio))

    return signals, labels


def load_additional_utterances(path_additional="data\\AdditionalUtterances\\latest_keywords\\",
                               label_to_num="label_to_num.txt", sample_rate=44100,
                               energy_threshold=0.18, subfolder_name="corrected"):
    """Function to load and return all additional utterances

        :param path_additional: Path to directory where the separate class directories are found
        :param label_to_num: Path to the label_to_num.txt file or the already loaded dict
        :param sample_rate: sample rate for loading audio data
        :param energy_threshold: Energy threshold under which an audio sample gets discarded
        :param subfolder_name: Name of subfolder if correct_additional_utterances() has already been run.
                               Put None if that's not the case.
        :return: signals (list of np arrays of audio data), labels (list of corresponding labels)
        """

    # Load dict from txt if dict wasn't directly provided
    if isinstance(label_to_num, str):
        with open(label_to_num, 'r') as file:
            label_to_num = json.load(file)

    if not isinstance(label_to_num, dict):
        print("label_to_num seems to be neither a dict nor a path to a json dict.")
        return

    signals = []
    labels = []
    ignored_audio = []

    class_dirs = os.listdir(path_additional)

    for class_dir in class_dirs:
        # Convert class folder label to numerical label
        label_num = label_to_num[class_dir]

        target_folder_path = os.path.join(path_additional, class_dir)
        # if subfolder_name is provided, assume that right data is found there
        # and correct_additional_utterances() has already been run
        if subfolder_name is not None:
            target_folder_path = os.path.join(target_folder_path, subfolder_name)

        audio_files = os.listdir(target_folder_path)
        for audio_file in audio_files:
            file_path = os.path.join(target_folder_path, audio_file)

            # Load signal and calculate total energy
            signal, sr = librosa.load(file_path, sr=sample_rate)
            energy = np.sum(librosa.feature.rms(signal))

            # Skip if energy is below threshold
            if energy <= energy_threshold:
                ignored_audio.append(os.path.join(class_dir, audio_file))
                continue

            signals.append(signal)
            labels.append(label_num)

    print("Following files where under the energy threshold of {} and "
          "were not loaded: {}".format(energy_threshold, ignored_audio))

    return signals, labels
