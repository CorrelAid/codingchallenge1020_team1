import numpy as np
import librosa
import librosa.feature
import json
import os
import csv


def build_new_trainingset(path_original="data\\audio_files",
                          path_additional="data\\AdditionalUtterances\\latest_keywords\\",
                          path_new_dataset="data\\training", train_csv_path="Train.csv",
                          energy_threshold=0.18, target_samplerate=48000,
                          target_channels=1, corrected=True):
    """A function to convert all utterances to unified sample rate and channel audio files

        :param path_original: Path to directory where .wav files are located
        :param path_additional: Path to the directory in which the class directories are found
        :param path_new_dataset: Path to directory where new, converted dataset should be saved
        :param train_csv_path: Path to the Train.csv file
        :param energy_threshold: Energy threshold under which an audio sample gets discarded
        :param target_samplerate: Sample rate that all data gets converted to
        :param target_channels: Target channel number all data gets converted to. Needs to be 1 or 2.
        :param corrected: Boolean that shows whether additional data should be gotten from a subdirectory
        called "corrected", as would be the result of correct_additional_utterances
        :return: labels, a dictionary containing mappings from all training file names to their labels
        """

    # First deal with original data
    # Load Train.csv file
    csv_file = open(train_csv_path, newline='')
    train_csv = csv.reader(csv_file, delimiter=',')
    # Call iterator one time to get rid of first row that only contains descriptions
    train_csv.__next__()

    if not os.path.isdir(path_new_dataset):
        os.mkdir(path_new_dataset)
        print("Creating new folder for data at {}".format(os.path.abspath(path_new_dataset)))

    labels = {}
    ignored_audio = []

    print("Starting iteration through original data.")

    for row in train_csv:
        filename, label = row
        # Get only name of wav file without rest of path in Train.csv
        filename = filename.split('/')[-1]
        # Convert string label to number label

        # Using os.path.join for all path operations to make it OS agnostic
        file_path = os.path.join(path_original, filename)

        # Load signal and calculate total energy
        signal, sr = librosa.load(file_path, sr=44100)
        energy = np.sum(librosa.feature.rms(signal))

        # Skip if energy is below threshold
        if energy <= energy_threshold:
            ignored_audio.append(filename)
            continue

        file_new_path = os.path.join(path_new_dataset, filename)
        command_to_run = "ffmpeg -i {} -ac {} -ar {} {}".format(file_path, target_channels, target_samplerate, file_new_path)
        os.system(command_to_run)

        labels[filename] = label

    print("Finished original data, iterating through additional data now.")
    # Now go through additional utterances and do the same
    # Create list of all class folders
    class_dirs = os.listdir(path_additional)

    # Catch some pathing mistakes (though not all)
    if len(class_dirs) == 0:
        print("The given directory is empty.")
        return

    elif len(class_dirs[0].split(".")) != 1:
        print("The first entry in given directory is not a folder. You should"
              "give the path to the directory in which the class-subfolders are.")
        return

    for class_dir in class_dirs:
        # Path to distinctive class subfolder
        class_path = os.path.join(path_additional, class_dir)
        if corrected:
            class_path = os.path.join(path_additional, class_dir, "corrected")

        # List of all audio files in respective class directory
        audio_files = os.listdir(class_path)

        for filename in audio_files:
            file_path = os.path.join(class_path, filename)

            # Load signal and calculate total energy
            signal, sr = librosa.load(file_path, sr=44100)
            energy = np.sum(librosa.feature.rms(signal))

            # Skip if energy is below threshold
            if energy <= energy_threshold:
                ignored_audio.append(filename)
                continue

            file_new_path = os.path.join(path_new_dataset, filename)

            # Run FFMPEG command on CMD to convert audio file to target location (in new subfolder)
            # with new samplerate and channel number
            command_to_run = "ffmpeg -i {} -ac {} -ar {} {}".format(file_path, target_channels, target_samplerate, file_new_path)
            os.system(command_to_run)

            labels[filename] = class_dir

    # Save labels dict
    with open(os.path.join(path_new_dataset, 'labels.txt'), 'w') as outfile:
        json.dump(labels, outfile)

    print("Following files where under the energy threshold of {} and "
          "were not loaded: {}".format(energy_threshold, ignored_audio))

    return labels


def correct_additional_utterances(path_additional, subfolder_name="corrected"):
    """A function to convert the additional utterances of the Zindi-Challenge to proper WAV files

    The function uses FFMPEG to convert the additional given .wav files (which all actually use OGG
    codec) to proper .wav files to avoid handling problems. It assumes the original given folder
    structure in which every class has a separate folder with some .wav files in there. It will create
    a new subfolder and store the converted versions (with the same name as the original) in there.

    :param path_additional: Path to the directory in which the class directories are found
    :param subfolder_name: Name of the newly created subfolder for the new converted audio
    :return: Returns None if aborted due to wrong input
    """

    # Create list of all class folders
    class_folders = os.listdir(path_additional)

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
        class_path = os.path.join(path_additional, class_folder)
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


def load_testset(path_original="data\\audio_files", train_csv_path="Train.csv", sample_rate=44100):
    """Function to load and return all (unlabeled) test utterances

    :param path_original: Path to directory where original .wav files are located
    :param train_csv_path: Path to the Train.csv file
    :param sample_rate: sample rate for loading audio data
    :return: signals (list of np arrays of test audio data)
    """

    # Load Train.csv file
    csv_file = open(train_csv_path, newline='')
    train_csv = csv.reader(csv_file, delimiter=',')
    # Call iterator one time to get rid of first row that only contains descriptions
    train_csv.__next__()

    # Create list with all train samples (only file names) from original folder
    all_train_samples = [row[0].split('/')[-1] for row in train_csv]

    all_audio_files = os.listdir(path_original)

    signals = []

    for filename in all_audio_files:

        # Using os.path.join for all path operations to make it OS agnostic
        file_path = os.path.join(path_original, filename)

        # Skip sample if it's part of the training set and thus not test set
        if filename in all_train_samples:
            continue

        # Load signal and save to list
        signal, sr = librosa.load(file_path, sr=sample_rate)
        signals.append(signal)

    return signals
