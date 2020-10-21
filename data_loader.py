import numpy
import librosa
import os

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

