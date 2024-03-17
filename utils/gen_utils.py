import os
import random
import numpy as np
import librosa
import torch
#from pydub import AudioSegment
import subprocess
root_dir = 'C:/Users/Srinivas/PycharmProjects'#str(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))


def create_dir(dir: str) -> None:
    if os.path.isdir(dir):
        print(dir, 'already exists. Continuing ...')
    else:
        print('Creating new dir: ', dir)
        os.makedirs(dir)

# convert .mp3 files to .wav files in the human data folder
def mp32wav() -> None:
    flac_path = str(root_dir + '/Hackathon-VoiceDetection/data/human/mp3/')
    wav_path = str(root_dir + '/Hackathon-VoiceDetection/data/human/wav/')
    flac_files = [f for f in os.listdir(flac_path) if os.path.isfile(os.path.join(flac_path, f)) and f.endswith('.mp3')]

    for file in flac_files:
        print('Converting ' + str(file))
        print(str(flac_path + file));
        #temp = AudioSegment.from_file(str(flac_path + file))
         # temp.export(str(wav_path + os.path.splitext(file)[0]) + '.wav', format='wav')
        #subprocess.call(['ffmpeg', '-i', str(flac_path + file),
         #                str(wav_path + os.path.splitext(file)[0]) + '.wav'])
    print('Done converting \n')


# create the training and testing split lists for both classes
def create_splits(voice_wavs: str, ai_voice_wavs: str) -> list:
    # get total number of files in the both dirs and split the training and testing dataset
    # by a 80/20 ratio
    voice_list = [voice_wavs + name for name in os.listdir(voice_wavs)]
    voice_total = len(voice_list)
    voice_train_split = round(voice_total * 0.8)
    voice_test_split = voice_total - voice_train_split

    assert voice_train_split + voice_test_split == voice_total

    voice_train_list = random.sample(voice_list, voice_train_split)
    voice_test_list = random.sample(voice_list, voice_test_split)

    ai_voice_list = [ai_voice_wavs + name for name in os.listdir(ai_voice_wavs)]
    ai_voice_total = len(ai_voice_list)
    ai_voice_train_split = round(ai_voice_total * 0.8)
    ai_voice_test_split = ai_voice_total - ai_voice_train_split

    assert ai_voice_train_split + ai_voice_test_split == ai_voice_total

    ai_voice_train_list = random.sample(ai_voice_list, ai_voice_train_split)
    ai_voice_test_list = random.sample(ai_voice_list, ai_voice_test_split)

    # concat into two complete lists
    full_train_list = voice_train_list + ai_voice_train_list
    full_test_list = voice_test_list + ai_voice_test_list

    return full_train_list, full_test_list


# calculate accuracy of a prediction
def get_accuracy(prediction: str, label: str) -> float:
    matches = [torch.argmax(i) == torch.argmax(j) for i, j in zip(prediction, label)]
    accuracy = matches.count(True) / len(matches)
    return accuracy