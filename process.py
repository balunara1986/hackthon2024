import os
import shutil
import warnings
import argparse
import librosa
import torch
import torchvision
from PIL import Image
from ffnn.model import FFNN
from cnn.model import CNN
from utils.gen_utils import create_dir
from utils.ffnn_utils import apply_transforms, transforms_to_tensor
from utils.cnn_utils import get_melss
from flask import jsonify

warnings.filterwarnings('ignore', category=UserWarning)


def run(file, ffnn_path, cnn_path):
    # set device
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device( 'cpu')

    # other
    print('os.getcwd(): '+os.getcwd())
    print('os.pardir: '+os.pardir)
    root_dir = str(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
    temp_dir = str(os.path.abspath(os.path.join(os.getcwd(), 'temp')))
    print('root dir is: '+root_dir)
    to_tensor = torchvision.transforms.ToTensor()


    # get models in eval mode
    ffnn = FFNN()
    ffnn_path = os.path.abspath('saved_models/' + ffnn_path)
    ffnn.load_state_dict(torch.load(ffnn_path, map_location=torch.device('cpu')), strict=False)
    ffnn = ffnn.to(device)
    ffnn.eval()

    cnn = CNN()
    cnn_path = os.path.abspath('saved_models/' + cnn_path)
    cnn.load_state_dict(torch.load(cnn_path, map_location=torch.device('cpu')), strict=False)
    cnn = cnn.to(device)
    cnn.eval()

    # create temp dir to save melss image for current inference
    create_dir('temp')

    # get transforms and spectrogram image
    transforms = apply_transforms(file)
    melss = get_melss(file, 'temp/test.jpg')

    # convert transforms dict to tensor and
    # apply transforms to melss image
    transforms = transforms_to_tensor(transforms)
    melss = Image.open('temp/test.jpg')
    melss = melss.resize((32, 32))
    melss = to_tensor(melss)
    melss = melss.to(device)

    # make predictions
    ffnn_pred = ffnn(transforms)
    cnn_pred = cnn(melss.unsqueeze(0))

    # value, index
    _, predictions = torch.max(cnn_pred, 1)  # 1 is the dimension
    print(predictions)

    print(ffnn_pred)

    print(cnn_pred)
    # if both models agree that the audio is a human, return human
    # else, return AI
    if ffnn_pred[1] > 0.55 and cnn_pred[0][1] > 0.85:
        print(ffnn_pred)
        print(cnn_pred)
        print('\nhuman\n')

    else:
        print(ffnn_pred)
        print(cnn_pred)
        print('\nai\n')

    # delete temp dir after completion
    print('temp dir is: '+temp_dir )
    if os.path.isdir(temp_dir ):
        print('deleting temp dir ...\n')
        os.remove(temp_dir + '/test.jpg')
        shutil.rmtree(temp_dir)
    else:
        print('temp dir does not exist...\n')

    print('Inference complete ...')
    return ffnn_pred, cnn_pred

