from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode

landmarks_frame = pd.read_csv('/Users/apple/Dropbox/0_0 Purdue BME 2017/Deep_Learning/Final-Project/parcellation_1.csv')
n = 65
#img_name = landmarks_frame.ix[2, 2:4]
#landmarks = landmarks_frame.ix[n, 1:].as_matrix().astype('float')
#landmarks = landmarks.reshape(-1, 2)

print(landmarks_frame.ix[557,3])
print(landmarks_frame.ix[2,1])
print(landmarks_frame.ix[0,0]+landmarks_frame.ix[0,3])
print(landmarks_frame.ix[:,:])

#print('Image name: {}'.format(img_name))

#print('Landmarks shape: {}'.format(landmarks.shape))
#print('First 4 Landmarks: {}'.format(landmarks[:4]))
