########################################################################################################################
# Imports
########################################################################################################################
import numpy as np
import pandas as pd
import imageio
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

from keras.applications import imagenet_utils
from keras.utils import Sequence
from keras import backend as K

import matplotlib.pylab as pl
########################################################################################################################
# Define the generators

def dataGenerator(dataFR, batch_size = 4,path, n_channels=(3,3), isChanels=true):
  all_LIST = list(dataFR.groupby('ImageId'))
  img = []
  msk=[]
  while True:
    np.random.shuffle(all_batches)
    for img_id, mask in all_LIST:
      img_path = os.path.join(path, img_id)
      image=imread(img_path)
      c_mask = np.expand_dims(all_mask(c_masks['EncodedPixels'].values), -1)
      if train.img_scaling is isChanels:
        addI = c_img[::n_channels[0], ::n_channels[1]]
        addM = c_mask[::train.img_scaling[0], ::train.img_scaling[1]]
        img.append(addI)
        msk.append(addM)
        if len(out_rgb)>=batch_size:
          yield np.stack(img, 0)/255.0, np.stack(c_mask, 0).astype(np.float32)
          img, msk=[], []
########################################################################################################################
#These functions will help in calculating the ship area and will group them by imageId
def decoder(mask,shape=(768,768)):
    imgSize=np.zeros(shape[0]*shape[1],dtype=np.uint8)
    changeMask=mask.split()
    for i in range(len(changeMask)//2):  
        start = int(changeMask[2*i]) - 1
        length = int(changeMask[2*i+1])
        imgSize[start:start+length] = 1
    return imgSize.reshape(shape).T
    
def all_mask(imgEx):
    img = np.zeros((768,768),dtype = np.int16)  
    for signs in imgEx:
                   if (type(signs)==float):  break
                   else:
                        mask=decoder(signs)
                        img+=mask
    newImg=np.expand_dims(img, -1)              
    return newImg
  ########################################################################################################################

# function that merges two layers (Concatenate)
def merge(input1, input2):
  x = concatenate([input1, input2])
  return x
