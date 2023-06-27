from config import *
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.morphology import binary_opening, disk, label
from PIL import Image
from utils import utils, losses, generators
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras import models, layers
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from keras.losses import binary_crossentropy
import keras.backend as K

# Parameters
path = './'
train_path ='/kaggle/input/airbus-ship-detection/train_v2'
test_path='/kaggle/input/airbus-ship-detection/test_v2'
segmentation ='/kaggle/input/airbus-ship-detection/sample_submission_v2.csv
IMG_WIDTH = 768
IMG_HEIGHT = 768
IMG_CHANNELS = 3
TARGET_WIDTH = 128
TARGET_HEIGHT = 128
batch_size=16
image_shape=(768, 768)
img_scaling = (3, 3)
count=900

# Data Preparation
train = os.listdir(image_path)
test = os.listdir(test_exz)
train_set = pd.read_csv(segmentation)
print(masks.head())

## Drop images without ships
not_empty = train_set[train_set['EncodedPixels'].isna() == False]
print(df_train.describe())

##Random Undersampling to generate a better balanced data to work with
train_set['hasShips'] = train_set['EncodedPixels'].map(lambda s: 0 if type(s)==float else 1)
img_un = train_set.groupby(['ImageId'])['hasShips'].sum().reset_index()
img_un['sortData'] = img_un['hasShips'].map(lambda s: 1.0 if x>0 else 0.0)

#Undersample Empty Images
samples = 3000
balanced_data = img_un.groupby('ships').apply(lambda s: s.sample(samples) if len(s) > samples else s)
train_set.drop(['ships'], axis=1, inplace=True)

# Split the data into train and validation sets
train_X,valid_Set = train_test_split(balanced_data, test_size=0.3, random_state=42, stratify = balanced_data['ships'])



#2.2 Split & Image generators
train_ids, valid_ids = train_test_split(balanced_train_df, 
                 test_size = 0.2, 
                 stratify = balanced_train_df['ships'])

train_value = merge(train_set, train_ids)
valid_value = merge(train_set, valid_ids)

# Generate train data 
                
next_gen = dataGenerator(train_value)
train_x, train_y = next(train_gen)
valid_x, valid_y = next(dataGenerator(valid_value, count))


########################################################################################################################
# Model definition
########################################################################################################################
def unetModel(pretrained_weights = None, input_size = (256, 256, 3), NET_SCALING = NET_SCALING):
    inputs = layers.Input(input_size)

     conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
    drop4 = tf.keras.layers.Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    # Decoding Path
    up5 = Conv2D(256, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(pool4))
    merge5 = concatenate([drop4, up5], axis=3)
    conv5 = Conv2D(256, 3, activation='relu', padding='same')(merge5)
    conv5 = Conv2D(256, 3, activation='relu', padding='same')(conv5)

    up6 = Conv2D(128, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv5))
    merge6 = concatenate([conv3, up6], axis=3)
    conv6 = Conv2D(128, 3, activation='relu', padding='same')(merge6)
    conv6 = Conv2D(128, 3, activation='relu', padding='same')(conv6)

    up7 = Conv2D(64, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv2, up7], axis=3)
    conv7 = Conv2D(64, 3, activation='relu', padding='same')(merge7)
    conv7 = Conv2D(64, 3, activation='relu', padding='same')(conv7)

    up8 = Conv2D(32, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv1, up8], axis=3)
    conv8 = Conv2D(32, 3, activation='relu', padding='same')(merge8)
    conv8 = Conv2D(32, 3, activation='relu', padding='same')(conv8)
    conv8 = Conv2D(2, 3, activation='relu', padding='same')(conv8)

    # Output Layer
    output = Conv2D(1, 1, activation='sigmoid')(conv8)

    # Define the model
    model = Model(inputs=inputs, outputs=output)

    return model
detection_model = unetModel()


#3-2. callbacks setting
########################################################################################################################
early_stopping = EarlyStopping(patience=50, verbose=1)
checkpoint = ModelCheckpoint(filepath, monitor='val_dice_coef',save_best_only=True, verbose=1)

########################################################################################################################
# Train the network
def learning():
    detection_model.compile(optimizer='adam', loss=dice_p_bce, metrics=[dice_coef])
    step_count = min(MAX_TRAIN_STEPS, train_df.shape[0]//BATCH_SIZE)
    aug_gen = generators.create_aug_gen(generators.make_image_gen(train_df))
    loss_history = [seg_model.fit(aug_gen,
                                 steps_per_epoch=step_count,
                                 epochs=MAX_TRAIN_EPOCHS,
                                 validation_data=(valid_x, valid_y),
                                 callbacks=callbacks_list,
                                workers=1 # the generator is not very thread safe
                                           )]
    return loss_history


history = model.fit_generator(generator=training_generator,
                              steps_per_epoch=250,
                              epochs=500,
                              validation_data=validation_generator,
                              validation_steps=len(validation_generator),
                              callbacks=[checkpoint, logger, csv_logger],
                              verbose=1)
