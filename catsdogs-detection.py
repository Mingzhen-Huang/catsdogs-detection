import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import patches
import tensorflow as tf
import requests
import io
import zipfile
import sys
import json
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, precision_recall_curve

from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from tensorflow.keras import backend as K

import util
import model

bb_box_grid = util.get_grid()

detection_ds_train, detection_ds_val = util.make_detection_dataset_for_directory('./catsdogs_detection/', 32,bb_box_grid)
it = detection_ds_val.make_one_shot_iterator()
img,y = K.get_session().run(it.get_next())

detection_model = model.make_detection_model(bb_box_grid)
detection_model.compile(loss=model.detection_loss,
              optimizer='adam')
checkpoint = ModelCheckpoint('cats_dogs_detection.best_weights.hdf5', 
                             monitor='val_loss', 
                             verbose=0, 
                             save_best_only=True, 
                             save_weights_only=True, 
                             mode='auto')              
detection_model.fit(detection_ds_train,
                    epochs=20,
                    steps_per_epoch=2500//32,
                    validation_data=detection_ds_val,
                    validation_steps=50,
                    callbacks=[checkpoint]
                    )