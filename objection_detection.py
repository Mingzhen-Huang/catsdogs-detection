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

CLASSIFY_BATCH_SIZE = 32


model = model.make_classification_model()
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

checkpoint = ModelCheckpoint('cats_vs_dogs.best_weights.hdf5', 
                             monitor='val_acc', 
                             verbose=0, 
                             save_best_only=True, 
                             save_weights_only=True, 
                             mode='auto')
cats_dogs_train_dataset = util.make_dataset_for_directory('catsdogs/training_set/', CLASSIFY_BATCH_SIZE, augment=False)
cats_dogs_valid_dataset = util.make_dataset_for_directory('catsdogs/test_set/', 50, augment=False)
model.fit(cats_dogs_train_dataset, 
          epochs=30,
          steps_per_epoch=8000//32,
          validation_data=cats_dogs_valid_dataset,
          validation_steps=10,
          callbacks=[checkpoint, TensorBoard('./catsdogs-log'), EarlyStopping(patience=3)])
model.load_weights('cats_vs_dogs.best_weights.hdf5')
id_to_cls = lambda x: 'cat' if x == 1 else 'dog'
it = util.make_dataset_for_directory('catsdogs/test_set/', 1000, augment=False).make_one_shot_iterator()
x, y = tf.keras.backend.get_session().run(it.get_next())
y_pred = model.predict(x)
y_d = np.argmax(y, axis=1)
y_pred_d = np.argmax(y_pred, axis=1)

plt.figure(figsize=(15,15))
for i in range(100):
    plt.subplot(10,10,i+1)
    plt.imshow((np.squeeze(x[i]) + 1.0) / 2.0)
    plt.title("T: %s P: %s"%(id_to_cls(y_d[i]),id_to_cls(y_pred_d[i])), color='b' if y_d[i] == y_pred_d[i] else 'r')
    plt.xticks([]),plt.yticks([])
plt.show()
