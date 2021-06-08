import cv2
import os
import pathlib
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.keras as keras
import time
import WFLWDataSet
from tensorflow.keras.callbacks import TensorBoard

train_ds = WFLWDataSet.image_label_ds
val_ds = WFLWDataSet.val_image_label_ds


img = list(train_ds.take(1).as_numpy_iterator())[0][0]

IMAGE_WIDTH, IMAGE_HEIGHT,IMAGE_CHANNEL =  img.shape[-3:]

landmark = list(train_ds.take(1).as_numpy_iterator())[0][1]
print(len(landmark[0]))

model_path_name = "E:/ProcessDataWeights4.model"



model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNEL)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2),strides = (2,2)),

    tf.keras.layers.Conv2D(64, (3, 3),strides = (1,1), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(64, (3, 3),strides = (1,1), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2),strides = (2,2)),

    tf.keras.layers.Conv2D(64, (3, 3),strides = (1,1), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(64, (3, 3),strides = (1,1), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2),strides = (2,2)),

    tf.keras.layers.Conv2D(128, (3, 3),strides = (1,1), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(128, (3, 3),strides = (1,1), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2),strides = (1,1)),

    tf.keras.layers.Conv2D(256, (3, 3),strides = (1,1), activation='relu'),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1024, activation = 'relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(196, activation = None)
    ])

def test():
    index = random.randint(0,len(WFLWDataSet.test_filenames)-1)
    filename = WFLWDataSet.test_filenames[index]
    f = WFLWDataSet.load_and_preprocess_image(filename)
    logits = model(tf.expand_dims(f,0), training=False)  # Logits for this minibatch
    WFLWDataSet.test(filename,WFLWDataSet.test_landmarks[index],logits[0].numpy())
    cv2.waitKey()

#test()

'''
try:
    load_status = model.load_weights(model_path_name)
except:
    print("load model weights failed")
'''

model.summary()

timeStamp = str(int(time.time()))
tbCallBack = TensorBoard( log_dir='E:/Logs/'+timeStamp+'/')

opt = tf.keras.optimizers.Adam(0.001)
# Train
model.compile(optimizer=opt,
              loss=tf.keras.losses.MAE,
              metrics=[keras.metrics.mae])

steps_per_epoch = 100 #int(len(WFLWDataSet.train_filenames)/WFLWDataSet.BATCH_SIZE)
history = model.fit(train_ds, epochs=1,
                    validation_data=val_ds, verbose=1,callbacks=[tbCallBack])

model.save_weights(model_path_name)
print("model saved")


for i in range(4):
    index = random.randint(0,len(WFLWDataSet.test_filenames)-1)
    filename = WFLWDataSet.test_filenames[index]
    f = WFLWDataSet.load_and_preprocess_image(filename)
    logits = model(tf.expand_dims(f,0), training=False)  # Logits for this minibatch

    WFLWDataSet.test(filename,WFLWDataSet.test_landmarks[index],logits[0].numpy())

cv2.waitKey()