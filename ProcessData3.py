import cv2
import os
import pathlib
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.keras as keras
import time
import BioIDDataSet
from tensorflow.keras.callbacks import TensorBoard

train_ds,val_ds,train_image_paths,val_image_paths = BioIDDataSet.get_bioid_ds()

img = list(train_ds.take(1).as_numpy_iterator())[0][0]

IMAGE_WIDTH, IMAGE_HEIGHT,IMAGE_CHANNEL =  img.shape[-3:]

model_path_name = "E:/ProcessDataWeights3.model"

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='elu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNEL)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='elu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
	tf.keras.layers.Conv2D(128, (3, 3), activation='elu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='elu',kernel_regularizer = tf.keras.regularizers.L1(0.01)),
    tf.keras.layers.Dense(4, activation = 'elu')])


try:
    load_status = model.load_weights(model_path_name)
except:
    print("load model weights failed")



model.summary()

timeStamp = str(int(time.time()))
tbCallBack = TensorBoard( log_dir='E:/Logs/'+timeStamp+'/')


# Train
model.compile(optimizer="adam",
              loss=tf.keras.losses.MAE,
              metrics=[keras.metrics.mae])

history = model.fit(train_ds, epochs=10,steps_per_epoch=100,
                    validation_data=val_ds, verbose=2,callbacks=[tbCallBack])

model.save_weights(model_path_name)
print("model saved")


def pos_to_cv(eye_pos):
	return [[int(eye_pos[0]*384),int(eye_pos[1]*286)],[int(eye_pos[2]*384),int(eye_pos[3]*286)]]

for i in range(5):
	index = random.randint(0,len(val_ds)-1)
	

	val_image_label_ds_list = list(val_ds.as_numpy_iterator())
	x_batch_train = val_image_label_ds_list[index][0]
	y_batch_train = val_image_label_ds_list[index][1]

	logits = model(x_batch_train, training=False)  # Logits for this minibatch

	img_batch_index = random.randint(0,BioIDDataSet.BATCH_SIZE-1)
	img_index = index*BioIDDataSet.BATCH_SIZE+img_batch_index

	img = cv2.imread(val_image_paths[img_index])

	eye = pos_to_cv(y_batch_train[img_batch_index])

	cv2.circle(img,eye[0],2,(0,0,255))
	cv2.circle(img,eye[1],2,(0,0,255))

	eye = pos_to_cv(logits[img_batch_index])
	cv2.circle(img,eye[0],2,(0,255,0))
	cv2.circle(img,eye[1],2,(0,255,0))

	cv2.imshow("a%d"%i,img)


cv2.waitKey()