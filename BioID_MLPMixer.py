import cv2
import os
import pathlib
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras import datasets, layers
import time
from nd_mlp_mixer import MLPMixer
from tensorflow.keras.callbacks import TensorBoard
import BioIDDataSet


train_ds,val_ds,train_image_paths,val_image_paths = BioIDDataSet.get_bioid_ds()

img = list(train_ds.take(1).as_numpy_iterator())[0][0]

width, height,channel =  img.shape[-3:]

model_path_name = "E:/tf_learn/BioID_MLPMixerWeights.model"


# Prepare the model (add channel dimension to images)
inputs = layers.Input(shape=(width, height,channel))
#h = layers.Reshape([28, 28, 1])(inputs)
mlp_mixer = MLPMixer(num_classes=4, 
                     num_blocks=2, 
                     patch_size=4, 
                     hidden_dim=28, 
                     tokens_mlp_dim=height,
                     channels_mlp_dim=width)(inputs)
model = tf.keras.Model(inputs=inputs, outputs=mlp_mixer)
print(model.summary())

try:
    load_status = model.load_weights(model_path_name)
except:
    print("load model weights failed")

timeStamp = str(int(time.time()))
tbCallBack = TensorBoard( log_dir='E:/Logs/'+timeStamp+'/')

# Train
model.compile(optimizer='adam',
              loss=tf.keras.losses.MAE,
              metrics=[keras.metrics.mae])

history = model.fit(train_ds, epochs=20,steps_per_epoch=100,
                    validation_data=val_ds, verbose=2,callbacks=[tbCallBack])

#model.save_weights(model_path_name)
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