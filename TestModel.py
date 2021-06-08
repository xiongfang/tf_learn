import cv2
import os
import pathlib
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.keras as keras
import time

IMAGE_WIDTH = int(384/4)
IMAGE_HEIGHT = int(286/4)

path_root = "E:/DataSet/BioID_Face/data/BioID-FaceDatabase-V1.2"

data_root = pathlib.Path(path_root)

all_image_paths = list(data_root.glob('*.jpg'))
all_image_paths = [str(path) for path in all_image_paths]
train_image_paths = all_image_paths[:1000]
val_image_paths = all_image_paths[1000:]


def get_eye_pos(eye_file):
	with open(eye_file,"r") as f:
		eye = f.readline()
		eye = f.readline()
		pos_list = eye.split('\t')
	
		LX = float(pos_list[0])/384
		LY = float(pos_list[1])/286
		RX = float(pos_list[2])/384
		RY = float(pos_list[3])/286
		return [LX,LY,RX,RY]
def pos_to_cv(eye_pos):
	return [[int(eye_pos[0]*384),int(eye_pos[1]*286)],[int(eye_pos[2]*384),int(eye_pos[3]*286)]]

all_label_paths = list(data_root.glob('*.eye'))
all_label_paths = [str(path) for path in all_label_paths]

all_image_labels = [ get_eye_pos(path) for path in all_label_paths]

train_image_labels = all_image_labels[:1000]
val_image_labels = all_image_labels[1000:]

def preprocess_image(image):
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.image.resize(image, [IMAGE_WIDTH, IMAGE_HEIGHT])
  image /= 255.0  # normalize to [0,1] range
  return image

def load_and_preprocess_image(path):
  img = tf.io.read_file(path)
  return preprocess_image(img)


model_path_name = "E:/tf_learn/ProcessDataWeights.model"
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='elu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='elu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
	tf.keras.layers.Conv2D(128, (3, 3), activation='elu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='elu',kernel_regularizer = tf.keras.regularizers.L1(0.01)),
    tf.keras.layers.Dense(4, activation = 'elu')])
model.load_weights(model_path_name)

path_ds = tf.data.Dataset.from_tensor_slices(val_image_paths)
val_image_ds = path_ds.map(load_and_preprocess_image)
val_label_ds = tf.data.Dataset.from_tensor_slices(val_image_labels)
val_image_label_ds = tf.data.Dataset.zip((val_image_ds, val_label_ds))
val_image_label_ds = val_image_label_ds.batch(1,drop_remainder = True)

img_list = []
for i in range(5):
	index = random.randint(0,len(val_image_paths)-1)
	val_image_label_ds_list = list(val_image_label_ds.as_numpy_iterator())
	x_batch_train = val_image_label_ds_list[index][0]
	y_batch_train = val_image_label_ds_list[index][1]

	logits = model(x_batch_train, training=False)  # Logits for this minibatch

	img = cv2.imread(val_image_paths[index])

	eye = pos_to_cv(y_batch_train[0])

	cv2.circle(img,eye[0],2,(0,0,255))
	cv2.circle(img,eye[1],2,(0,0,255))

	eye = pos_to_cv(logits[0])
	cv2.circle(img,eye[0],2,(0,255,0))
	cv2.circle(img,eye[1],2,(0,255,0))
	img_list.append(img)


	cv2.imshow("a%d"%i,img)


cv2.waitKey()

