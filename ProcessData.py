import cv2
import os
import pathlib
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


IMAGE_SIZE = 32

path_root = "E:\tf_learn\BioID_Face\data\BioID-FaceDatabase-V1.2"


data_root = pathlib.Path(path_root)
print(data_root)

all_image_paths = list(data_root.glob('*.jpg'))
all_image_paths = [str(path) for path in all_image_paths]

#random.shuffle(all_image_paths)

image_count = len(all_image_paths)
print(image_count)

def get_eye_pos(eye_file):
	with open(eye_file,"r") as f:
		eye = f.readline()
		eye = f.readline()
		pos_list = eye.split('\t')
	
		LX = float(pos_list[0])
		LY = float(pos_list[1])
		RX = float(pos_list[2])
		RY = float(pos_list[3])
		return [LX,LY,RX,RY]

all_label_paths = list(data_root.glob('*.eye'))
all_label_paths = [str(path) for path in all_label_paths]

all_image_labels = [ get_eye_pos(path) for path in all_label_paths]

def preprocess_image(image):
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
  image /= 255.0  # normalize to [0,1] range
  return image

def load_and_preprocess_image(path):
  img = tf.io.read_file(path)
  return preprocess_image(img)

#构建dataset
path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
image_ds = path_ds.map(load_and_preprocess_image)

label_ds = tf.data.Dataset.from_tensor_slices(all_image_labels)
#label_ds = label_ds.map(get_eye_pos)
for label in label_ds.take(10):
  print(label.numpy())

#print(list(label_ds.as_numpy_iterator()))

image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))

BATCH_SIZE = 32
# 设置一个和数据集大小一致的 shuffle buffer size（随机缓冲区大小）以保证数据
# 被充分打乱。
ds = image_label_ds.shuffle(buffer_size=image_count)

#ds = ds.repeat()
ds = ds.batch(BATCH_SIZE,drop_remainder = True)

# 当模型在训练的时候，`prefetch` 使数据集在后台取得 batch。
ds = ds.prefetch(tf.data.AUTOTUNE)


model = tf.keras.Sequential([
  tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
  tf.keras.layers.MaxPooling2D((2, 2)),
  tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
  tf.keras.layers.MaxPooling2D((2, 2)),
  tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dense(4, activation = 'relu')])


model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=["accuracy"])

print(len(model.trainable_variables))

history = model.fit(ds, epochs=1, steps_per_epoch=3)