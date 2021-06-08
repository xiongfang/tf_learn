import cv2
import os
import pathlib
import random
import tensorflow as tf
import matplotlib.pyplot as plt

IMAGE_WIDTH = int(384/4)
IMAGE_HEIGHT = int(286/4)

path_root = "E:/DataSet/BioID_Face/data/BioID-FaceDatabase-V1.2"


data_root = pathlib.Path(path_root)
print(data_root)

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

all_label_paths = list(data_root.glob('*.eye'))
all_label_paths = [str(path) for path in all_label_paths]

all_image_labels = [ get_eye_pos(path) for path in all_label_paths]

train_image_labels = all_image_labels[:1000]
val_image_labels = all_image_labels[1000:]

rescale = tf.keras.layers.experimental.preprocessing.Rescaling(1./127.5, offset= -1)
def preprocess_image(image):
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.image.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
  image /= 255.0  # normalize to [0,1] range
  #image = rescale(image)
  return image

def load_and_preprocess_image(path):
  img = tf.io.read_file(path)
  return preprocess_image(img)

#构建dataset
path_ds = tf.data.Dataset.from_tensor_slices(train_image_paths)
image_ds = path_ds.map(load_and_preprocess_image)

label_ds = tf.data.Dataset.from_tensor_slices(train_image_labels)

path_ds = tf.data.Dataset.from_tensor_slices(val_image_paths)
val_image_ds = path_ds.map(load_and_preprocess_image)
val_label_ds = tf.data.Dataset.from_tensor_slices(val_image_labels)

#label_ds = label_ds.map(get_eye_pos)
#for label in label_ds.take(10):
#  print(label.numpy())

#print(list(label_ds.as_numpy_iterator()))

image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))

BATCH_SIZE = 32
# 设置一个和数据集大小一致的 shuffle buffer size（随机缓冲区大小）以保证数据
# 被充分打乱。
ds = image_label_ds.shuffle(buffer_size=len(train_image_paths))

ds = ds.repeat()
ds = ds.batch(BATCH_SIZE)

# 当模型在训练的时候，`prefetch` 使数据集在后台取得 batch。
ds = ds.prefetch(tf.data.AUTOTUNE)

val_image_label_ds = tf.data.Dataset.zip((val_image_ds, val_label_ds))
val_image_label_ds = val_image_label_ds.batch(BATCH_SIZE,drop_remainder = True)

def get_bioid_ds():
	return ds,val_image_label_ds,train_image_paths,val_image_paths