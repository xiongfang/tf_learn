import cv2
import os
import pathlib
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

IMAGE_WIDTH = 112
IMAGE_HEIGHT = 112
IMAGE_CHANNEL = 3

path_root = "E:/DataSet/WFLW"


data_root = pathlib.Path(path_root)
print(data_root)

train_label_file = str(data_root.joinpath("train_data"))+"/list.txt"
val_label_file = str(data_root.joinpath("test_data"))+"/list.txt"

def gen_data(file_list):
    with open(file_list,'r') as f:
        lines = f.readlines()
    filenames, landmarks,attributes,euler_angles = [], [], [],[]
    for line in lines:
        line = line.strip().split()
        path = line[0]
        landmark = line[1:197]
        attribute = line[197:203]
        euler_angle = line[203:206]

        landmark = np.asarray(landmark, dtype=np.float32)
        attribute = np.asarray(attribute, dtype=np.int32)
        euler_angle = np.asarray(euler_angle,dtype=np.float32)
        filenames.append(path)
        landmarks.append(landmark)
        attributes.append(attribute)
        euler_angles.append(euler_angle)
        
    filenames = np.asarray(filenames, dtype=np.str)
    landmarks = np.asarray(landmarks, dtype=np.float32)
    attributes = np.asarray(attributes, dtype=np.int32)
    euler_angles = np.asarray(euler_angles,dtype=np.float32)
    return (filenames, landmarks, attributes,euler_angles)


def preprocess_image(image):
  image = tf.image.decode_png(image, channels=IMAGE_CHANNEL)
  image = tf.image.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
  image /= 255.0  # normalize to [0,1] range
  #image = rescale(image)
  return image

def load_and_preprocess_image(path):
  img = tf.io.read_file(path)
  return preprocess_image(img)

train_filenames, train_landmarks, attributes,euler_angles = gen_data(train_label_file)

path_ds = tf.data.Dataset.from_tensor_slices(train_filenames)
image_ds = path_ds.map(load_and_preprocess_image,num_parallel_calls = tf.data.AUTOTUNE)
label_ds = tf.data.Dataset.from_tensor_slices(train_landmarks)
image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))

BATCH_SIZE = 64

# 设置一个和数据集大小一致的 shuffle buffer size（随机缓冲区大小）以保证数据
# 被充分打乱。
image_label_ds = image_label_ds.shuffle(buffer_size=1000)
image_label_ds = image_label_ds.batch(BATCH_SIZE)
# 当模型在训练的时候，`prefetch` 使数据集在后台取得 batch。
image_label_ds = image_label_ds.prefetch(tf.data.AUTOTUNE)

test_filenames, test_landmarks, attributes,euler_angles = gen_data(train_label_file)

path_ds = tf.data.Dataset.from_tensor_slices(test_filenames)
val_image_ds = path_ds.map(load_and_preprocess_image)
val_label_ds = tf.data.Dataset.from_tensor_slices(test_landmarks)
val_image_label_ds = tf.data.Dataset.zip((val_image_ds, val_label_ds))
val_image_label_ds = val_image_label_ds.batch(BATCH_SIZE)

#val_image_label_ds = val_image_label_ds.prefetch(tf.data.AUTOTUNE)

def test(filename,landmark,landmark_pre):
    img = cv2.imread(filename)
    h,w,_ = img.shape
    landmark = landmark.reshape(-1,2)*[h,w]
    for (x,y) in landmark.astype(np.int32):
        cv2.circle(img, (x,y),1,(0,0,255))
    if landmark_pre is None:
        ...
    else:
        landmark = landmark_pre.reshape(-1,2)*[h,w]
        for (x,y) in landmark.astype(np.int32):
            cv2.circle(img, (x,y),1,(0,255,0))
    #cv2.imshow(filename, img)
    return img
