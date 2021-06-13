import cv2
import os
import pathlib
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from mark_operator import MarkOperator

MO = MarkOperator()

IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
IMAGE_CHANNEL = 1
HEATMAP_SIZE = 128

image_shape = (IMAGE_WIDTH,IMAGE_HEIGHT,IMAGE_CHANNEL)

path_root = "E:/DataSet/WFLW"

FILE_WIDTH = 112
FILE_HEIGHT = 112

NUM_MARKS = 2

data_root = pathlib.Path(path_root)
print(data_root)

train_label_file = str(data_root.joinpath("train_data"))+"/list.txt"
val_label_file = str(data_root.joinpath("test_data"))+"/list.txt"


def generate_heatmaps(marks_norm, map_size):
    """A convenient function to generate heatmaps from marks."""
    #marks_norm = marks / img_size
    heatmaps = MO.generate_heatmaps(marks_norm, map_size=map_size)

    return heatmaps

def normalize(inputs):
    """Preprocess the inputs. This function follows the official implementation
    of HRNet.

    Args:
        inputs: a TensorFlow tensor of image.

    Returns:
        a normalized image.
    """
    img_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    img_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    # Normalization
    return ((inputs / 255.0) - img_mean)/img_std

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
            landmark = landmark.reshape(-1, 2)
            landmark = np.pad(landmark, ((0, 0), (0, 1)),mode='constant', constant_values=0) #增加z=0

            temp = landmark
            landmark = []
            #landmark.append(temp[54])
            landmark.append(temp[96])
            landmark.append(temp[97])
            #landmark.append(temp[90])
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
    #image /= 255.0  #normalize to [0,1] range
    #image = rescale(image)
    image = normalize(image)
    return image

def load_and_preprocess_image(path):
    img = tf.io.read_file(path)
    return preprocess_image(img)

def preprocess_mark(landmarks):
    for landmark in landmarks:
        heatmaps = generate_heatmaps(landmark,(HEATMAP_SIZE, HEATMAP_SIZE)) #(2,256,256)
        heatmaps = [heatmaps[0]+heatmaps[1]]
        heatmaps = np.transpose(heatmaps, (1, 2, 0)) #(256,256,2)
        #last = heatmaps[0]
        #for h in heatmaps[1:]:
        #    last+=h
        yield heatmaps

        
train_filenames, train_landmarks, attributes,euler_angles = gen_data(train_label_file)

train_filenames = train_filenames[:1000]
train_landmarks = train_landmarks[:1000]

path_ds = tf.data.Dataset.from_tensor_slices(train_filenames)
image_ds = path_ds.map(load_and_preprocess_image,num_parallel_calls = tf.data.AUTOTUNE)

label_ds = tf.data.Dataset.from_generator(preprocess_mark,output_types=tf.float32,output_shapes=(HEATMAP_SIZE, HEATMAP_SIZE, NUM_MARKS),args=[train_landmarks])

image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))

BATCH_SIZE = 32

# 设置一个和数据集大小一致的 shuffle buffer size（随机缓冲区大小）以保证数据
# 被充分打乱。
image_label_ds = image_label_ds.shuffle(buffer_size=1000)
image_label_ds = image_label_ds.batch(BATCH_SIZE)
# 当模型在训练的时候，`prefetch` 使数据集在后台取得 batch。
image_label_ds = image_label_ds.prefetch(tf.data.AUTOTUNE)

test_filenames, test_landmarks, attributes,euler_angles = gen_data(val_label_file)
test_filenames = test_filenames[:100]
test_landmarks = test_landmarks[:100]

path_ds = tf.data.Dataset.from_tensor_slices(test_filenames)
val_image_ds = path_ds.map(load_and_preprocess_image)
val_label_ds = tf.data.Dataset.from_generator(preprocess_mark,output_types=tf.float32,output_shapes=(HEATMAP_SIZE, HEATMAP_SIZE, NUM_MARKS),args=[test_landmarks])

val_image_label_ds = tf.data.Dataset.zip((val_image_ds, val_label_ds))
val_image_label_ds = val_image_label_ds.batch(BATCH_SIZE)

#val_image_label_ds = val_image_label_ds.prefetch(tf.data.AUTOTUNE)

def top_k_indices(x, k):
    """Returns the k largest element indices from a numpy array. You can find
    the original code here: https://stackoverflow.com/q/6910641
    """
    flat = x.flatten()
    indices = np.argpartition(flat, -k)[-k:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, x.shape)


def get_peak_location(heatmap, image_size=(256, 256)):
    """Return the interpreted location of the top 2 predictions."""
    h_height, h_width = heatmap.shape
    [y1, y2], [x1, x2] = top_k_indices(heatmap, 2)
    x = (x1 + (x2 - x1)/4) / h_width * image_size[0]
    y = (y1 + (y2 - y1)/4) / h_height * image_size[1]

    return int(x), int(y)

def parse_heatmaps(heatmaps, image_size):
    # Parse the heatmaps to get mark locations.
    marks = []
    heatmaps = np.transpose(heatmaps, (2, 0, 1))
    for heatmap in heatmaps:
        marks.append(get_peak_location(heatmap, image_size))

    # Show individual heatmaps stacked.
    #heatmap_grid = np.hstack(heatmaps[:8])
    #for row in range(1, 12, 1):
    #    heatmap_grid = np.vstack(
    #        [heatmap_grid, np.hstack(heatmaps[row:row+8])])

    return np.array(marks), None


def test(filename,landmark,landmark_pre=None):
    img = cv2.imread(filename)
    return test_img(img,landmark,landmark_pre)

def test_img(img,landmark,landmark_pre=None):
    img = cv2.resize(img,(FILE_WIDTH,FILE_HEIGHT))
    for mark in landmark:
        cv2.circle(img, tuple(mark.astype(int)),1,(0,0,255))
    if landmark_pre is None:
        ...
    else:
        for mark in landmark_pre:
            cv2.circle(img, tuple(mark.astype(int)),1,(0,255,0))
    #cv2.imshow(filename, img)
    return img

#将normal_marks转为指定尺寸的marks
def transpose_marks(landmarks):
    results = []
    for mark in landmarks:
        mark = mark[:2]*(FILE_WIDTH,FILE_HEIGHT)
        results.append(mark)
    return results

if __name__ == "__main__":
    index = 0#random.randint(0,len(test_filenames)-1)
    img = test(test_filenames[index],transpose_marks(test_landmarks[index]))
    cv2.imshow(test_filenames[index],img)
    cv2.waitKey()

    for index,(img,label) in enumerate(image_label_ds.take(1)):
        marks,_ = parse_heatmaps(label[0].numpy(),(FILE_WIDTH,FILE_HEIGHT))
        img = test_img(cv2.cvtColor(img[0].numpy(),cv2.COLOR_RGB2BGR),marks)
        cv2.imshow(train_filenames[index],img)
        cv2.waitKey()