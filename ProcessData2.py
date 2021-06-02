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

path_root = "E:/tf_learn/BioID_Face/data/BioID-FaceDatabase-V1.2"


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

def preprocess_image(image):
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.image.resize(image, [IMAGE_WIDTH, IMAGE_HEIGHT])
  image /= 255.0  # normalize to [0,1] range
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

#ds = ds.repeat()
ds = ds.batch(BATCH_SIZE,drop_remainder = True)

# 当模型在训练的时候，`prefetch` 使数据集在后台取得 batch。
ds = ds.prefetch(tf.data.AUTOTUNE)

model_path_name = "E:/tf_learn/ProcessDataWeights.model"
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(4, activation = 'relu')])


try:
    load_status = model.load_weights(model_path_name)
except:
    print("load model weights failed")



model.summary()

# Instantiate an optimizer to train the model.
optimizer = keras.optimizers.Adam(learning_rate=0.001)
# Instantiate a loss function.
tf_loss_fn = tf.keras.losses.MAE;
def loss_fn(labels,logits):
    v = tf.subtract(labels,logits)
    return tf.reduce_sum(tf.abs(v))

epochs = 20

timeStamp = str(int(time.time()))
summary_writer = tf.summary.create_file_writer('E:/Logs/'+timeStamp+'/')

for epoch in range(epochs):
    print("\nStart of epoch %d" % (epoch,))

    # Iterate over the batches of the dataset.
    for step, (x_batch_train, y_batch_train) in enumerate(ds):

        # Open a GradientTape to record the operations run
        # during the forward pass, which enables auto-differentiation.
        with tf.GradientTape() as tape:

            # Run the forward pass of the layer.
            # The operations that the layer applies
            # to its inputs are going to be recorded
            # on the GradientTape.
            logits = model(x_batch_train, training=True)  # Logits for this minibatch

            # Compute the loss value for this minibatch.
            loss_value = tf_loss_fn(y_batch_train, logits)

        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        grads = tape.gradient(loss_value, model.trainable_weights)

        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        
        # Log every 200 batches.
        #if step % 200 == 0:
        print(
            "Training loss (for one batch) at step %d: %.6f"
            % (step, float(np.sum(loss_value)))
        )
        print("Seen so far: %s samples" % ((step + 1) * BATCH_SIZE))
    with summary_writer.as_default():
        tf.summary.scalar('loss', float(np.sum(loss_value)), epoch + 1)

val_image_label_ds = tf.data.Dataset.zip((val_image_ds, val_label_ds))
val_image_label_ds = val_image_label_ds.batch(1,drop_remainder = True)
# Iterate over the batches of the dataset.
for index,(x_batch_train,y_batch_train) in enumerate(val_image_label_ds):
    logits = model(x_batch_train, training=False)  # Logits for this minibatch
    loss_value = tf_loss_fn(y_batch_train, logits)
    with summary_writer.as_default():
        tf.summary.scalar('val_loss', float(np.sum(loss_value)), index + 1)

    #print("logits[(%.6f,%.6f),(%.6f,%.6f)] true[(%.6f,%.6f),(%.6f,%.6f)]"%(logits[0][0],logits[0][1],logits[0][2],logits[0][3]
    #      ,y_batch_train[0][0],y_batch_train[0][1],y_batch_train[0][2],y_batch_train[0][3]))
model.save_weights(model_path_name)
print("model saved")
