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
from callbacks import LogImages

train_ds = WFLWDataSet.image_label_ds
val_ds = WFLWDataSet.val_image_label_ds


img = list(train_ds.take(1).as_numpy_iterator())[0][0]

IMAGE_WIDTH, IMAGE_HEIGHT,IMAGE_CHANNEL =  img.shape[-3:]

landmark = list(train_ds.take(1).as_numpy_iterator())[0][1]
print(len(landmark[0]))

name = "hrnetv2"

# Checkpoint is used to resume training.
checkpoint_dir = os.path.join("E:/checkpoints", name)
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
    print("Checkpoint directory created: {}".format(checkpoint_dir))

timeStamp = str(int(time.time()))
log_dir='E:/Logs/'+timeStamp+'/'

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
    tf.keras.layers.Dense(1024, activation = 'relu',kernel_regularizer = tf.keras.regularizers.L2(0.00001)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(196, activation = None)
    ])


model.summary()

def test():
    index = random.randint(0,len(WFLWDataSet.test_filenames)-1)
    filename = WFLWDataSet.test_filenames[index]
    f = WFLWDataSet.load_and_preprocess_image(filename)
    logits = model(tf.expand_dims(f,0), training=False)  # Logits for this minibatch
    img = WFLWDataSet.test(filename,WFLWDataSet.test_landmarks[index],logits[0].numpy())
    cv2.imshow(filename,img)

#test()

# Model built. Restore the latest model if checkpoints are available.
latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
if latest_checkpoint:
    print("Checkpoint found: {}, restoring...".format(latest_checkpoint))
    model.load_weights(latest_checkpoint)
    print("Checkpoint restored: {}".format(latest_checkpoint))
else:
    print("Checkpoint not found. Model weights will be initialized randomly.")


# Schedule the learning rate with (epoch to start, learning rate) tuples
schedule = [(1, 0.001),
            (30, 0.0001),
            (50, 0.00001)]

# Save a checkpoint. This could be used to resume training.
callback_checkpoint = keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(checkpoint_dir, name),
    save_weights_only=True,
    verbose=1,
    save_best_only=True)

callback_tensorboard = TensorBoard(log_dir=log_dir,
                        histogram_freq=1024,
                        write_graph=True,
                        update_freq='batch' #'epoch'
                        )
# Learning rate decay.
#callback_lr = EpochBasedLearningRateSchedule(schedule)

index = random.randint(0,len(WFLWDataSet.test_filenames)-1)

# Log a sample image to tensorboard.
callback_image = LogImages(log_dir, WFLWDataSet.test_filenames[index],WFLWDataSet.test_landmarks[index])

# List all the callbacks.
callbacks = [callback_checkpoint, callback_tensorboard, #callback_lr,
                callback_image]

opt = tf.keras.optimizers.Adam(0.00001)
# Train
model.compile(optimizer=opt,
              loss=tf.keras.losses.MAE,
              metrics=[keras.metrics.mae])

steps_per_epoch = 2 #int(len(WFLWDataSet.train_filenames)/WFLWDataSet.BATCH_SIZE)
history = model.fit(train_ds, epochs=10
                    #,steps_per_epoch=steps_per_epoch
                    ,validation_data=val_ds
                    ,verbose=1
                    ,callbacks=callbacks
                    #,validation_steps=1
                    )


for i in range(4):
    test()
cv2.waitKey()