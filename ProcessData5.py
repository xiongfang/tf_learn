import cv2
import os
import pathlib
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.keras as keras
import time
import WFLWDataSet3D as WFLWDataSet
from tensorflow.keras.callbacks import TensorBoard
from callbacks import LogImages
from network import hrnet_v2

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

input_shape = (IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNEL)

#model = hrnet_v2(input_shape=input_shape, output_channels=WFLWDataSet.NUM_MARKS,
#                    width=10, name=name)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNEL)),
    #tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2),strides = (2,2)),
    tf.keras.layers.Conv2D(64, (3, 3),strides = (1,1), activation='relu'),
    #tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(64, (6, 6),strides = (1,1), activation='relu'),
    #tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(WFLWDataSet.NUM_MARKS, (1, 1),strides = (1,1), activation=None),
    #tf.keras.layers.BatchNormalization(),

    #tf.keras.layers.Flatten(),
    #tf.keras.layers.Dense(1024, activation = 'relu',kernel_regularizer = tf.keras.regularizers.L2(0.00001)),
    #tf.keras.layers.BatchNormalization(),
    #tf.keras.layers.Dense(196, activation = None)
    ])

model.summary()

def loss_fn(y_true,y_pre):
    return tf.reduce_mean(tf.square(y_true - y_pre))

def test():
    for index,(img,label) in enumerate(train_ds):
        marks,_ = WFLWDataSet.parse_heatmaps(label[0].numpy(),(WFLWDataSet.FILE_WIDTH,WFLWDataSet.FILE_HEIGHT))
        heatmaps = model(img,training=False)  # Logits for this minibatch
        print(np.sum(loss_fn(label,heatmaps)))
        marks_pre, _ = WFLWDataSet.parse_heatmaps(heatmaps[0], (WFLWDataSet.FILE_WIDTH,WFLWDataSet.FILE_HEIGHT))
        img = WFLWDataSet.test_img(cv2.cvtColor(img[0].numpy(),cv2.COLOR_RGB2BGR),marks,marks_pre)
        cv2.imshow("%d"%index,img)


        cv2.waitKey()

'''
    filenames = WFLWDataSet.train_filenames
    landmarks = WFLWDataSet.train_landmarks
    index = random.randint(0,len(filenames)-1)
    filename = filenames[index]
    f = WFLWDataSet.load_and_preprocess_image(filename)
    heatmaps = model.predict(tf.expand_dims(f,0))[0]  # Logits for this minibatch
    marks, _ = WFLWDataSet.parse_heatmaps(heatmaps, (WFLWDataSet.FILE_WIDTH,WFLWDataSet.FILE_HEIGHT))
    img = WFLWDataSet.test(filename,WFLWDataSet.transpose_marks(landmarks[index]),marks)
    cv2.imshow(filename,img)
'''


# Model built. Restore the latest model if checkpoints are available.
latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
if latest_checkpoint:
    print("Checkpoint found: {}, restoring...".format(latest_checkpoint))
    model.load_weights(latest_checkpoint)
    print("Checkpoint restored: {}".format(latest_checkpoint))
else:
    print("Checkpoint not found. Model weights will be initialized randomly.")


test()
cv2.waitKey()

# Schedule the learning rate with (epoch to start, learning rate) tuples
schedule = [(1, 0.001),
            (30, 0.0001),
            (50, 0.00001)]

# Save a checkpoint. This could be used to resume training.
callback_checkpoint = keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(checkpoint_dir, name),
    save_weights_only=True,
    verbose=1,
    save_best_only=False)

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

opt = tf.keras.optimizers.Adam(0.001)
# Train
#model.compile(optimizer=opt,
#              #loss=tf.keras.losses.MSE,
#              loss = loss,
#              metrics=[loss])

#history = model.fit(train_ds, epochs=100
#                    #,steps_per_epoch=steps_per_epoch
#                    ,validation_data=val_ds
#                    ,verbose=1
#                    ,callbacks=callbacks
#                    #,validation_steps=1
#                    )



epochs = 100
#step_count = 200
summary_writer = tf.summary.create_file_writer(log_dir)

for epoch in range(epochs):
    print("\nStart of epoch %d" % (epoch,))

    # Iterate over the batches of the dataset.
    for step, (x_batch_train, y_batch_train) in enumerate(train_ds):

        # Open a GradientTape to record the operations run
        # during the forward pass, which enables auto-differentiation.
        with tf.GradientTape() as tape:

            # Run the forward pass of the layer.
            # The operations that the layer applies
            # to its inputs are going to be recorded
            # on the GradientTape.
            logits = model(x_batch_train, training=True)  # Logits for this minibatch

            # Compute the loss value for this minibatch.
            loss_value = loss_fn(y_batch_train, logits)

        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        grads = tape.gradient(loss_value, model.trainable_weights)

        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        opt.apply_gradients(zip(grads, model.trainable_weights))

        
        # Log every 200 batches.
        #if step % 200 == 0:
        print(
            "Training loss (for one batch) at step %d: %.6f"
            % (step, float(np.sum(loss_value)))
        )
        print("Seen so far: %s samples" % ((step + 1) * WFLWDataSet.BATCH_SIZE))
        with summary_writer.as_default():
            tf.summary.scalar('loss', float(np.sum(loss_value)), epoch * WFLWDataSet.BATCH_SIZE+step)
        #if step>step_count:
        #    break
        

model.save_weights(checkpoint_dir+"/"+name)
print("model saved")

test()
cv2.waitKey()
