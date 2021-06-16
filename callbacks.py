"""A module containing custom callbacks."""
import cv2
import tensorflow as tf
from tensorflow import keras
import numpy as np
import random

import WFLWDataSet3D as WFLWDataSet

class LogImages(keras.callbacks.Callback):
    def __init__(self, logdir):
        super().__init__()
        self.file_writer = tf.summary.create_file_writer(logdir)
    def on_epoch_end(self, epoch, logs={}):

        index = random.randint(0,len(WFLWDataSet.test_filenames)-1)
        filename = WFLWDataSet.test_filenames[index]
        landmark = WFLWDataSet.test_landmarks[index]

        self.filename = filename
        img = WFLWDataSet.cv_load_and_process_image(filename)
        self.true_heatmap = WFLWDataSet.generate_heatmaps(img,landmark) #(64,64,1)
        self.true_heatmap = np.expand_dims(self.true_heatmap,-1)
        self.true_heatmap = np.expand_dims(self.true_heatmap,0) #(1,64,64,1)
        #self.landmark = WFLWDataSet.transpose_marks(landmark)

        # Do prediction.
        f = WFLWDataSet.load_and_preprocess_image(self.filename)
        heatmaps = self.model.predict(tf.expand_dims(f,0)) # Logits for this minibatch
        #marks, _ = WFLWDataSet.parse_heatmaps(heatmaps, (WFLWDataSet.FILE_WIDTH,WFLWDataSet.FILE_HEIGHT))
        #image = WFLWDataSet.test(self.filename,self.true_heatmap[0],heatmaps)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #image_normal = image/255.0
        with self.file_writer.as_default():
            # tf.summary needs a 4D tensor
            #img_tensor = tf.expand_dims(image, 0)
            #heatmaps = tf.transpose(heatmaps,(2,0,1)) #(2,64,64)
            #heatmaps = tf.expand_dims(heatmaps,-1) #(2,64,64,1)
            #tf.summary.image("test-sample", heatmaps, step=epoch)
            tf.summary.image("pre", heatmaps, step=epoch)
            tf.summary.image("true", self.true_heatmap, step=epoch)
