"""A module containing custom callbacks."""
import cv2
import tensorflow as tf
from tensorflow import keras

import WFLWDataSet3D as WFLWDataSet

class LogImages(keras.callbacks.Callback):
    def __init__(self, logdir, filename,landmark):
        super().__init__()
        self.file_writer = tf.summary.create_file_writer(logdir)
        self.filename = filename
        self.landmark = WFLWDataSet.transpose_marks(landmark)
    def on_epoch_end(self, epoch, logs={}):
        # Do prediction.
        f = WFLWDataSet.load_and_preprocess_image(self.filename)
        heatmaps = self.model.predict(tf.expand_dims(f,0))[0]  # Logits for this minibatch
        marks, _ = WFLWDataSet.parse_heatmaps(heatmaps, (WFLWDataSet.FILE_WIDTH,WFLWDataSet.FILE_HEIGHT))
        image = WFLWDataSet.test(self.filename,self.landmark,marks)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_normal = image/255.0
        with self.file_writer.as_default():
            # tf.summary needs a 4D tensor
            img_tensor = tf.expand_dims(image_normal, 0)
            tf.summary.image("test-sample", img_tensor, step=epoch)
