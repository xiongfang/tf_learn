from nd_mlp_mixer import MLPMixer
import tensorflow as tf
from tensorflow.keras import datasets, layers
import time
from tensorflow.keras.callbacks import TensorBoard

# Load data
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0
train_images, test_images = train_images.astype("float32"), test_images.astype("float32")
height, width = train_images.shape[-2:]
num_classes = 10

# Prepare the model (add channel dimension to images)
inputs = layers.Input(shape=(height, width))
h = layers.Reshape([28, 28, 1])(inputs)
mlp_mixer = MLPMixer(num_classes=10, 
                     num_blocks=2, 
                     patch_size=4, 
                     hidden_dim=28, 
                     tokens_mlp_dim=28,
                     channels_mlp_dim=28)(h)
model = tf.keras.Model(inputs=inputs, outputs=mlp_mixer)
print(model.summary())

timeStamp = str(int(time.time()))
tbCallBack = TensorBoard( log_dir='E:/Logs/'+timeStamp+'/')

# Train
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
history = model.fit(train_images, train_labels, batch_size=64, epochs=10,
                    validation_data=(test_images, test_labels), verbose=2,callbacks=[tbCallBack])