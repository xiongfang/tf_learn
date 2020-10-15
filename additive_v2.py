import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import TensorBoard
import random
import time
import numpy as np


class MyModel(tf.keras.Model):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.ags = tf.Variable(tf.random.uniform([2],0,1.0))

    def call(self,x):
        return tf.reduce_sum( tf.multiply(self.ags,x),1)



model = MyModel()

print("Variables:", model.variables)

model.compile(
    # By default, fit() uses tf.function().  You can
    # turn that off for debugging, but it is on now.
    run_eagerly=False,

    # Using a built-in optimizer, configuring as an object
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),

    # Keras comes with built-in MSE error
    # However, you could use the loss function
    # defined above
    loss=tf.keras.losses.mean_squared_error,
)

NUM_EXAMPLES = 1000

# A vector of random x values
x = []
y = []
# Calculate y
for i in range(NUM_EXAMPLES):
    test_a = random.uniform(0,10)
    test_b = random.uniform(0,10)
    x.append([test_a,test_b])
    y.append(x[i][0]+x[i][1])

timeStamp = str(int(time.time()))
tbCallBack = TensorBoard( log_dir='E:/Logs/'+timeStamp+'/')

model.fit(x, y, epochs=10, batch_size=1000,callbacks=[tbCallBack])

#测试
print(model([[3,5],[4,7],[23,7]]))
