import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.python.ops import rnn, rnn_cell
import sys

# learning data

data_size = 5 
num_data = 10

x_data = []
y_data = []

normalizer = 1

for i in range(num_data):
    input_temp = []
    for j in range(i,i+data_size):
        input_temp.append(j*normalizer)
    x_data.append(input_temp)
    output_temp = normalizer*(i+data_size)
    y_data.append(output_temp) 

x_data = np.array(x_data, dtype = np.float32)
y_data = np.array(y_data, dtype = np.float32)

y_data = np.reshape(y_data,[-1,1])

print(x_data)
print(y_data)
print(x_data.shape)
print(y_data.shape)


rnn_size = 10
train_steps = 10000

saver = tf.train.Saver()
with tf.Session()as sess:
    saver.restore(sess,"./nn.ckpt")
    print(sess.run(hypothesis, feed_dict = {train_x : x_data, train_y : y_data}))
#print(sess.run(train_output, feed_dict = {train_x : x_data, train_y : y_data}))

