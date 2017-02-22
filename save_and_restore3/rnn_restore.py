import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.python.ops import rnn, rnn_cell
import sys

test_data_size = 15

normalizer = 1

test_x_data = []
for i in range(3,3+test_data_size):
    test_x_data.append(i)

test_x_data = np.array(test_x_data, dtype = np.float32)
test_x_data = np.reshape(test_x_data, [-1,test_data_size,1])

print(test_x_data.shape)

new_saver = tf.train.import_meta_graph('rnn.meta')
sess = tf.Session()
new_saver.restore(sess, tf.train.latest_checkpoint('./'))

all_vars = tf.get_collection('hypothesis')
test_output = all_vars[0]

input_vars = tf.get_collection('test_input')

test_x = input_vars[0]
test_y = input_vars[1]

print(test_x_data)
print(sess.run(test_output, feed_dict = {test_x : test_x_data}))
