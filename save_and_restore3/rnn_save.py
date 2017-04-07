import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.python.ops import rnn, rnn_cell
import sys

# learning data
rnn_size = 1
train_steps = 10000
test_data_size = 15

data_size = 10
num_data = 100

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
x_data = np.reshape(x_data, [num_data,data_size,1])
y_data = np.array(y_data, dtype = np.float32)
y_data = np.reshape(y_data,[-1,1])

#print(x_data)
#print(y_data)
print(x_data.shape)
print(y_data.shape)

test_x_data = []
for i in range(3,3+test_data_size):
    test_x_data.append(i)

test_x_data = np.array(test_x_data, dtype = np.float32)
test_x_data = np.reshape(test_x_data, [-1,test_data_size,1])

print(test_x_data.shape)

# train
train_x = tf.placeholder('float', [None, data_size,1 ])
train_y = tf.placeholder('float', [None,1])
test_x = tf.placeholder('float', [1,test_data_size,1])
test_y = tf.placeholder('float', [1]) 

train_x_temp = tf.transpose(train_x, [1,0,2])
train_x_temp = tf.reshape(train_x_temp, [-1,1])
train_x_temp =tf.split(0, data_size, train_x_temp)

test_x_temp = tf.transpose(test_x, [1,0,2])
test_x_temp = tf.reshape(test_x_temp, [-1,1])
test_x_temp =tf.split(0, test_data_size, test_x_temp)

layer = {'weights':tf.Variable(tf.random_normal([rnn_size, 1])),
            'biases':tf.Variable(tf.random_normal([1]))}
 
with tf.variable_scope("rnn") as scope:
    lstm_cell = rnn_cell.LSTMCell(rnn_size)
    train_outputs, train_states = rnn.rnn(lstm_cell, train_x_temp, dtype=tf.float32) 
    train_output = tf.matmul(train_outputs[-1],layer['weights']) + layer['biases']
    scope.reuse_variables()
    test_outputs, test_states = rnn.rnn(lstm_cell, test_x_temp, dtype=tf.float32) 
    test_output = tf.matmul(test_outputs[-1],layer['weights']) + layer['biases']


tf.add_to_collection('test_input', test_x)
tf.add_to_collection('test_input', test_y)
tf.add_to_collection('hypothesis',test_output)

error = tf.reduce_mean(tf.square(train_output-train_y))
optimizer = tf.train.AdamOptimizer().minimize(error)

saver = tf.train.Saver()
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(train_steps+1):
    a,c = sess.run([optimizer, error],feed_dict = {train_x : x_data, train_y : y_data})
    if i%500==0:
        print(c)

print(sess.run(train_output, feed_dict = {train_x : x_data, train_y : y_data}))

print("-----------------------------------------------")
print(test_x_data)
print(sess.run(test_output, feed_dict = {test_x : test_x_data}))
saver.save(sess, 'rnn')
#print(sess.run(train_output, feed_dict = {train_x : x_data, train_y : y_data}))
