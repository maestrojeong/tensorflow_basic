import numpy as np
import tensorflow as tf
import pandas as pd
import sys

# learning data
rnn_size = 1
train_steps = 100000
test_data_size = 10

data_size = 10
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
x_data = np.reshape(x_data, [num_data,data_size,1])
y_data = np.array(y_data, dtype = np.float32)
y_data = np.reshape(y_data,[-1,1])

print(x_data)
print(y_data)
print(x_data.shape)
print(y_data.shape)

batch_size = 1

sess = tf.Session()
# train
train_x = tf.placeholder('float', [None, data_size, 1 ])
train_y = tf.placeholder('float', [None, 1])

train_x_temp = tf.transpose(train_x, [1,0,2])
train_x_temp = tf.reshape(train_x_temp, [-1, 1])
train_x_temp =tf.split(train_x_temp, num_or_size_splits=int(data_size), axis=0)
print(sess.run(tf.shape(train_x_temp), feed_dict = {train_x :  x_data}))

layer = {'weights':tf.Variable(tf.random_normal([rnn_size, 1])),
            'biases':tf.Variable(tf.random_normal([1]))}
 
lstm_cell = tf.contrib.rnn.LSTMCell(rnn_size)
with tf.variable_scope("rnn") as scope:
    state = lstm_cell.zero_state(num_data, tf.float32)
    train_outputs = []
    flag = 0
    for input_ in train_x_temp:
        if flag==0:
            flag=1
        else : 
            scope.reuse_variables()
        output, state = lstm_cell(input_, state)
        train_outputs.append(output)
    
    train_output = tf.matmul(train_outputs[-1],layer['weights']) + layer['biases']

error = tf.reduce_mean(tf.square(train_output-train_y))
optimizer = tf.train.AdamOptimizer().minimize(error)

sess.run(tf.global_variables_initializer())
for i in range(train_steps+1):
    a,c = sess.run([optimizer, error],feed_dict = {train_x : x_data, train_y : y_data})
    if i%500==0:
        print("cost = {}".format(c))

print(sess.run(train_output, feed_dict = {train_x : x_data}))

