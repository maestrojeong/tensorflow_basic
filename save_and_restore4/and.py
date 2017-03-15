import tensorflow as tf
import numpy as np
import sys

xy = np.loadtxt('and.txt',unpack=True, dtype = 'float32')
x_train = np.transpose(xy[:-1])
y_train = np.reshape(xy[-1],[4,1])
print(x_train)
print(y_train)

x = tf.placeholder(tf.float32, shape=[4, 2], name = 'x')
y = tf.placeholder(tf.float32, shape=[4, 1], name = 'y_')

w = tf.Variable(tf.truncated_normal([2, 1], stddev = 1),name = "and_weight")
b = tf.Variable(tf.ones([1]), name = "and_bias")

y_hat = tf.sigmoid(tf.matmul(x, w) + b)

error = - tf.reduce_sum(y*tf.log(y_hat) + (1- y)*tf.log(1-y_hat))
train_step = tf.train.GradientDescentOptimizer(1e-2).minimize(error)

correct_prediction = tf.equal(tf.floor(y_hat+0.5), y)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

tf.add_to_collection("and", y_hat)
tf.add_to_collection("input", x)
tf.add_to_collection("output", y)

for i in range(1000):
    sess.run(train_step, feed_dict={x: x_train, y : y_train})
    if i%100 == 0:
        print("Error = {}".format(sess.run(error, feed_dict={x : x_train, y: y_train})))

print(sess.run(w))
print(sess.run(b))
print(sess.run(y_hat, feed_dict={x: x_train}))
train_accuracy = sess.run(accuracy, feed_dict={x: x_train, y : y_train})
print("train_accuracy : {}".format(train_accuracy))
saver.save(sess, './save/and')
saver.export_meta_graph(filename = './save/and.meta')

