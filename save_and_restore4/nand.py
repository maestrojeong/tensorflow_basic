import tensorflow as tf
import numpy as np
import sys

xy = np.loadtxt('nand.txt',unpack=True, dtype = 'float32')
x_train = np.transpose(xy[:-1])
y_train = np.reshape(xy[-1],[4,1])

sess = tf.Session()
new_saver = tf.train.import_meta_graph('./save/and.meta')
new_saver.restore(sess, './save/and')

print(x_train)
print(y_train)

temp = tf.get_collection("and")[0]
x = tf.get_collection("input")[0]
y = tf.placeholder(tf.float32, shape=[4, 1], name = 'y_')

w = tf.Variable(-5/3*tf.ones([1, 1]),name = "nand_weight")
b = tf.Variable(0.6*tf.ones([1]), name = "nand_bias")

tf.add_to_collection("local", w)
tf.add_to_collection("local", b)

y_hat = tf.sigmoid(tf.matmul(temp, w) + b)

error = - tf.reduce_sum(y*tf.log(y_hat) + (1- y)*tf.log(1-y_hat))
train_step = tf.train.GradientDescentOptimizer(1e-2).minimize(error)

correct_prediction = tf.equal(tf.floor(y_hat+0.5), y)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.variables_initializer(tf.get_collection("local")))

i = 0 
while True:
    try:
        print(sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)[i]))
        i+=1
    except IndexError:
        break;


for i in range(1000):
    sess.run(train_step, feed_dict={x: x_train, y : y_train})
    if i%100 == 0:
        print("Error = {}".format(sess.run(error, feed_dict={x : x_train, y: y_train})))


print(sess.run(w))
print(sess.run(b))
print(sess.run(y_hat, feed_dict={x: x_train}))
train_accuracy = sess.run(accuracy, feed_dict={x: x_train, y : y_train})
print("train_accuracy : {}".format(train_accuracy))

i = 0 
while True:
    try:
        print(sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)[i]))
        i+=1
    except IndexError:
        break;


