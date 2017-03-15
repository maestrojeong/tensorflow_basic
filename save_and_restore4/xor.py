import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

xy = np.loadtxt('xor.txt',unpack=True, dtype = 'float32')
x_train = np.transpose(xy[:-1])
y_train = np.reshape(xy[-1],[4,1])
print(x_train)
print(y_train)

sess = tf.Session()

new_saver = tf.train.import_meta_graph('./save/my-model.meta')
new_saver.restore(sess, './save/my-model')
new_saver = tf.train.import_meta_graph('./save/my-model.meta')
new_saver.restore(sess, './save/my-model')

x1 = tf.get_collection('input')[0]
x2 = tf.get_collection(
y1 = tf.get_collection('output')[0]
y2 =
h_pool2 = tf.get_collection('conv2')[0]


i = 0 
while True:
    try:
        print(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)[i]) 
        i+=1
    except IndexError:
        break;

sess.run(tf.variables_initializer(tf.get_collection("local")))

epoch = 1
for j in range(epoch):
    for i in range(550):
        batch = mnist.train.next_batch(100)
        train_step.run(feed_dict={x: batch[0], y_: batch[1]})
        if i%50 == 49:
            train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1]})
            print("train_accuracy : {}".format(train_accuracy))
                    
test_accuracy = accuracy.eval(feed_dict={x: mnist.test.images, y_:mnist.test.labels})
print("test accuracy = {}".format(test_accuracy))

