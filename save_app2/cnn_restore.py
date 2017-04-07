import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import sys

mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)

def conv_relu(input, kernel_shape, pool_shape, bias_shape):
    w = tf.Variable(tf.random_normal(kernel_shape,stddev=0.1),name='weights')
    b = tf.Variable(tf.constant(0.1, shape = bias_shape), name = 'biases')
    
    tf.add_to_collection("local", w)
    tf.add_to_collection("local", b)
    
    conv = tf.nn.conv2d(input, w, strides = [1,1,1,1]
                       ,padding = 'VALID')
    relu = tf.nn.relu(conv + b)
    pool = tf.nn.max_pool(relu, ksize=pool_shape,
                          strides=pool_shape, padding='SAME')
    return pool

def fully_connected_layer(input, input_size, output_size):
    w2 = tf.Variable(tf.random_normal([input_size, output_size], stddev=0.1),name='fc_weights')
    b2 = tf.Variable(tf.constant(0.1,shape = [output_size]), name = "fc_biases")
    tf.add_to_collection("local", w2)
    tf.add_to_collection("local", b2)
    return tf.matmul(input, w2) + b2

sess = tf.InteractiveSession()
new_saver = tf.train.import_meta_graph('./save/my-model.meta')
new_saver.restore(sess, './save/my-model')

x = tf.get_collection('input')[0]
h_pool2 = tf.get_collection('conv2')[0]

y_ = tf.placeholder(tf.float32, shape=[None,10])

def my_image_filter(input_images):
    with tf.variable_scope("conv3"):
        conv3 = conv_relu(input_images,[2,2,50,100],[1,2,2,1],[100])
    return conv3

result1= tf.reshape(my_image_filter(h_pool2), [-1, 2*2*100])
y_hat = tf.nn.softmax(fully_connected_layer(result1 ,2*2*100, 10))

with tf.variable_scope("final"):
    error = - tf.reduce_sum(y_*tf.log(y_hat))
    train = tf.train.AdamOptimizer(1e-4).minimize(loss = error, var_list = tf.get_collection("local"))
    prediction = tf.equal(tf.argmax(y_hat,1), tf.argmax(y_,1))
    correct = tf.reduce_mean(tf.cast(prediction, tf.float32))
'''
i = 0 
while True:
    try:
        print(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)[i]) 
        i+=1
    except IndexError:
        break;
i = 0 
while True:
    try:
        print(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)[i]) 
        i+=1
    except IndexError:
        break;
'''
# tf.global_variables_initailizer() = tf.variables_initializer(tf.get_collection(tf.GraphKeys.GLOBAL_VARIBLES)
                                    #tf.variables_initailizer(tf.global_variables())

sess.run(tf.variables_initializer(tf.get_collection("local")))
sess.run(tf.variables_initializer([v for v in tf.get_collection("variables") if v.name.startswith("final")]))
epoch = 1
for j in range(epoch):
    for i in range(550):
        batch = mnist.train.next_batch(100)
        train.run(feed_dict={x: batch[0], y_: batch[1]})
        if i%50 == 49:
            train_accuracy = correct.eval(feed_dict={x:batch[0], y_: batch[1]})
            print("train_accuracy : {}".format(train_accuracy))
                    
test_accuracy = correct.eval(feed_dict={x: mnist.test.images, y_:mnist.test.labels})
print("test accuracy = {}".format(test_accuracy))

