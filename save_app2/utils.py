import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

def conv_relu(input, kernel_shape, pool_shape, bias_shape):
    w = tf.Variable(tf.random_normal(kernel_shape,stddev=0.1),name='weights')
    b = tf.Variable(tf.constant(0.1, shape = bias_shape), name = 'biases')
    
    conv = tf.nn.conv2d(input, w, strides = [1,1,1,1]
                       ,padding = 'VALID')
    relu = tf.nn.relu(conv + b)
    pool = tf.nn.max_pool(relu, ksize=pool_shape,
                          strides=pool_shape, padding='SAME')
    return pool

def my_image_filter(input_images):
    with tf.variable_scope("conv1"):
        conv1 = conv_relu(input_images,[5,5,1,30],[1,2,2,1],[30])
    with tf.variable_scope("conv2"):
        conv2 = conv_relu(conv1,[3,3,30,50],[1,2,2,1],[50])
    return conv2

def fully_connected_layer(input, input_size, output_size):
    w2 = tf.Variable(tf.random_normal([input_size, output_size], stddev=0.1),name='fc_weights')
    b2 = tf.Variable(tf.constant(0.1,shape = [output_size]), name = "fc_biases")
    return tf.matmul(input, w2) + b2

