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

mnist = input_data.read_data_sets('./MNIST_data', one_hot=True)
x = tf.placeholder(tf.float32, shape=[None, 784], name = 'x')
y_ = tf.placeholder(tf.float32, shape=[None, 10], name = 'y_')

x_image = tf.reshape(x, [-1,28,28,1])
h_pool2 = my_image_filter(x_image)

tf.add_to_collection("conv2", h_pool2)
tf.add_to_collection("input", x)

with tf.variable_scope("var1"):
    result1= tf.reshape(h_pool2, [-1, 5*5*50])
    y_temp = tf.nn.relu(fully_connected_layer(result1 ,5*5*50, 500))
with tf.variable_scope("var2"):
    y_hat = tf.nn.softmax(fully_connected_layer(y_temp, 500, 10))

cross_entropy = - tf.reduce_sum(y_*tf.log(y_hat))
train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_hat,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

epoch = 1 
for j in range(epoch):
    for i in range(10):
        batch = mnist.train.next_batch(100)
        train_step.run(feed_dict={x: batch[0], y_: batch[1]})
        if i%50 == 49:
            train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1]})
            print("train_accuracy : {}".format(train_accuracy))
            
                    
test_accuracy = accuracy.eval(feed_dict={x: mnist.test.images, y_:mnist.test.labels})
print("test accuracy = {}".format(test_accuracy))
saver.save(sess, './save/my-model')
#saver.export_meta_graph(filename = './save/my-model.meta')
