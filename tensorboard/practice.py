# coding: utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

tf.reset_default_graph()

mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)

batch_size = 100
training_epochs = 5
logs_path = "./log"

with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, shape=[None, 784], name="x-input") 
    y_ = tf.placeholder(tf.float32, shape=[None, 10], name="y-input")

with tf.name_scope("weights"):
    W = tf.Variable(tf.zeros([784, 10]), name = "weights")

with tf.name_scope("biases"):
    b = tf.Variable(tf.zeros([10]), name = "biases")

with tf.name_scope("softmax"):
    y = tf.nn.softmax(tf.matmul(x,W) + b)


with tf.name_scope('cross_entropy'):
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

with tf.name_scope('train'):
    train_op = tf.train.GradientDescentOptimizer(1e-1).minimize(cross_entropy)

with tf.name_scope('Accuracy'):
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tf.summary.scalar("cost_track", cross_entropy)
tf.summary.scalar("accuracy", accuracy)
tf.summary.histogram("weights_hist",W)
tf.summary.histogram("biases_hist",b)

summary_op = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(logs_path, sess.graph)
    
    for epoch in range(training_epochs):
        batch_count = int(mnist.train.num_examples/batch_size)
        for i in range(batch_count):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            _, summary = sess.run([train_op, summary_op], feed_dict={x: batch_x, y_: batch_y})
            writer.add_summary(summary, epoch * batch_count + i)
        if epoch % 5 == 0: 
            print ("Epoch: ", epoch) 
    print ("Accuracy: ", accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
    print ("done")
