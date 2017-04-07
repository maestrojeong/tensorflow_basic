from utils import *

x_data = [1., 2., 3., 4.]
y_data = [2., 4., 6., 8.]

W = tf.Variable(tf.random_uniform([1], -100., 100.), name = 'weights')
b = tf.Variable(tf.random_uniform([1], -100., 100.), name = 'biases')

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

hypothesis = W * X + b

cost = tf.reduce_mean(tf.square(hypothesis - Y))

Utility.print_keys("variables")
Utility.print_keys("trainable_variables")
# hypothesis, cost is just operation
sys.exit()

