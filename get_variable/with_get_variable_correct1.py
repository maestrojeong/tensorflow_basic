from utils import *

x_data = [1.]
y_data = [2.]

def linear(x):
    w = tf.get_variable('weights', shape = [1], initializer = tf.constant_initializer(0.0))
    b = tf.get_variable('biases', shape = [1], initializer = tf.constant_initializer(0.0))
    return x*w + b

x = tf.placeholder(tf.float32, shape = [None])
y = tf.placeholder(tf.float32, shape = [None])

with tf.variable_scope("layer1"):
    hypo = linear(x)

with tf.variable_scope("layer1", reuse= True):
    hypothesis = linear(linear(x))

cost = tf.reduce_mean(tf.square(hypothesis - y))
train = tf.train.GradientDescentOptimizer(1).minimize(cost)

util = Utility()
util.print_keys("trainable_variables")
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(10):
    with tf.variable_scope("layer1", reuse=True):
        w = tf.get_variable('weights',[1])
        print(sess.run(w))
    _, error = sess.run([train, cost], feed_dict={x : x_data, y : y_data})
    print("cost : %.4f"%error)
