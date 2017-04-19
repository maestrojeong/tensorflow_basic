from utils import *

xy = np.loadtxt('input.txt', unpack=True, dtype='float32')
x_data = np.transpose(xy[0:-1])
y_data = xy[-1]

nfeatures = 3

x = tf.placeholder(tf.float32, [None, nfeatures])
y = tf.placeholder(tf.float32, [None])

W = tf.Variable(tf.random_uniform([nfeatures, 1], -1, 1), name = 'weights')

W_wrap = dropconnect_wrapper(W, 0.5)

y_hat = tf.matmul(x, W_wrap)

cost = tf.reduce_mean(tf.square(y_hat - y_data))

train = tf.train.GradientDescentOptimizer(0.1).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

print_variables("trainable_variables")

for step in range(5):
    sess.run(train, feed_dict = {x : x_data, y : y_data})
    print(" w : {}".format(sess.run(W)))
