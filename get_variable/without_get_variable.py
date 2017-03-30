from utils import *

x_data = [1.]
y_data = [2.]

var_dict = {
        'weights' :  tf.Variable(tf.zeros([1]), name = 'weights')
        ,'biases' : tf.Variable(tf.zeros([1]), name = 'biases')
        }

def linear(x, variable_dict):
    return x*variable_dict['weights']+variable_dict['biases']    

x = tf.placeholder(tf.float32, shape = [None])
y = tf.placeholder(tf.float32, shape = [None])

hypothesis = linear(linear(x, var_dict), var_dict)

cost = tf.reduce_mean(tf.square(hypothesis - y))
train = tf.train.GradientDescentOptimizer(1).minimize(cost)

util = Utility()
util.print_keys("trainable_variables")
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(10):
    print(sess.run([var_dict['weights'], var_dict['biases']]))
    _, error = sess.run([train, cost], feed_dict={x : x_data, y : y_data})
    print("cost : %.4f"%error)
