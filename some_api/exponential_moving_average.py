import tensorflow as tf
import sys

x_data = [1., 2., 3.]
y_data = [1., 2., 3.]

w = tf.Variable(1.0,name ='w')
b = tf.Variable(0.1,name ='b')

hypothesis = w * x_data + b
cost = tf.reduce_mean(tf.square(hypothesis - y_data))

def print_variables(keys):
    i = 0
    print(keys)
    while True:
        try:
            print(tf.get_collection(keys)[i])
            i+=1
        except IndexError:
            break;

print_variables("trainable_variables")
train = tf.train.GradientDescentOptimizer(0.01).minimize(loss = cost, var_list = [w, b])

ema = tf.train.ExponentialMovingAverage(decay = 1.0)

temp = ema.apply([w, b])
print_variables("moving_average_variables")

sess = tf.Session()
sess.run(tf.global_variables_initializer())

with tf.control_dependencies([train]):
    train_wrap = tf.group(temp)

print(ema.average(w))
print(ema.average_name(w))
print(ema.average(b))
print(ema.average_name(b))
print(ema.variables_to_restore())

for step in range(10):
    sess.run(train_wrap)
    print("{} step : {} and {}".format(step, sess.run(w),sess.run(b)))
    print("{} step : {} and {}".format(step, sess.run(ema.average(w)), sess.run(ema.average(b))))
