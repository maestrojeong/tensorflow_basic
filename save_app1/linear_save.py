import tensorflow as tf

x_data = [1., 2., 3., 4.]
y_data = [2., 4., 6., 8.]

W = tf.Variable(tf.random_uniform([1], -1., 1.), name = "W")
b = tf.Variable(tf.random_uniform([1], -1., 1.), name = "b")

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

hypothesis = W * X + b

tf.add_to_collection('hypo', hypothesis)
tf.add_to_collection('input', X)
tf.add_to_collection('input', Y)
tf.add_to_collection('vars', W)
tf.add_to_collection('vars', b)

cost = tf.reduce_mean(tf.square(hypothesis - Y))

train = tf.train.GradientDescentOptimizer(1e-3).minimize(cost) 

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# fit the line
for step in range(2001):
    sess.run(train, feed_dict={X: x_data, Y: y_data})
    if step % 100 == 0:
        print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W), sess.run(b))

saver = tf.train.Saver()
saver.save(sess, './save/linear')

print(sess.run(hypothesis, feed_dict={X: 5}))
print(sess.run(hypothesis, feed_dict={X: 2.5}))
