import tensorflow as tf

# Only requires for name to be same
W_r = tf.Variable(tf.random_uniform([1], -1., 1.), name = "W")
b_r = tf.Variable(tf.random_uniform([1], -1., 1.), name = "b")

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

hypothesis = W_r * X + b_r

sess = tf.Session()
saver = tf.train.Saver()

print("tf.train.latest_checkpoint('./save/') = {}".format(tf.train.latest_checkpoint('./save/')))
print("tf.global_variables")
print([v.name for v in tf.global_variables()])

saver.restore(sess, tf.train.latest_checkpoint('./save/'))

print(sess.run(hypothesis, feed_dict={X: 5}))
print(sess.run(hypothesis, feed_dict={X: 2.5}))
