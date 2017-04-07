import tensorflow as tf

x_data = [1., 2., 3., 4.]
y_data = [2., 4., 6., 8.]

sess = tf.Session()

saver = tf.train.import_meta_graph('./save/linear.meta')
print(tf.train.latest_checkpoint('./save/'))
#saver.restore(sess, './save/model')
saver.restore(sess, tf.train.latest_checkpoint('./save/'))

hypothesis =  tf.get_collection('hypo')[0]

print(tf.get_collection('vars'))

input_vars = tf.get_collection('input')
X = input_vars[0]
Y = input_vars[1]

print(sess.run(hypothesis, feed_dict={X: 5}))
print(sess.run(hypothesis, feed_dict={X: 2.5}))

sess.close()
