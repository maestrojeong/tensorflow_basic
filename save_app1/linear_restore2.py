import tensorflow as tf

sess = tf.Session()
saver = tf.train.import_meta_graph('./save/linear.meta')
saver.restore(sess, tf.train.latest_checkpoint('./save/'))

print("tf.global_variables")
print([v.name for v in tf.global_variables()])

print("tf.get_collection('vars')")
print(tf.get_collection('vars'))

input_vars = tf.get_collection('input')
X = input_vars[0]
Y = input_vars[1]

hypothesis =  tf.get_collection('hypo')[0]
print(sess.run(hypothesis, feed_dict={X : 5}))
print(sess.run(hypothesis, feed_dict={X : 2.5}))
sess.close()
