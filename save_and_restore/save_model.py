import tensorflow as tf

v1 = tf.Variable(1.32, name="v3")
v2 = tf.Variable(1.33, name="v4")

init = tf.global_variables_initializer()

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    print(v1.eval(sess))
    print(v2.eval(sess))
    save_path="./save/model"
    saver.save(sess, save_path)
