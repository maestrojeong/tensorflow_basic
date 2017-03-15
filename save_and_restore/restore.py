import tensorflow as tf

v1 = tf.Variable(1.0, name="v3")
v2 = tf.Variable(2.0, name="v4")
#v3 = tf.Variable(1.0, name="v5")
saver = tf.train.Saver()

with tf.Session() as sess:
    save_path="./save/model"
    sess.run(tf.global_variables_initializer())
    print(sess.run(v1))
    print(sess.run(v2))
    saver.restore(sess, save_path)
    print(sess.run(v1))
    print(sess.run(v2))
    print("Model restored.")

