import tensorflow as tf

def save_model(save_path):
    saver = tf.train.Saver()
    saver.save(sess, save_path)

v1 = tf.Variable(1.32, name="v1")
v2 = tf.Variable(1.33, name="v2")
init = tf.global_variables_initializer()
save_path="./save/model"
sess = tf.Session()
sess.run(init)
print("v1 : {}".format(sess.run(v1)))
print("v2 : {}".format(sess.run(v2)))
save_model(save_path) 
