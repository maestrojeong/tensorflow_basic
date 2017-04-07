import tensorflow as tf

def master_initializer():
    uninitailized_variables=[] 
    for v in tf.global_variables():
        try :
            sess.run(v)
        except tf.errors.FailedPreconditionError:
            uninitailized_variables.append(v)
    return tf.variables_initializer(uninitailized_variables)

v3 = tf.Variable(1.0, name="v3", trainable=False )
v4 = tf.Variable(2.0, name="v4")
print("Before restoration")
print("tf.global_variables()")
print(tf.global_variables())
print("tf.trainable_variables()")
print(tf.trainable_variables())
print('tf.get_collection("variables")')
print(tf.get_collection("variables"))
print('tf.get_collection("trainable_variable")')
print(tf.get_collection("trainable_variables"))

print("tf.GraphKeys.GLOBAL_VARIABLES : {}".format(tf.GraphKeys.GLOBAL_VARIABLES))
print("tf.GraphKeys.TRAINABLE_VARIABLES : {}".format(tf.GraphKeys.TRAINABLE_VARIABLES))

print("After restoration")
sess = tf.Session()
saver = tf.train.import_meta_graph('./save/model.meta')
print("trainable variables")
print(tf.trainable_variables())

sess.run(master_initializer())
saver.restore(sess, "./save/model")
print("v3 : {}".format(sess.run(v3)))
print("v4 : {}".format(sess.run(v4)))
print([v.name for v in tf.trainable_variables()])

for v in tf.global_variables():
    print(v)
