import tensorflow as tf

flags = tf.app.flags
conf = flags.FLAGS
flags.DEFINE_float("var1",0.01,"Document")

print(conf.var1)
print(conf.__dict__)

#print(tf.get_default_graph().as_graph_def())
