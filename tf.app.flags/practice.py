import tensorflow as tf

flags = tf.app.flags
conf = flags.FLAGS
flags.DEFINE_float("var1",0.01,"Document")

print(conf.var1)
print(conf.__dict__)
print(conf.__flags)

'''
0.01
{'__flags': {'var1': 0.01}, '__parsed': True}
{'var1': 0.01}
'''
