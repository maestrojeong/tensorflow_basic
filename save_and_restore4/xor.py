import tensorflow as tf
import numpy as np
import sys

xy = np.loadtxt('xor.txt',unpack=True, dtype = 'float32')
x_train = np.transpose(xy[:-1])
y_train = np.reshape(xy[-1],[4,1])
print(x_train)
print(y_train)

sess = tf.Session()

new_saver = tf.train.import_meta_graph('./save/and.meta')
new_saver.restore(sess, './save/and')

x1 = tf.get_collection('input')[0]
temp1 = tf.get_collection('and')[0]

sess2 = tf.Session()
#new_saver2 = tf.train.import_meta_graph('./save/nand.meta')
#new_saver2.restore(sess2, './save/nand')
#x2 = tf.get_collection('input2')[0]
#temp2 = tf.get_collection('nand')[0]

#temp = tf.stack(temp1,temp2)

#print(temp)

sys.exit()

i = 0 
while True:
    try:
        print(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)[i]) 
        i+=1
    except IndexError:
        break;

sess.run(tf.variables_initializer(tf.get_collection("local")))

epoch = 1
for j in range(epoch):
    for i in range(550):
        batch = mnist.train.next_batch(100)
        train_step.run(feed_dict={x: batch[0], y_: batch[1]})
        if i%50 == 49:
            train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1]})
            print("train_accuracy : {}".format(train_accuracy))
                    
test_accuracy = accuracy.eval(feed_dict={x: mnist.test.images, y_:mnist.test.labels})
print("test accuracy = {}".format(test_accuracy))

