from utils import *

mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)
x = tf.placeholder(tf.float32, shape=[None, 784], name = 'x')
y_ = tf.placeholder(tf.float32, shape=[None, 10], name = 'y_')

x_image = tf.reshape(x, [-1,28,28,1])
h_pool2 = my_image_filter(x_image)

tf.add_to_collection("conv2", h_pool2)
tf.add_to_collection("input", x)

with tf.variable_scope("var1"):
    result1= tf.reshape(h_pool2, [-1, 5*5*50])
    y_temp = tf.nn.relu(fully_connected_layer(result1 ,5*5*50, 500))
with tf.variable_scope("var2"):
    y_hat = tf.nn.softmax(fully_connected_layer(y_temp, 500, 10))

cross_entropy = - tf.reduce_sum(y_*tf.log(y_hat))
train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_hat,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

epoch = 2 
for j in range(epoch):
    for i in range(550):
        batch = mnist.train.next_batch(100)
        train_step.run(feed_dict={x: batch[0], y_: batch[1]})
        if i%50 == 49:
            train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1]})
            print("train_accuracy : {}".format(train_accuracy))
            
                    
test_accuracy = accuracy.eval(feed_dict={x: mnist.test.images, y_:mnist.test.labels})
print("test accuracy = {}".format(test_accuracy))
saver = tf.train.Saver()
saver.save(sess, './save/my-model')
#saver.export_meta_graph(filename = './save/my-model.meta')
