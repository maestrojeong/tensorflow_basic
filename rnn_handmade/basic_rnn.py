from utils import *

# learning data
rnn_size = 1
train_steps = 10000
test_data_size = 10

data_size = 5
num_data = 10

x_data = []
y_data = []

for i in range(num_data):
    input_temp = []
    for j in range(i,i+data_size):
        input_temp.append(j)
    output_temp = i+data_size
    
    x_data.append(input_temp)
    y_data.append(output_temp) 

x_data = np.array(x_data, dtype = np.float32)
x_data = np.transpose(x_data, [1,0])
y_data = np.array(y_data, dtype = np.float32)

print(x_data)
print(y_data)
print(x_data.shape)
print(y_data.shape)

class MaeRNN:
    def __init__(self, size = 10):
        self.rnn_size = size

    def basic_rnn_cell(self, i, s=None):
        '''
        input :
            i = [batch, input_size]
        return :
            [batch, rnn_size]
        '''
        iw = tf.get_variable(name = "input_weights"
                , shape = [i.get_shape()[1], self.rnn_size]
                , initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01))
        if s==None:
            s = tf.Variable(tf.zeros([i.get_shape()[0], self.rnn_size]), trainable = False)

        sw = tf.get_variable(name = "state_weights"
                , shape = [self.rnn_size, self.rnn_size]
                , initializer = tf.constant_initializer(0.0))
        
        b = tf.get_variable(name = "biases"
                , shape = [self.rnn_size]
                , initializer = tf.truncated_normal_initializer(mean = 0.0, stddev = 0.01))
        
        return tf.nn.tanh(tf.matmul(i, iw) + tf.matmul(s,sw) + b)
    
train_x = tf.placeholder('float', [data_size, num_data])
train_y = tf.placeholder('float', [num_data])
rnn_size = 1
temp = MaeRNN(rnn_size)

temp_x = tf.unstack(train_x)

outputs = list()
with tf.variable_scope("rnn") as scope:
    for i in range(data_size):
        if i==0:
            state = temp.basic_rnn_cell(tf.expand_dims(temp_x[i],1))
            outputs.append(state)
        else:
            scope.reuse_variables()
            state = temp.basic_rnn_cell(tf.expand_dims(temp_x[i], 1), state)
            outputs.append(state)

layer = {'weights':tf.Variable(tf.random_normal([rnn_size, 1])),
            'biases':tf.Variable(tf.random_normal([1]))}
prediction = tf.reshape(tf.matmul(outputs[-1], layer['weights'])+ layer['biases'],[-1]) 
error = tf.reduce_mean(tf.square(prediction - train_y))
global_step = tf.Variable(0.0, trainable = False)
learning_rate = tf.train.exponential_decay(learning_rate= 1e-2, global_step= global_step,
                                                decay_steps = 1000, decay_rate = 0.1, staircase=True)
optimizer = tf.train.AdamOptimizer(1e-2).minimize(loss = error, global_step = global_step)

util = Utility()
util.print_keys("trainable_variables")

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(3000):
    _,c = sess.run([optimizer, error],feed_dict = {train_x : x_data, train_y : y_data})
    if i%500==0:
        print(sess.run(learning_rate))
        print("{}th step cost : {}".format(i, c))

print(sess.run(outputs, feed_dict = {train_x : x_data}))
