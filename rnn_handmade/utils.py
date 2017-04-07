import tensorflow as tf
import numpy as np
import sys 

class Utility:
    def print_keys(self, string):
        print("Collection name : {}".format(string))
        i = 0
        while True:
            try:
                print(tf.get_collection(string)[i])
                i+=1
            except IndexError:
                break;
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
 
