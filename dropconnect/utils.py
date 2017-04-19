import tensorflow as tf
import numpy as np
import sys

def dropconnect_wrapper(w, prob):
    selector = tf.sign(prob - tf.random_uniform(get_size(w)
                                            ,minval = 0
                                            , maxval=1
                                            , dtype = tf.float32))

    selector = (selector + 1)/2

    return selector*w

def get_size(w):
    return w.get_shape().as_list()

def sample(prob):
    '''
        input :
            prob 2D tensor
        return:
            sample 1 accroding to the probability 
    '''
    return (tf.sign(prob - tf.random_uniform(prob.get_shape(),minval = 0, maxval=1, dtype = tf.float32)) + 1)/2



def print_variables(keys):
    i = 0
    print(keys)
    while True:
        try:
            print(tf.get_collection(keys)[i])
            i+=1
        except IndexError:
            break;
