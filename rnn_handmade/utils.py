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

