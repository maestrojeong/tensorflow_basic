import tensorflow as tf
import sys 

class Utility:
    @staticmethod
    def print_keys(string):
        print("Collection name : {}".format(string))
        i = 0
        while True:
            try:
                print(tf.get_collection(string)[i])
                i+=1
            except IndexError:
                break;

