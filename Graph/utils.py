import tensorflow as tf
import sys 

class Utility:
    def print_keys(string):
        print("Collection name : {}".format(string))
        i = 0
        while True:
            try:
                print(tf.get_collection(string)[i])
                i+=1
            except IndexError:
                break;
    def print_nodes(graph):
        print("Graph : {}".format(graph))
        temp = [n.name for n in graph.as_graph_def().node]
        for i in range(len(temp)):
            print(temp[i])
    
    def print_graph_properties(graph):
        print("building_function : {}".format(graph.building_function))
        print("finalized : {}".format(graph.finalized))
        print("graph_def_versions : {}".format(graph.graph_def_versions))
        print("seed : {}".format(graph.seed))
        print("version : {}".format(graph.version))

