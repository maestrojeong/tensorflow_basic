from utils import *
import sys
tf.reset_default_graph()

g1 = tf.Graph()
with g1.as_default() as g:
    with g.name_scope( "g1" ) as scope:
        matrix1 = tf.constant([[3., 3.]], name = 'matrix1')
        matrix2 = tf.constant([[2.],[2.]], name = 'matrix2')
        product = tf.matmul(matrix1, matrix2, name = "product")

tf.reset_default_graph()

g2 = tf.Graph()
with g2.as_default() as g:
    with g.name_scope( "g2" ) as scope:
        matrix1 = tf.constant([[4., 4.]], name = 'matrix1')
        matrix2 = tf.constant([[5.],[5.]], name = 'matrix2')
        product = tf.matmul(matrix1, matrix2, name = "product")

tf.reset_default_graph()
#print("Default graph")
#print(tf.get_default_graph().as_graph_def())
print("Graph g1")
print(g1.as_graph_def())
#Utility.print_graph_properties(g1)
sys.exit()

#print(product)
Utility().print_nodes(g1)
Utility().print_nodes(g2)
with tf.Session(graph = g1)as sess:
    product = g1.get_tensor_by_name("g1/product:0")
    print(sess.run(product))
    #print(matrix1.name)
    print(product.name)
