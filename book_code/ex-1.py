'''Tensorflow operations with session from p. 36 of tbe book.'''

import tensorflow as tf

def main():
    a = tf.add(5,3)
    b = tf.multiply(a,4)
    c = tf.add(b,3)

    sess = tf.Session()
    writer = tf.summary.FileWriter("logs/", sess.graph)
    init = tf.global_variables_initializer()

    sess.run(init)

main()
