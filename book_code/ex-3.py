'''Artificial neural network and noisy data with console output from p. 41 and p. 44 of the book.'''

import tensorflow as tf
import numpy as np

'''Adds a layer to the artificial neural network.'''
def add_layer(inputs,in_size,out_size,activation_function=None):
    with tf.name_scope("layer"):
        with tf.name_scope("weights"):
            Weights = tf.Variable(tf.random_normal([in_size,out_size]))
    with tf.name_scope("biases"):
            biases = tf.Variable(tf.zeros([1,out_size]) + 0.1)
    with tf.name_scope("Wx_plus_b"):
        Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

'''Runs p. 41 artificial neural network code with p. 44 noisy data.'''
def main():
    #Define placeholders for network inputs
    with tf.name_scope("inputs"):
        xs = tf.placeholder(tf.float32, [None,1],name="x_input")
        ys = tf.placeholder(tf.float32, [None,1],name="y_input")

    #Hidden Layer
    l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)

    #Output Layer
    prediction = add_layer(l1, 10, 1, activation_function=None)

    #Calculating loss
    with tf.name_scope("loss"):
        loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),reduction_indices=[1]))
    with tf.name_scope("train"):
        train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    sess = tf.Session()
    writer = tf.summary.FileWriter("logs/", sess.graph)
    init = tf.global_variables_initializer()

    sess.run(init)

    # P44 Noisy data

    x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
    noise = np.random.normal(0, 0.05, x_data.shape)
    y_data = (x_data)**2 + noise

    for i in range(1000):
        sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
        if i % 50 == 0:
            print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))

main()
