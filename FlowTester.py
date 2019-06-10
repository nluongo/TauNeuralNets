import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
from sklearn.model_selection import KFold

# Define and open path to the flat file that we are reading Et information from
flat_file_path = os.path.join(os.path.expanduser('~'), 'TauTrigger', 'Formatted Data Files', 'RecoEt_PO_Flat.txt')
flat_file = open(flat_file_path, 'r')

cell_ets = [[2, 0], [0, 2], [1.5, 1.5]]
true_ets = [1], [4], [4]

# Convert Et lists to numpy arrays for use with tensorflow
cell_ets = np.array(cell_ets)
true_ets = np.array(true_ets)

sum_ets = np.array([sum(i) for i in cell_ets])

print(cell_ets)
print(sum_ets)
print(true_ets)

print(true_ets-sum_ets)
shift_et = np.mean(true_ets-sum_ets)
print(shift_et)


# Get total number of events
event_num = 3

lr = 0.1
epochs = 1
num_layers = 2


def train_layerweighted_nn(cell_ets, true_ets, train, test):
    print('Starting training...')

    # Create placeholders for the values that will be input when training
    x = tf.placeholder(tf.float32, [None, num_layers])
    y = tf.placeholder(tf.float32, [None, 1])

    # Create variables for the values that will be calculated when training
    W = tf.Variable(np.array(np.ones([num_layers, 1]), dtype=np.float32))
    b = tf.Variable(tf.zeros([1]))

    # Initialize global variables
    init = tf.global_variables_initializer()

    # Calculate predicted Et by multiplying layer weights by layer Ets and then adding bias value
    y_pred = tf.matmul(x, W)
    y_pred = tf.add(y_pred, shift_et)
    #y_pred = tf.add(y_pred, b)

    # Calculate cost measures based on the difference between predicted Et and true Et
    y_diff = tf.subtract(y, y_pred)
    cost = tf.reduce_mean(tf.square(y_diff))
    resolution = tf.reduce_mean(y_diff)

    g = tf.gradients(cost, W)

    # Create optimizer to minimize the square of the cost variable
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(cost)

    with tf.Session() as sess:
        sess.run(init)

        # Print functions for testing purposes
        x_in, weights, init_y_pred, y_true, y_diff, err, res, grad = sess.run([x, W, y_pred, y, y_diff, cost, resolution, g],
                                                                              feed_dict={x: cell_ets[test], y: true_ets[test]})
        print('Initial values')
        print('x_in')
        print(x_in)
        print('weights')
        print(weights)
        print('init_y_pred')
        print(init_y_pred)
        print('y_true')
        print(y_true)
        print('y_diff')
        print(y_diff)

        print('cost')
        print(err)
        print('resolution')
        print(res)

        print('gradient')
        print(grad)


        # Calculate initial cost and resolution before any training
        err, res = sess.run([cost, resolution], feed_dict={x: cell_ets[test], y: true_ets[test]})

        # Create arrays to hold the changing values of cost and resolution as the network trains
        cost_history = np.array(err)
        resolution_history = np.array(res)

        # For each epoch, train network and add current cost and resolution values to arrays
        for i in range(epochs):

            result = sess.run([optimizer], feed_dict={x: cell_ets[train], y: true_ets[train]})
            err, res = sess.run([cost, resolution], feed_dict={x: cell_ets[test], y: true_ets[test]})

            #if i == 0:
            #    print('err:',err)
            #    print(sess.run(W))

            cost_history = np.append(cost_history, err)
            resolution_history = np.append(resolution_history, res)

            #if i%10 == 0:
            #    print('Epoch: {0}, Error: {1}, Resolution: {2}'.format(i, err, res))

        end_weights, end_bias = sess.run([W, b], feed_dict={x: cell_ets[test], y: true_ets[test]})

    return resolution_history, cost_history, end_weights, end_bias


# Train network on totality of data and use the results as final measure
resolution_history, cost_history, end_weights, end_bias = train_layerweighted_nn(cell_ets, true_ets, range(event_num), range(event_num))

# Get end weights into manageable format
end_weights = [end_weights[i][0] for i in range(len(end_weights))]
print('End weights')
print(end_weights)
print('End resolution')
print(resolution_history[-1])
print('End cost')
print(cost_history[-1])
