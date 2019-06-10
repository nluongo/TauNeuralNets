import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
from sklearn.model_selection import KFold

# Define and open path to the flat file that we are reading Et information from
sig_flat_file_path = os.path.join(os.path.expanduser('~'), 'TauTrigger', 'Formatted Data Files', 'Flat Files', 'Et', 'RecoEt_PO_Flat.txt')
sig_flat_file = open(sig_flat_file_path, 'r')

back_flat_file_path = os.path.join(os.path.expanduser('~'), 'TauTrigger', 'Formatted Data Files', 'Flat Files', 'Et', 'RecoEt_ETBack_Flat.txt')
back_flat_file = open(back_flat_file_path, 'r')

cell_ets = []
true_ets = []

# Read each line from flat file and store the first five values as layer Ets and the last as the true Et
for line in sig_flat_file:
    ets = line.split(',')
    floats = [float(et) for et in ets[0:5]]
    cell_ets.append(floats)
    #true_ets.append([float(ets[5])])
    true_ets.append([float(100)])

for line in back_flat_file:
    ets = line.split(',')
    floats = [float(et) for et in ets[0:5]]
    cell_ets.append(floats)
    #true_ets.append([float(ets[5])])
    true_ets.append([float(0)])

# Convert Et lists to numpy arrays for use with tensorflow
cell_ets = np.array(cell_ets)
true_ets = np.array(true_ets)

# Get total number of events
event_num = len(cell_ets)

kfold = KFold(5, True, 1)

np.random.seed(6)

lr = 0.001
epochs = 5000
num_layers = 5


def train_layerweighted_nn(cell_ets, true_ets, train, test):
    print('Starting training...')

    # Create placeholders for the values that will be input when training
    x = tf.placeholder(tf.float32,[None, num_layers])
    y = tf.placeholder(tf.float32,[None, 1])

    # Create variables for the values that will be calculated when training
    W = tf.Variable(tf.random_uniform([num_layers, 1], dtype=np.float32))
    #W = tf.scalar_mul(coef_et, W)
    b = tf.Variable(np.zeros([1], dtype=np.float32))

    # Initialize global variables
    init = tf.global_variables_initializer()

    # Calculate predicted Et by multiplying layer weights by layer Ets and then adding bias value
    y_pred = tf.matmul(x, W)
    y_pred = tf.add(y_pred, b)

    # Calculate cost measures based on the difference between predicted Et and true Et
    y_diff = tf.subtract(y, y_pred)
    resolution = tf.reduce_mean(y_diff)
    y_diff = tf.square(y_diff)
    cost = tf.reduce_mean(y_diff)

    # Create optimizer to minimize the square of the cost variable
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(cost)

    with tf.Session() as sess:
        sess.run(init)

        # Print functions for testing purposes
        x_in, weights, init_y_pred, y_true, y_diff, err, res = sess.run([x, W, y_pred, y, y_diff, cost, resolution],
                                                                        feed_dict={x: cell_ets[test], y: true_ets[test]})
        '''
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
        '''

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

            if i%10 == 0:
                print('Epoch: {0}, Error: {1}, Resolution: {2}'.format(i, err, res))

        end_weights, end_bias = sess.run([W, b], feed_dict={x: cell_ets[test], y: true_ets[test]})

    return resolution_history, cost_history, end_weights, end_bias

'''
res_list = []

# Train the network for each kfold split and add the ending resolution to the list
for train, test in kfold.split(cell_ets):
    resolution_history, cost_history, end_weights, end_bias = train_layerweighted_nn(cell_ets, true_ets, train, test)
    res_list.append(resolution_history[-1])

# Print information on the ending resolution across all kfold splits
print(res_list)
print('Mean: ', np.mean(res_list))
print('St Dev: ', np.std(res_list))
'''
# Train network on totality of data and use the results as final measure
resolution_history, cost_history, end_weights, end_bias = train_layerweighted_nn(cell_ets, true_ets, range(event_num), range(event_num))

# Get end weights into manageable format
end_weights = [end_weights[i][0] for i in range(len(end_weights))]
print(end_weights)
print(end_bias[0])

title = 'HundredSigAndBackBias'
with PdfPages(title + '.pdf') as pdf:
    # Save cost vs. epoch plot into pdf
    plt.plot(range(epochs+1), cost_history)
    plt.axis([0, epochs, 0, np.max(cost_history)])
    plt.xlabel('Epochs')
    plt.ylabel('Cost')
    plt.title(title + ' Cost vs. Epochs')
    pdf.savefig()
    plt.close()

    # Save resolution vs epoch plot into pdf
    plt.plot(range(epochs+1), resolution_history)
    plt.axis([0, epochs, np.min(resolution_history), np.max(resolution_history)])
    plt.xlabel('Epochs')
    plt.ylabel('Average Resolution')
    plt.title(title + ' Average Resolution vs. Epochs')
    pdf.savefig()
    plt.close()

    # Save bar plot of layer weights into pdf
    plt.bar([0, 1, 2, 3, 4], end_weights, align='center')
    plt.xlabel('Layer')
    plt.xticks(range(5), ('0', '1', '2', '3', 'Hadronic'))
    plt.ylabel('Weight')
    plt.title(title + ' Layer Weights')
    pdf.savefig()
    plt.close()
