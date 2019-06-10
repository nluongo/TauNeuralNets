import tensorflow as tf
import numpy as np
import json

#First read in data from text file
EventEts = []
with open("../RootScripts/TauEts.txt", "r") as inputfile:
    All_Et = json.load(inputfile)

Layer_Event_Et = All_Et[0]
Total_Event_Et = All_Et[1]

print(Layer_Event_Et[0])
print(np.array(Layer_Event_Et[0])/Total_Event_Et[0])
print(Total_Event_Et[0])

Normalized_Layer_Event_Et = np.zeros((5281,5))

for i in range(5281):
    #print(i)
    Normalized_Layer_Event_Et[i] = np.array(Layer_Event_Et[i]) / Total_Event_Et[i]

#print(Normalized_Layer_Event_Et[1])

np_et = np.array(Normalized_Layer_Event_Et, dtype=np.float32)
np.expand_dims(np_et, 1)
tf_et = tf.convert_to_tensor(np_et, np.float32)

dataset = tf.data.Dataset.from_tensor_slices(tf_et)
print(dataset.output_shapes)

rando = tf.random_uniform([3])

sess = tf.Session()
print(sess.run(tf_et))
print(sess.run(rando))
sess.close()
'''
datapoint_size = 5281
batch_size = 1
steps = 100000
learn_rate = 0.000001

W = tf.Variable(tf.constant([1,1.1,1,1,1], dtype=tf.float32, shape=(5,1)), name="W")
b = tf.Variable(tf.zeros([1]), name="b")

x = tf.placeholder(tf.float32, shape=(1,5), name="x")
y = tf.constant(1.0)

product = tf.matmul(x,W)
prediction = product # + b

cost = tf.reduce_mean(tf.square(prediction-y))

train_step = tf.train.GradientDescentOptimizer(learn_rate).minimize(cost)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
for i in range(steps):
    j = i%5281
    if j%500 == 0:
        print("i:")
        print(i)
        print("cost:")
        print(sess.run(cost, feed_dict={x:np.expand_dims(Normalized_Layer_Event_Et[:5], axis=1)}))
        #print("W:")
        #print(sess.run(W))
        #print("Et:")
        #print(Normalized_Layer_Event_Et[i])
        #print("prediction:")
        #print(sess.run(prediction, feed_dict={x:np.expand_dims(Normalized_Layer_Event_Et[i], axis=0)}))
        #print("Second prediction:")
        #print(sess.run(prediction, feed_dict={x:np.expand_dims(Normalized_Layer_Event_Et[i], axis=0)}))
        print()
    sess.run(train_step, feed_dict={x:np.expand_dims(Normalized_Layer_Event_Et[j], axis=0)})
weights, cost = sess.run([W,cost], {x:np.expand_dims(Normalized_Layer_Event_Et[0], axis=0)})
print("W: %s cost: %s" % (weights, cost))
sess.close()
'''