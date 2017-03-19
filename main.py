from __future__ import print_function

import csv
import tensorflow as tf
from tensorflow.contrib import rnn
import time
import datetime
import numpy as np

data = []

with open('005930.csv', 'rb') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        data.append(row)

timestamps = []      

for i in range(1, len(data)):
    timestamp=time.mktime(datetime.datetime.strptime(data[i][0],"%d-%b-%y").timetuple())
    timestamps.append(timestamp)

timestamps = np.array(timestamps, dtype = np.float64)
print(timestamps)

data = np.array(data[1:], dtype=np.float64)
print(data)

learning_rate = 0.001
training_iters = 100000
batch_size = 128
display_step = 10

n_input = 28
n_steps = 28
n_hidden = 128
n_classes = 10

x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])

weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}


x = tf.transpose(x, [1, 0, 2])
x = tf.reshape(x, [-1, n_input])
x = tf.split(x, n_steps, 0)

lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

pred = tf.matmul(outputs[-1], weights['out']) + biases['out']

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

        

        
    

