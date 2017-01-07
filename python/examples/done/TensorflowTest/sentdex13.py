# Convolutional NN

# accuracy: 0.96950001

import tensorflow as tf
import numpy as np
# import sys
# sys.path.append("/usr/lib/python2.7/site-packages/tensorflow")

import os
path = os.getcwd() + "/sentdex13checkpoint/"
ckpt_ext = 'model.ckpt'

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/tensorflow/mnist/input_data", one_hot=True)


n_classes = 10
n_epochs = 10  # number of iterations
batch_size = 100

keep_rate = 0.8 # dropout

# height x width
# TIP: add [None, 784] to avoid possibly confusing errors in the future
x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def maxpool2d(x):
	#                        size of window (2x2)        movement of window (2x2)
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def conv_neural_network(data):
	# nput data * weights + bias
	weights = {'W_conv1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
			   'W_conv2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
			   'W_fc': tf.Variable(tf.random_normal([7*7*64, 1024])),
			   'out': tf.Variable(tf.random_normal([1024, n_classes]))}
	biases = {'b_conv1': tf.Variable(tf.random_normal([32])),
			  'b_conv2': tf.Variable(tf.random_normal([64])),
			  'b_fc': tf.Variable(tf.random_normal([1024])),
			  'out': tf.Variable(tf.random_normal([n_classes]))}

	# reshape to 28 x 28 image
	data = tf.reshape(x, shape=[-1, 28, 28, 1])

	conv1 = conv2d(data, weights['W_conv1'])
	conv1 = maxpool2d(conv1)

	conv2 = conv2d(conv1, weights['W_conv2'])
	conv2 = maxpool2d(conv2)

	# fully connected
	fc = tf.reshape(conv2, [-1, 7*7*64])
	fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])

	tf.nn.dropout(fc, keep_rate)

	output = tf.matmul(fc, weights['out']) + biases['out']
	saver = tf.train.Saver()

	return output, saver

def train_neural_network(data):
	prediction, saver = conv_neural_network(data)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y))
	optimizer = tf.train.AdamOptimizer().minimize(cost)
	
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for epoch in range(n_epochs):
			epoch_loss = 0
			for _ in range(int(mnist.train.num_examples/batch_size)):
				batch_xs, batch_ys = mnist.train.next_batch(batch_size)
				_, c = sess.run([optimizer, cost], feed_dict={x: batch_xs, y: batch_ys})
				epoch_loss += c;
			print('Epoch', epoch, 'completed out of', n_epochs, 'loss:', epoch_loss)
		
		os.mkdir(path)
		saver.save(sess, path + ckpt_ext)

		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		print('Accuracy:', accuracy.eval({x: mnist.test.images,
                                      y : mnist.test.labels}))

train_neural_network(x)
