# Recurrent NN

import tensorflow as tf
import numpy as np
from tensorflow.python.ops import rnn, rnn_cell
# import sys
# sys.path.append("/usr/lib/python2.7/site-packages/tensorflow")
import os
path = os.getcwd() + "/sentdex11checkpoint/"
ckpt_ext = 'model.ckpt'

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/tensorflow/mnist/input_data", one_hot=True)


n_classes = 10
n_epochs = 10  # number of iterations
batch_size = 128

chunk_size = 28
n_chunks = 28
rnn_size = 128

# height x width
# TIP: add [None, 784] to avoid possibly confusing errors in the future
x = tf.placeholder('float', [None, n_chunks, chunk_size])
y = tf.placeholder('float')

def recurrent_neural_network(input):
	# nput data * weights + bias
	layer = {'weights': tf.Variable(tf.random_normal([rnn_size, n_classes])),
			 'biases':  tf.Variable(tf.random_normal([n_classes]))}

	input = tf.transpose(input, [1, 0, 2])
	input = tf.reshape(input, [-1, chunk_size])
	input = tf.split(0, chunk_size, input)

	lstm_cell = rnn_cell.BasicLSTMCell(rnn_size)
	outputs, states = rnn.rnn(lstm_cell, input, dtype=tf.float32)

	output = tf.matmul(outputs[-1], layer['weights']) + layer['biases']
	
	saver = tf.train.Saver()
	return output, saver

def train_neural_network(data):
	prediction, saver = recurrent_neural_network(data)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y))

	optimizer = tf.train.AdamOptimizer().minimize(cost)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		ckpt = tf.train.get_checkpoint_state(path)
		if ckpt and ckpt.model_checkpoint_path:
			saver.restore(sess, ckpt.model_checkpoint_path)
		else: # train neural network
			for epoch in range(n_epochs):
				epoch_loss = 0
				for _ in range(int(mnist.train.num_examples/batch_size)):
					batch_xs, batch_ys = mnist.train.next_batch(batch_size)
					batch_xs = batch_xs.reshape((batch_size, n_chunks, chunk_size))

					_, c = sess.run([optimizer, cost], feed_dict={x: batch_xs, y: batch_ys})
					epoch_loss += c;
				print('Epoch', epoch, 'completed out of', n_epochs, 'loss:', epoch_loss)
			os.mkdir(path)
			saver.save(sess, path + ckpt_ext)

		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		print('Accuracy:', accuracy.eval({x: mnist.test.images.reshape((-1, n_chunks, chunk_size)),
                                      y : mnist.test.labels}))

train_neural_network(x)
