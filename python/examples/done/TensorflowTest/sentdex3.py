import tensorflow as tf
import numpy as np
# import sys
# sys.path.append("/usr/lib/python2.7/site-packages/tensorflow")
import os
path = os.getcwd() + "/sentdex3checkpoint/"
ckpt_ext = 'model.ckpt'

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/tensorflow/mnist/input_data", one_hot=True)


n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10
n_epochs = 10  # number of iterations
batch_size = 100

# height x width
# TIP: add [None, 784] to avoid possibly confusing errors in the future
x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

def neural_network_model(data):
	# nput data * weights + bias
	hidden_1_layer = {'weights': tf.Variable(tf.random_normal([784, n_nodes_hl1])),
					  'biases':  tf.Variable(tf.random_normal([n_nodes_hl1]))}
	hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
					  'biases':  tf.Variable(tf.random_normal([n_nodes_hl2]))}
	hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
					  'biases':  tf.Variable(tf.random_normal([n_nodes_hl3]))}
	output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
					'biases':  tf.Variable(tf.random_normal([n_classes]))}

	l1 = tf.matmul(data, hidden_1_layer['weights']) + hidden_1_layer['biases']
	l1 = tf.nn.relu(l1)

	l2 = tf.matmul(l1, hidden_2_layer['weights']) + hidden_2_layer['biases']
	l2 = tf.nn.relu(l2)

	l3 = tf.matmul(l2, hidden_3_layer['weights']) + hidden_3_layer['biases']
	l3 = tf.nn.relu(l3)

	output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']
	saver = tf.train.Saver()

	return output, saver

def train_neural_network(data):
	prediction, saver = neural_network_model(data)
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