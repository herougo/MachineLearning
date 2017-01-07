import tensorflow as tf
import numpy as np
# import sys
# sys.path.append("/usr/lib/python2.7/site-packages/tensorflow")

x1 = tf.constant(5)
x2 = tf.constant(6)

# result = x1 * x2 works, but use the tf versions
result  = tf.mul(x1, x2)

print("Print the tensor object thing")
print(result)

with tf.Session() as sess:
	output = sess.run(result)
	print("Print result of the evaluation")
	print(output)