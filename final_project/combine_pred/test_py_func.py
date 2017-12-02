import tensorflow as tf
import numpy as np

def my_func(x):
      # x will be a numpy array with the contents of the placeholder below
        return np.sinh(x)

inp = tf.placeholder(tf.float32)
y = tf.py_func(my_func, [inp], tf.float32)

z = y + 100

with tf.Session() as sess:
    print(sess.run(z, feed_dict = {inp: 1}))
