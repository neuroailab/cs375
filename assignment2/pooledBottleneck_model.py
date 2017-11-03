"""
Pooled version of the shallow bottleneck autoencoder
"""

import os
import numpy as np
import tensorflow as tf



def pBottleneck_model(inputs, train=True, norm=True, **kwargs):
    """
    A pooled shallow bottleneck convolutional autoencoder model..
    """
    # propagate input targets
    outputs = inputs
#    dropout = .5 if train else None
    input_to_network = inputs['images']
    
    
    shape = input_to_network.get_shape().as_list()
    stride = 16
    hidden_size = 2#np.ceil(shape[1]/stride)
    deconv_size = 12#(shape[1]/hidden_size).astype(int)
    
    
    ### YOUR CODE HERE
    with tf.variable_scope('conv1') as scope:
        convweights = tf.get_variable(shape=[7, 7, 3, 64], dtype=tf.float32, 
                                  initializer=tf.contrib.layers.xavier_initializer(), name='weights')
        conv = tf.nn.conv2d(input_to_network, convweights,[1, 4, 4, 1], padding='SAME')
        biases = tf.get_variable(initializer=tf.constant_initializer(0),
                                 shape=[64], dtype=tf.float32, trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        relu = tf.nn.relu(bias, name='relu')
        pool = tf.nn.max_pool(value=relu, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME', name='pool')
        # assign layers to output
        outputs['input'] = input_to_network
        outputs['conv1_kernel'] = convweights
        outputs['conv1'] = relu
        outputs['pool1'] = pool
    
    print(outputs['input'].shape)
    print(outputs['conv1'].shape)
    print(outputs['pool1'].shape)
        
    with tf.variable_scope('deconv2') as scope:
        deconvweights = tf.get_variable(shape=[deconv_size, deconv_size, 3, 64], dtype=tf.float32, 
                                  initializer=tf.contrib.layers.xavier_initializer(), name='weights')
        deconv = tf.nn.conv2d_transpose(outputs['pool1'], deconvweights, 
         outputs['input'].shape, [1, 12, 12, 1], padding='VALID', name=None)
        # assign layers to output
        outputs['deconv2'] = deconv
        
    ### END OF YOUR CODE
    for k in ['input','conv1', 'deconv2']:
        assert k in outputs, '%s was not found in outputs' % k

    return outputs, {}

def pBottleneckSparse_model(inputs, train=True, norm=True, **kwargs):
    """
    A pooled shallow bottleneck convolutional autoencoder model..
    """
    # propagate input targets
    outputs = inputs
#    dropout = .5 if train else None
    input_to_network = inputs['images']
    
    
    shape = input_to_network.get_shape().as_list()
    stride = 16
    hidden_size = 2#np.ceil(shape[1]/stride)
    deconv_size = 12#(shape[1]/hidden_size).astype(int)
    
    
    ### YOUR CODE HERE
    with tf.variable_scope('conv1') as scope:
        convweights = tf.get_variable(shape=[7, 7, 3, 64], dtype=tf.float32, 
                                  initializer=tf.contrib.layers.xavier_initializer(), name='weights')
        conv = tf.nn.conv2d(input_to_network, convweights,[1, 4, 4, 1], padding='SAME')
        biases = tf.get_variable(initializer=tf.constant_initializer(0),
                                 shape=[64], dtype=tf.float32, trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        relu = tf.nn.relu(bias, name='relu')
        pool = tf.nn.max_pool(value=relu, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME', name='pool')
        # assign layers to output
        outputs['input'] = input_to_network
        outputs['conv1_kernel'] = convweights
        outputs['conv1'] = relu
        outputs['pool1'] = pool
        outputs['convweights'] = convweights
    
    print(outputs['input'].shape)
    print(outputs['conv1'].shape)
    print(outputs['pool1'].shape)
        
    with tf.variable_scope('deconv2') as scope:
        deconvweights = tf.get_variable(shape=[deconv_size, deconv_size, 3, 64], dtype=tf.float32, 
                                  initializer=tf.contrib.layers.xavier_initializer(), name='weights')
        deconvRegularizer = tf.nn.l2_loss(deconvweights)
        deconv = tf.nn.conv2d_transpose(outputs['pool1'], deconvweights, 
         [256, 24, 24, 3], [1, 12, 12, 1], padding='VALID', name=None)
        # assign layers to output
        outputs['deconv2'] = deconv
        outputs['deconvweights'] = deconvweights
        
    ### END OF YOUR CODE
    for k in ['input','conv1', 'deconv2']:
        assert k in outputs, '%s was not found in outputs' % k

    return outputs, {}

def pbottle_loss(inputs, outputs):  
    return tf.nn.l2_loss(outputs['images'] - outputs['deconv2'])

def pbottleSparse_loss(inputs, outputs):  
    return tf.nn.l2_loss(outputs['images'] - outputs['deconv2']) + 0.001*tf.reduce_sum(tf.abs(outputs['pool1'])) + tf.nn.l2_loss(outputs['convweights']) + tf.nn.l2_loss(outputs['deconvweights'])