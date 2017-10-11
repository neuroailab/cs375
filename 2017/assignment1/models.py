"""
Please implement a standard AlexNet model here as defined in the paper
    https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf

Note: Although you will only have to edit a small fraction of the code at the
beginning of the assignment by filling in the blank spaces, you will need to
build on the completed starter code to fully complete the assignment,
We expect that you familiarize yourself with the codebase and learn how to
setup your own experiments taking the assignments as a basis. This code does
not cover all parts of the assignment and only provides a starting point. To
fully complete the assignment significant changes have to be made and new
functions need to be added after filling in the blanks. Also, for your projects
we won't give out any code and you will have to use what you have learned from
your assignments. So please always carefully read through the entire code and
try to understand it. If you have any questions about the code structure,
we will be happy to answer it.

Attention: All sections that need to be changed to complete the starter code
are marked with EDIT!
"""

import os
import numpy as np
import tensorflow as tf

def alexnet_model(inputs, train=True, norm=True, **kwargs):
    """
    AlexNet model definition as defined in the paper:
    https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf

    You will need to EDIT this function. Please put your AlexNet implementation here.
    
    Note: 
    1.) inputs['images'] is a [BATCH_SIZE x HEIGHT x WIDTH x CHANNELS] array coming
    from the data provider.
    2.) You will need to return 'output' which is a dictionary where 
    - output['pred'] is set to the output of your model
    - output['conv1'] is set to the output of the conv1 layer
    - output['conv1_kernel'] is set to conv1 kernels
    - output['conv2'] is set to the output of the conv2 layer
    - output['conv2_kernel'] is set to conv2 kernels
    - and so on...
    The output dictionary should include the following keys for AlexNet:
    ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'pool1', 
     'pool2', 'pool5', 'fc6', 'fc7', 'fc8'] 
    as well as the respective ['*_kernel'] keys for the kernels
    3.) Set your variable scopes to the name of the respective layers, e.g.
        with tf.variable_scope('conv1'):
            outputs['conv1'] = ...
            outputs['pool1'] = ...
    and
        with tf.variable_scope('fc6'):
            outputs['fc6'] = ...
    and so on. 
    4.) Use tf.get_variable() to create variables, while setting name='weights'
    for each kernel, and name='bias' for each bias for all conv and fc layers.
    For the pool layers name='pool'.

    These steps are necessary to correctly load the pretrained alexnet model
    from the database for the second part of the assignment.
    """

    # propagate input targets
    outputs = inputs
    dropout = .5 if train else None
    input_to_network = inputs['images']

    ### YOUR CODE HERE
    
    # def conv():
    #   return outputs, kernel 
    
    # conv1
    # pool1
    # conv2
    # pool2
    # conv3
    # conv4
    # conv5
    # pool5
    # fc6
    # fc7
    # fc8
    
    # conv function, returns conv output and the kernel
    # pool returns output
    # fc returns output
    kernel = tf.get_variable(initializer=init,
                            shape=[ksize[0], ksize[1], in_depth, out_depth],
                            dtype=tf.float32,
                            regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                            name='weights')
    
    # the actual conv layer takes in a kernel and uses the following to build the tf 'module'
    conv = tf.nn.conv2d(inp, kernel,
                        strides=strides,
                        padding=padding)
    output = tf.nn.bias_add(conv, biases, name=name)    
    
    ### END OF YOUR CODE

    for k in ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'pool1',
            'pool2', 'pool5', 'fc6', 'fc7', 'fc8', 'conv1_kernel', 'pred']:
        assert k in outputs, '%s was not found in outputs' % k
    return outputs, {}


def conv(inp,
         out_depth,
         ksize=[3,3],
         strides=[1,1,1,1],
         padding='SAME',
         kernel_init='xavier',
         kernel_init_kwargs=None,
         bias=0,
         weight_decay=None,
         activation='relu',
         batch_norm=True,
         name='conv'
         ):

    # assert out_shape is not None
    if weight_decay is None:
        weight_decay = 0.
    if isinstance(ksize, int):
        ksize = [ksize, ksize]
    if kernel_init_kwargs is None:
        kernel_init_kwargs = {}
    in_depth = inp.get_shape().as_list()[-1]

    # weights
    init = initializer(kernel_init, **kernel_init_kwargs)
    kernel = tf.get_variable(initializer=init,
                            shape=[ksize[0], ksize[1], in_depth, out_depth],
                            dtype=tf.float32,
                            regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                            name='weights')
    init = initializer(kind='constant', value=bias)
    biases = tf.get_variable(initializer=init,
                            shape=[out_depth],
                            dtype=tf.float32,
                            regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                            name='bias')
    # ops
    conv = tf.nn.conv2d(inp, kernel,
                        strides=strides,
                        padding=padding)
    output = tf.nn.bias_add(conv, biases, name=name)

    if activation is not None:
        output = getattr(tf.nn, activation)(output, name=activation)
    if batch_norm:
        output = tf.nn.batch_normalization(output, mean=0, variance=1, offset=None,
                            scale=None, variance_epsilon=1e-8, name='batch_norm')
    return output
