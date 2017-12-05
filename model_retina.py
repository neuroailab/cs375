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
import tensorflow.contrib.keras as K

def dense_(inputs, shape, activation=tf.nn.softplus):
    """
    Args:
        shape: [input, output]
    """
    weights = tf.get_variable('weights', shape, tf.float32, initializer=tf.random_normal_initializer(stddev=0.05),
                              kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    biases = tf.get_variable('biases', [shape[1]], tf.float32, tf.zeros_initializer())
    FC = tf.nn.xw_plus_b(inputs, weights, biases, name='FC')
    if activation is not None:
        out = activation(FC)
    else:
        out = FC
    return out

def conv_(inp, conv_shape, stride, padding='SAME', reg=None):
    if reg is not None:
        weights = tf.get_variable(shape=conv_shape, dtype=tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer(),
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(reg),
                                  name='weights')
    else:
        weights = tf.get_variable(shape=conv_shape, dtype=tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer(), name='weights')
    conv = tf.nn.conv2d(inp, weights,[1, stride, stride, 1], padding=padding, name='conv')
    biases = tf.get_variable(initializer=tf.zeros_initializer(), shape=[conv_shape[3]], dtype=tf.float32, name='bias')
    bias = tf.nn.bias_add(conv, biases)
    return out, weights

def gaussian_noise_(input_layer, std):
    noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32) 
    return input_layer + noise

def relu_(inp):
    return tf.nn.relu(inp)

def ae_model(inputs, train=True, norm=True, **kwargs):
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
#     dropout = .5 if train else None
    input_to_network = inputs['images']
    
    outputs['input'] = input_to_network
    
    with tf.variable_scope('conv1'):
        temp, outputs['conv1_kernel'] = conv_(input_to_network,[15,15,3,16],1,padding='VALID')
        if train:
            outputs['conv1'] = relu_(gaussian_noise_(temp),std=0.1)
        else:
            outputs['conv1'] = relu_(temp)
            
    with tf.variable_scope('conv2'):
        temp, outputs['conv2_kernel'] = conv_(outputs['conv1'],[9,9,16,8],1,padding='VALID',regularizer=1e-3)
        if train:
            outputs['conv2'] = relu_(gaussian_noise_(temp),std=0.1)
        else:
            outputs['conv2'] = relu_(temp)
    
    with tf.variable_scope('fc'):
        flat_len = np.product(outputs['conv2'].shape.as_list()[1:])
        flatten = tf.reshape(outputs['conv2'], [-1, flat_len]) 
        outputs['pred'] = dense_(flatten,[flat_len,5])
        
    ### END OF YOUR CODE
    for k in ['deconv']:
        assert k in outputs, '%s was not found in outputs' % k

    return outputs, {}

def ae_loss(inputs, outputs):  
    return tf.nn.l2_loss(outputs['images'] - outputs['deconv'])
