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

def get_conv(inp, conv_shape, stride, activation=tf.nn.relu):
    weights = tf.get_variable(shape=conv_shape, dtype=tf.float32, 
                              initializer=tf.contrib.layers.xavier_initializer(), name='weights')
    conv = tf.nn.conv2d(inp, weights,[1, stride, stride, 1], padding='SAME', name='conv')
    biases = tf.get_variable(initializer=tf.zeros_initializer(), shape=[conv_shape[3]], dtype=tf.float32, name='biases')
    bias = tf.nn.bias_add(conv, biases)
    if activation is not None:
        act = activation(bias)
    else:
        act = bias
    out = tf.contrib.layers.batch_norm(act, scale=True)
    return out, weights

def get_deconv(inp, conv_shape, stride, output_shape, activation=tf.nn.relu):
    weights = tf.get_variable(shape=conv_shape, dtype=tf.float32, 
                              initializer=tf.contrib.layers.xavier_initializer(), name='weights')
    conv = tf.nn.conv2d_transpose(inp, weights, output_shape, [1, stride, stride, 1],padding='SAME', name='deconv')
    biases = tf.get_variable(initializer=tf.zeros_initializer(), shape=[conv_shape[2]], dtype=tf.float32, name='biases')
    bias = tf.nn.bias_add(conv, biases)
    if activation is not None:
        out = activation(bias)
    else:
        out = bias
    return out, weights

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
    
    with tf.variable_scope('conv'):
        outputs['relu'], outputs['conv_kernel'] = get_conv(input_to_network,[7,7,3,64],16)
    with tf.variable_scope('deconv'):
        outputs['deconv'] = get_deconv(outputs['relu'],[12,12,3,64],12,input_to_network.shape)
    
#     shape = input_to_network.get_shape().as_list()
#     stride = 16
#     hidden_size = 2
#     deconv_size = 12
    
#     ### YOUR CODE HERE
#     outputs['input'] = input_to_network  
#     conv_layer = K.layers.Conv2D(64,7,strides=(stride,stride),
#                                       padding='same',
#                                       kernel_initializer='glorot_normal')
#     outputs['conv_kernel'] = conv_layer
#     outputs['conv'] = conv_layer(input_to_network)
#     outputs['relu'] = K.layers.Activation('relu')(outputs['conv'])
#     outputs['deconv'] = K.layers.Conv2DTranspose(3,deconv_size,
#                                                  deconv_size,padding='valid',
#                                                  kernel_initializer='glorot_normal')(outputs['relu'])
    
    ### END OF YOUR CODE
    for k in ['deconv']:
        assert k in outputs, '%s was not found in outputs' % k

    return outputs, {}

def ae_model_in(inputs, train=True, norm=True, **kwargs):
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
    
    with tf.variable_scope('conv'):
        outputs['relu'], outputs['conv_kernel'] = get_conv(input_to_network,[7,7,3,256],16)
    with tf.variable_scope('deconv'):
        outputs['deconv'], w = get_deconv(outputs['relu'],[12,12,3,256],16,input_to_network.shape)
    
    
    ### END OF YOUR CODE
    for k in ['deconv']:
        assert k in outputs, '%s was not found in outputs' % k

    return outputs, {}

def ae_model_sparse(inputs, train=True, norm=True, **kwargs):
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
    
    with tf.variable_scope('conv'):
        outputs['relu'], outputs['conv_kernel'] = get_conv(input_to_network,[7,7,3,64],16)
    with tf.variable_scope('deconv'):
        outputs['deconv'], outputs['deconv_kernel'] = get_deconv(outputs['relu'],[12,12,3,64],12,input_to_network.shape)
    
    ### END OF YOUR CODE
    for k in ['deconv']:
        assert k in outputs, '%s was not found in outputs' % k

    return outputs, {}

def ae_loss_sparse(inputs, outputs):
    return tf.nn.l2_loss(outputs['images'] - outputs['deconv']) + tf.reduce_mean(0.01 * (tf.nn.l2_loss(outputs['deconv_kernel']) + tf.nn.l2_loss(outputs['conv_kernel']))) + tf.reduce_mean(0.01 * outputs['deconv'])

def ae_loss(inputs, outputs):  
    return tf.nn.l2_loss(outputs['images'] - outputs['deconv'])
