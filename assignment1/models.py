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
    with tf.variable_scope('conv1') as scope:
        weights = tf.get_variable(shape=[11, 11, 3, 96], dtype=tf.float32, 
                                  initializer=tf.contrib.layers.xavier_initializer(), name='weights')
        conv = tf.nn.conv2d(input_to_network, weights,[1, 4, 4, 1], padding='VALID')
        biases = tf.get_variable(initializer=tf.constant_initializer(0),
                                 shape=[96], dtype=tf.float32, trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        relu = tf.nn.relu(bias, name='relu')
        lrn = tf.nn.local_response_normalization(relu, depth_radius=2, bias=1, alpha=.00002, beta=.75) # bias=Kappa?
        pool = tf.nn.max_pool(value=lrn, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool')
        # assign layers to output
        outputs['conv1_kernel'] = weights
        outputs['conv1'] = lrn
        outputs['pool1'] = pool
        
    with tf.variable_scope('conv2') as scope:
        weights = tf.get_variable(shape=[5, 5, 96, 256], dtype=tf.float32, 
                                  initializer=tf.contrib.layers.xavier_initializer(), name='weights')
        conv = tf.nn.conv2d(outputs['pool1'], weights,[1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable(initializer=tf.constant_initializer(0),
                                 shape=[256], dtype=tf.float32, trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        relu = tf.nn.relu(bias, name='relu')
        lrn = tf.nn.local_response_normalization(relu, depth_radius=2, bias=1, alpha=.00002, beta=.75) # bias=Kappa?
        pool = tf.nn.max_pool(value=lrn, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool')
        # assign layers to output
        outputs['conv2_kernel'] = weights
        outputs['conv2'] = lrn
        outputs['pool2'] = pool
        
    with tf.variable_scope('conv3') as scope:
        weights = tf.get_variable(shape=[3, 3, 256, 384], dtype=tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer(), name='weights')
        conv = tf.nn.conv2d(outputs['pool2'], weights,[1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable(initializer=tf.constant_initializer(0),
                                 shape=[384], dtype=tf.float32, trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        relu = tf.nn.relu(bias, name='relu')
        # assign layers to output
        outputs['conv3_kernel'] = weights
        outputs['conv3'] = relu
        
    with tf.variable_scope('conv4') as scope:
        weights = tf.get_variable(shape=[3, 3, 384, 384], dtype=tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer(), name='weights')
        conv = tf.nn.conv2d(outputs['conv3'], weights,[1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable(initializer=tf.constant_initializer(0),
                                 shape=[384], dtype=tf.float32, trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        relu = tf.nn.relu(bias, name='relu')
        # assign layers to output
        outputs['conv4_kernel'] = weights
        outputs['conv4'] = relu
        
    with tf.variable_scope('conv5') as scope:
        weights = tf.get_variable(shape=[3, 3, 384, 256], dtype=tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer(), name='weights')
        conv = tf.nn.conv2d(outputs['conv4'], weights,[1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable(initializer=tf.constant_initializer(0),
                                 shape=[256], dtype=tf.float32, trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        relu = tf.nn.relu(bias, name='relu')
        pool = tf.nn.max_pool(value=relu, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
        # assign layers to output
        outputs['conv5_kernel'] = weights
        outputs['conv5'] = relu
        outputs['pool5'] = pool
        
    with tf.variable_scope('fc6') as scope:
        shape = np.product(outputs['pool5'].shape.as_list()[1:])
        flatten = tf.reshape(outputs['pool5'], [-1, shape]) 
        weights = tf.get_variable(shape=[shape,4096], dtype=tf.float32, 
                                  initializer=tf.truncated_normal_initializer(stddev=.01), name='weights')
        biases = tf.get_variable(initializer=tf.constant_initializer(0.1),
                                 shape=[4096], dtype=tf.float32, trainable=True, name='biases')
        fc = tf.nn.relu_layer(flatten, weights, biases, name='fc')
        if dropout==None:
            dropout_layer = tf.nn.dropout(fc, 1, name='dropout')
        else:
            dropout_layer = tf.nn.dropout(fc, dropout, name='dropout')
        outputs['fc6'] = dropout_layer
                              
    with tf.variable_scope('fc7') as scope:
        weights = tf.get_variable(shape=[4096,4096], dtype=tf.float32, 
                                  initializer=tf.truncated_normal_initializer(stddev=0.01), name='weights')
        biases = tf.get_variable(initializer=tf.constant_initializer(0.1),
                                 shape=[4096], dtype=tf.float32, trainable=True, name='biases')
        fc = tf.nn.relu_layer(outputs['fc6'], weights, biases, name='fc')
        if dropout==None:
            dropout_layer = tf.nn.dropout(fc, 1, name='dropout')
        else:
            dropout_layer = tf.nn.dropout(fc, dropout, name='dropout')
        outputs['fc7'] = dropout_layer
                              
    with tf.variable_scope('fc8') as scope:
        weights = tf.get_variable(shape=[4096,1000], dtype=tf.float32, 
                                  initializer=tf.truncated_normal_initializer(stddev=.01), name='weights')
        biases = tf.get_variable(initializer=tf.constant_initializer(0),
                                 shape=[1000], dtype=tf.float32, trainable=True, name='biases')
        fc = tf.nn.xw_plus_b(outputs['fc7'], weights, biases)
        outputs['fc8'] = fc
        outputs['pred'] = fc

    
    ### END OF YOUR CODE
    for k in ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'pool1',
            'pool2', 'pool5', 'fc6', 'fc7', 'fc8', 'conv1_kernel', 'pred']:
        assert k in outputs, '%s was not found in outputs' % k

    return outputs, {}


def lenet_model(inputs, train=True, norm=True, **kwargs):
    """
    Approximate LeNet model definition as defined in the paper. Uses similar conv/feature sizes
    but implements relu, maxpooling, LRN. Basically a very small version of AlexNet.
    http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=726791
    
    """
    # propagate input targets
    outputs = inputs
    input_to_network = inputs['images']
    
    with tf.variable_scope('le_conv1') as scope:
        weights = tf.get_variable(shape=[5, 5, 3, 128], dtype=tf.float32, 
                                  initializer=tf.contrib.layers.xavier_initializer(), name='weights')
        conv = tf.nn.conv2d(input_to_network, weights,[1, 4, 4, 1], padding='SAME')
        biases = tf.get_variable(initializer=tf.constant_initializer(0),
                                 shape=[128], dtype=tf.float32, trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        relu = tf.nn.relu(bias, name='relu')
        lrn = tf.nn.local_response_normalization(relu, depth_radius=5, bias=2, alpha=.0001, beta=.75) # bias=Kappa?
        pool = tf.nn.max_pool(value=lrn, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool')
        outputs['conv1_kernel'] = weights
        outputs['conv1'] = lrn
        outputs['pool1'] = pool
        
    with tf.variable_scope('le_conv2') as scope:
        weights = tf.get_variable(shape=[3, 3, 128, 128], dtype=tf.float32, 
                                  initializer=tf.contrib.layers.xavier_initializer(), name='weights')
        conv = tf.nn.conv2d(outputs['pool1'], weights,[1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable(initializer=tf.constant_initializer(0),
                                 shape=[128], dtype=tf.float32, trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        relu = tf.nn.relu(bias, name='relu')
        lrn = tf.nn.local_response_normalization(relu, depth_radius=5, bias=2, alpha=.0001, beta=.75) # bias=Kappa?
        pool = tf.nn.max_pool(value=lrn, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool')
        outputs['conv2_kernel'] = weights
        outputs['conv2'] = lrn
        outputs['pool2'] = pool
        
    with tf.variable_scope('le_conv3') as scope:
        weights = tf.get_variable(shape=[3, 3, 128, 128], dtype=tf.float32, 
                                  initializer=tf.contrib.layers.xavier_initializer(), name='weights')
        conv = tf.nn.conv2d(outputs['pool2'], weights,[1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable(initializer=tf.constant_initializer(0),
                                 shape=[128], dtype=tf.float32, trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        relu = tf.nn.relu(bias, name='relu')
        lrn = tf.nn.local_response_normalization(relu, depth_radius=5, bias=2, alpha=.0001, beta=.75) # bias=Kappa?
        pool = tf.nn.max_pool(value=lrn, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool')
        outputs['conv3_kernel'] = weights
        outputs['conv3'] = lrn
        outputs['pool3'] = pool
        
    with tf.variable_scope('le_fc') as scope:
        shape = np.product(outputs['pool3'].shape.as_list()[1:])
        flatten = tf.reshape(outputs['pool3'], [-1, shape]) 
        weights = tf.get_variable(shape=[shape,1000], dtype=tf.float32, 
                                  initializer=tf.truncated_normal_initializer(), name='weights')
        biases = tf.get_variable(initializer=tf.constant_initializer(0),
                                 shape=[1000], dtype=tf.float32, trainable=True, name='biases')
        wb = tf.nn.xw_plus_b(flatten, weights, biases, name='fc')
        fc = tf.nn.relu(wb, name='relu') 
        outputs['fc3'] = fc
        outputs['pred'] = fc
    
    ### END OF YOUR CODE
    for k in ['conv1', 'conv2', 'pred']:
        assert k in outputs, '%s was not found in outputs' % k

    return outputs, {}

def gabor_model(inputs, train=True, norm=True, **kwargs):
    """
    Hard-coded one-layer network with a convolutional layer of gabor filters followed by relu, 
    max pooling and local response normalization.
    
    """
    
    # propagate input targets
    outputs = inputs
    dropout = .5 if train else None
    input_to_network = inputs['images']
    
    with tf.variable_scope('conv1') as scope:
        #will be 96 43x43 filters with 6 spatial frequencies and 16 orientations
        weights = tf.get_variable(shape=[43, 43, 3, 96], dtype=tf.float32, 
                                  initializer=tf.constant_initializer(gabor_initializer()), trainable=False, name='weights')
        conv = tf.nn.conv2d(input_to_network, weights,[1, 6, 6, 1], padding='SAME')#want to produce ~30x30 outputs
        biases = tf.Variable(tf.constant(0.0, shape=[96], dtype=tf.float32), trainable=False, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        relu = tf.nn.relu(bias, name='relu')
        pool = tf.nn.max_pool(value=relu, ksize=[1, 3, 3, 1], strides=[1, 3, 3, 1], padding='SAME', name='pool')
        lrn = tf.nn.local_response_normalization(pool, depth_radius=5, bias=2, alpha=.0001, beta=.75) # bias=Kappa?
        
        #shape = np.product(lrn.shape.as_list()[1:])
        #flatten = tf.reshape(lrn, [-1, shape]) 
        #dummyBias = tf.get_variable(shape=[shape,1000], dtype=tf.float32, 
        #                          initializer=tf.contrib.layers.xavier_initializer(), trainable=True, name='dummyBias')
        #dummyFC = tf.nn.bias_add(flatten, dummyBias, name='dummyFC')
        
        
        outputs['conv1_kernel'] = weights
        outputs['conv1'] = lrn
        
    with tf.variable_scope('fc2') as scope:
        shape = np.product(outputs['conv1'].shape.as_list()[1:])
        flatten = tf.reshape(outputs['conv1'], [-1, shape])
        weights = tf.get_variable(shape=[16224,1000], dtype=tf.float32, 
                                  initializer=tf.contrib.layers.xavier_initializer(), name='weights')
        biases = tf.Variable(tf.constant(0, shape=[1000], dtype=tf.float32), trainable=True, name='biases')
        fc = tf.nn.xw_plus_b(flatten, weights, biases)
        
        outputs['pred'] = fc
    
    ### END OF YOUR CODE
    for k in ['conv1', 'conv1_kernel', 'pred']:
        assert k in outputs, '%s was not found in outputs' % k

    return outputs, {}

def gabor_initializer(): #creates a filter bank of 43x43 gabors with 16 orientations and 6 spatial frequencies.
    from skimage.filters import gabor_kernel
    #import skimage.filters.gabor_kernel as gabor_kernel #doesn't work either...
    from skimage import io
    import matplotlib.pyplot as plt 
    import math
    import numpy as np
    kernels = np.zeros((43,43,3,96))
    bandwidths = (.16,.24,.32,.48,.9,1.6)#makes each filter 43x43
    for sfIndex,spatFreq in enumerate((1/2., 1/3., 1/4., 1/6., 1/11., 1/18.)):
        for oriIndex,orientation in enumerate(range(1,17)):
            newKernel = gabor_kernel(frequency=spatFreq,theta=math.pi*2.*(orientation/16.),bandwidth=bandwidths[sfIndex])
            newKernel = newKernel.real
            kernels[:,:,:,sfIndex*16+oriIndex]=np.zeros((43,43,3))
            mismatch = 43-len(newKernel)
            border = int(np.floor(mismatch/2.))
            kernels[border:border+len(newKernel),border:border+len(newKernel),0,sfIndex*16+oriIndex]=newKernel
            kernels[border:border+len(newKernel),border:border+len(newKernel),1,sfIndex*16+oriIndex]=newKernel
            kernels[border:border+len(newKernel),border:border+len(newKernel),2,sfIndex*16+oriIndex]=newKernel
            kernels = np.float32(kernels)
    return kernels
