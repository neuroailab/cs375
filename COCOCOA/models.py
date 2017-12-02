import os
import numpy as np
import tensorflow as tf

# *********************************************************************
# Helper Functions
# *********************************************************************

def dense_(inputs, shape, activation=tf.nn.softplus):
    """
    Args:
        shape: [input, output]
    """
    weights = tf.get_variable(shape=shape, dtype=tf.float32,
                              initializer=tf.random_normal_initializer(stddev=0.05),
                              regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                              name='weights')
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
                                  regularizer=tf.contrib.layers.l2_regularizer(reg),
                                  name='weights')
    else:
        weights = tf.get_variable(shape=conv_shape, dtype=tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer(), name='weights')
    conv = tf.nn.conv2d(inp, weights,[1, stride, stride, 1], padding=padding, name='conv')
    biases = tf.get_variable(initializer=tf.zeros_initializer(), shape=[conv_shape[3]], dtype=tf.float32, name='bias')
    out = tf.nn.bias_add(conv, biases)
    return out, weights


# *********************************************************************
# BRAIN model
# *********************************************************************


def brain_model(inputs, conv_parameters,  hidden_dim,
                attention_layer='conv1', attention_type='feature',
                attention_mechanism='flat'):
    """ 
    Define BRAIN model
    
    Args:
        conv_parameters: list of parameters. Each element should be [filter_height, filter_width, 
                                                                    in_channels, out_channels, stride]
        hidden_dim: size of FC layer after convolutions
        attention_layer: the name of the layer that attention will boost
        attention_type: either "feature" or "spacial"
        attention_mechanism: either "flat" or "deconv". Determines whether the attentional bias is constructed via
            deconvolution
    """
    batch_size = inputs.shape[0]
    layers = {}
    layer_weights = {}
    for n, conv_param in enumerate(conv_parameters):
        layer_name = 'conv%s' % n
        stride = conv_param[-1]
        conv_shape = conv_param[0:4]
        if n == 1:
            inputs = inputs['images']
        else:
            inputs = layers['conv%s' % n-1]
        with tf.variable_scope(layer_name):
            conv, c_k = conv_(inputs,conv_shape,stride,padding='SAME')
            layers[layer_name] = conv
            layers[layer_weights] = c_k
    last_conv = layers[layer_name]
    
    # hidden layer where the goal is incorporated with sensory information
    with tf.variable_scope('fc'):
        flatten = tf.contrib.layers.flatten(last_conv)
        # concatenate flattened image with one-hot target
        fc_input = tf.concat([flatten, inputs['target']], axis=1)
        FC = dense_(fc_input, [batch_size, hidden_dim], activation=None)
        layers['FC_hidden'] = FC
    
    
    if attention_mechanism == "deconv":
        
    else:
        attention_bias = FC
        
        
    
    
            
        