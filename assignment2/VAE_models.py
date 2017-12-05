import numpy as np
import tensorflow as tf

def get_FC(inputs, shape, activation=tf.nn.tanh):
    """
    Args:
        shape: [input, output]
    """
    weights = tf.get_variable('weights', shape, tf.float32, tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable('biases', [shape[1]], tf.float32, tf.zeros_initializer())
    FC = tf.nn.xw_plus_b(inputs, weights, biases, name='FC')
    if activation is not None:
        out = activation(FC)
    else:
        out = FC
    return out

def get_conv(inp, conv_shape, stride, activation=tf.nn.tanh):
    print(conv_shape)    
    weights = tf.get_variable(shape=conv_shape, dtype=tf.float32, 
                              initializer=tf.contrib.layers.xavier_initializer(), name='weights')
    conv = tf.nn.conv2d(inp, weights,[1, stride, stride, 1], padding='SAME', name='conv')
    biases = tf.get_variable(initializer=tf.zeros_initializer(), shape=[conv_shape[-1]], dtype=tf.float32, name='biases')
    bias = tf.nn.bias_add(conv, biases)
    if activation is not None:
        act = activation(bias)
    else:
        act = bias
    out = tf.contrib.layers.batch_norm(act, scale=True)
    return out, weights

def get_deconv(inp, conv_shape, stride, output_shape, activation=tf.nn.tanh):
    weights = tf.get_variable(shape=conv_shape, dtype=tf.float32, 
                              initializer=tf.contrib.layers.xavier_initializer(), name='weights')
    conv = tf.nn.conv2d_transpose(inp, weights, output_shape, [1, stride, stride, 1],padding='SAME', name='deconv')
    biases = tf.get_variable(initializer=tf.zeros_initializer(), shape=[conv_shape[2]], dtype=tf.float32, name='biases')
    bias = tf.nn.bias_add(conv, biases)
    if activation is not None:
        out = activation(bias)
    else:
        out = bias
    return out
    
def VAEold(inputs, train=True,  n_latent=50, **kwargs):
    outputs = inputs
    filter1_size = [4,4,3,64]
    dropout_rate = .5
    # encoder
    with tf.variable_scope('conv1'):
        conv1, conv1_weights = get_conv(inputs['images'], filter1_size, 2)
    flat_len = np.product(conv1.shape.as_list()[1:])
    flatten = tf.reshape(conv1, [-1, flat_len]) 
    #with tf.variable_scope('latent'):
    #    latent = get_FC(flatten, [flat_len, n_hidden])
    with tf.variable_scope('mu'):
        mu = get_FC(flatten, [flat_len, n_latent], activation=None)
    with tf.variable_scope('logstd'):
        logstd = get_FC(flatten, [flat_len, n_latent], activation=None)
    outputs['conv1'] = conv1
    outputs['conv1_weights'] = conv1_weights
    outputs['mu'] = mu
    outputs['logstd'] = logstd
    
    # magic reparameterization trick to the rescue!
    noise = tf.random_normal([1, n_latent])
    z = tf.add(mu, tf.multiply(noise, tf.exp(.5*logstd)), name='latent_encoding') # where the magic happens
    outputs['z'] = z
    
    # decoder
    with tf.variable_scope('reconstruction'):
        reconstruction = get_FC(z, [n_latent, flat_len], tf.sigmoid)
        
    # deconvolve
    expanded = tf.reshape(reconstruction, conv1.shape, name='expand')
    with tf.variable_scope('pred'):
        deconv1 = get_deconv(expanded, filter1_size, 2, inputs['images'].shape, activation=None)
    outputs['reconstruction'] = reconstruction
    outputs['pred'] = deconv1
    return outputs, {}

def VAE(inputs, train=True,  n_latent=50, **kwargs):
    outputs = inputs
    filters = [[4,4,3,64], [4,4,64,128]]
    dropout_rate = .5
    # encoder
    with tf.variable_scope('conv1'):
        conv1, conv1_weights = get_conv(inputs['images'], filters[0], 2)
    with tf.variable_scope('conv2'):
        conv2, conv2_weights = get_conv(conv1, filters[1], 2)
        
    flat_len = np.product(conv2.shape.as_list()[1:])
    flatten = tf.reshape(conv2, [-1, flat_len]) 
    #with tf.variable_scope('latent'):
    #    latent = get_FC(flatten, [flat_len, n_hidden])
    with tf.variable_scope('mu'):
        mu = get_FC(flatten, [flat_len, n_latent], activation=None)
    with tf.variable_scope('logstd'):
        logstd = get_FC(flatten, [flat_len, n_latent], activation=None)
    outputs['conv1'] = conv1
    outputs['conv1_weights'] = conv1_weights
    outputs['conv2'] = conv2
    outputs['mu'] = mu
    outputs['logstd'] = logstd
    
    # magic reparameterization trick to the rescue!
    noise = tf.random_normal([1, n_latent])
    z = tf.add(mu, tf.multiply(noise, tf.exp(.5*logstd)), name='latent_encoding') # where the magic happens
    outputs['z'] = z
    
    # decoder
    with tf.variable_scope('reconstruction'):
        reconstruction = get_FC(z, [n_latent, flat_len], tf.sigmoid)
        
    # deconvolve
    expanded = tf.reshape(reconstruction, conv1.shape, name='expand')
    with tf.variable_scope('deconv2'):
        deconv2 = get_deconv(expanded, filters[2], 2, conv1.shape, activation=None)
    with tf.variable_scope('pred'):
        deconv1 = get_deconv(deconv2, filters[1], 2, inputs['images'].shape, activation=None)
    outputs['reconstruction'] = reconstruction
    outputs['pred'] = deconv1
    return outputs, {}

def VAE_loss(inputs, outputs, **kwargs):
    '''
    Defines the loss = l2(inputs - outputs) + l2(weights)
    '''
    # flatten the input images
    flat_len = inputs.get_shape().as_list()[0]
    inputs = tf.reshape(inputs, [flat_len, -1])
    pred = tf.reshape(outputs['pred'], [flat_len, -1])
    '''
    #from tutorial...odd
    # the total optimization target
    log_likelihood = tf.reduce_sum(inputs * tf.log(pred + 1e-9) + (1 - inputs) * tf.log(1 - pred + 1e-9), axis=1)
    KL_div = -0.5 * tf.reduce_sum(1 + 2 * outputs['logstd'] - tf.pow(outputs['mu'],2) - tf.exp(2 * outputs['logstd']), axis = 1)
    variational_lower_bound = tf.reduce_mean(log_likelihood - KL_div)
    return -variational_lower_bound

    '''
    # from http://kvfrans.com/variational-autoencoders-explained/
    KL = 0.5 * tf.reduce_sum(tf.square(outputs['mu']) + tf.square(outputs['logstd']) - tf.log(tf.square(outputs['logstd'])) - 1,1)
    reconstruction_loss = tf.nn.l2_loss(inputs - pred)
    return  reconstruction_loss + KL

       


def VAE_validation(inputs, outputs, **kwargs):
    '''
    Wrapper for using the loss function as a validation target
    '''
    return {'total_loss': VAE_loss(inputs['images'], outputs),
            'pred': outputs['pred'],
            'gt': inputs['images']}