from __future__ import absolute_import, division, print_function
from collections import OrderedDict

import numpy as np
import tensorflow as tf

class ConvNet(object):
    """Basic implementation of ConvNet class compatible with tfutils.
    """

    def __init__(
            self, 
            seed=None, 
            **kwargs):
        self.seed = seed
        self.output = None
        self._params = OrderedDict()

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, value):
        name = tf.get_variable_scope().name
        if name not in self._params:
            self._params[name] = OrderedDict()
        self._params[name][value['type']] = value

    @property
    def graph(self):
        return tf.get_default_graph().as_graph_def()

    def initializer(
            self, kind='xavier', 
            stddev=0.01):

        if kind == 'xavier':
            init = tf.contrib.layers.xavier_initializer(seed=self.seed)

        elif kind == 'trunc_norm':
            init = tf.truncated_normal_initializer(
                    mean=0, stddev=stddev, 
                    seed=self.seed)

        elif kind == 'variance_scaling_initializer':
            init = tf.contrib.layers.variance_scaling_initializer(
                    seed=self.seed)

        else:
            raise ValueError('Please provide an appropriate initialization '
                             'method: xavier or trunc_norm')
        return init

    @tf.contrib.framework.add_arg_scope
    def batchnorm(
            self, is_training, 
            in_layer=None, decay=0.997, 
            epsilon=1e-5):
        if not in_layer:
            in_layer = self.output
        self.output = tf.layers.batch_normalization(
                inputs=in_layer, axis=-1,
                momentum=decay, epsilon=epsilon, 
                center=True, scale=True, 
                training=is_training, fused=True,
                name=in_layer.name.replace(':', '__') + '_batchnorm'
                )
        return self.output

    @tf.contrib.framework.add_arg_scope
    def conv(self,
             out_shape,
             ksize=3,
             stride=1,
             padding='SAME',
             init='xavier',
             stddev=.01,
             bias=0,
             activation='relu',
             train=True,
             add_bn=False,
             weight_decay=None,
             in_layer=None,
             layer='conv',
             ):
        # Set parameters
        if in_layer is None:
            in_layer = self.output
        if not weight_decay:
            weight_decay = 0.

        # Get conv kernel shape
        in_shape = in_layer.get_shape().as_list()[-1]
        conv2d_strides = [1, stride, stride, 1]
        if isinstance(ksize, int):
            ksize1 = ksize
            ksize2 = ksize
        else:
            ksize1, ksize2 = ksize
        conv_k_shape = [ksize1, ksize2, in_shape, out_shape]

        # Define variable
        bias_init = tf.constant_initializer(bias)
        with tf.variable_scope(layer, reuse=tf.AUTO_REUSE):
            kernel = tf.get_variable(
                    initializer=self.initializer(init, stddev=stddev),
                    shape=conv_k_shape,
                    dtype=tf.float32,
                    regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                    name='weights')
            biases = tf.get_variable(
                    initializer=bias_init,
                    shape=[out_shape],
                    dtype=tf.float32,
                    regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                    name='bias')

        # Do the actual computation
        conv = tf.nn.conv2d(
                in_layer, kernel,
                strides=conv2d_strides,
                padding=padding,
                name = layer + '_conv')
        self.output = tf.nn.bias_add(
                conv, biases, 
                name= layer + '_convbias')

        # Whether adding batch normalization
        if add_bn:
            with tf.variable_scope(layer, reuse=tf.AUTO_REUSE):
                self.output = self.batchnorm(train)
        # Add activation
        if activation:
            self.output = self.activation(kind=activation)

        # Set parameters (this dict will be appended to the final params)
        self.params = {'input': in_layer.name,
                       'type': 'conv',
                       'num_filters': out_shape,
                       'stride': stride,
                       'kernel_size': (ksize1, ksize2),
                       'padding': padding,
                       'init': init,
                       'stddev': stddev,
                       'bias': bias,
                       'activation': activation,
                       'weight_decay': weight_decay,
                       'seed': self.seed}
        return self.output

    @tf.contrib.framework.add_arg_scope
    def fc(self,
           out_shape,
           init='xavier',
           stddev=.01,
           bias=1,
           activation='relu',
           dropout=.5,
           in_layer=None,
           weight_decay=None,
           layer='fc',
           ):

        # Set parameters
        if weight_decay is None:
            weight_decay = 0.

        if in_layer is None:
            in_layer = self.output
        resh = tf.reshape(in_layer,
                          [in_layer.get_shape().as_list()[0], -1],
                          name='reshape')
        in_shape = resh.get_shape().as_list()[-1]

        # Define variable
        bias_init = tf.constant_initializer(bias)
        with tf.variable_scope(layer, reuse=tf.AUTO_REUSE):
            kernel = tf.get_variable(
                    initializer=self.initializer(init, stddev=stddev),
                    shape=[in_shape, out_shape],
                    dtype=tf.float32,
                    regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                    name='weights')
            biases = tf.get_variable(
                    initializer=bias_init,
                    shape=[out_shape],
                    dtype=tf.float32,
                    regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                    name='bias')

        # Do the actual computation
        fcm = tf.matmul(resh, kernel, name=layer + '_mult')
        self.output = tf.nn.bias_add(fcm, biases, name=layer + '_bias')

        # Add activation or dropout
        if activation is not None:
            self.activation(kind=activation)
        if dropout is not None:
            self.dropout(dropout=dropout)

        self.params = {'input': in_layer.name,
                       'type': 'fc',
                       'num_filters': out_shape,
                       'init': init,
                       'bias': bias,
                       'stddev': stddev,
                       'activation': activation,
                       'dropout': dropout,
                       'weight_decay': weight_decay,
                       'seed': self.seed}
        return self.output

    @tf.contrib.framework.add_arg_scope
    def lrn(
            self,
            depth_radius=2,
            bias=1,
            alpha=0.0001,
            beta=.75,
            in_layer=None,
            layer='lrn'):
        if in_layer is None:
            in_layer = self.output
        with tf.variable_scope(layer, reuse=tf.AUTO_REUSE):
            self.output = tf.nn.lrn(
                    in_layer,
                    depth_radius=np.float(depth_radius),
                    bias=np.float(bias),
                    alpha=alpha,
                    beta=beta,
                    name='norm')
        self.params = {'input': in_layer.name,
                       'type': 'lrnorm',
                       'depth_radius': depth_radius,
                       'bias': bias,
                       'alpha': alpha,
                       'beta': beta}
        return self.output

    @tf.contrib.framework.add_arg_scope
    def pool(self,
             ksize=3,
             stride=2,
             padding='SAME',
             pool_type='maxpool',
             in_layer=None):
        # Set parameters
        if in_layer is None:
            in_layer = self.output
        if isinstance(ksize, int):
            ksize1 = ksize
            ksize2 = ksize
        else:
            ksize1, ksize2 = ksize
        if isinstance(stride, int):
            stride1 = stride
            stride2 = stride
        else:
            stride1, stride2 = stride
        ksizes = [1, ksize1, ksize2, 1]
        strides = [1, stride1, stride2, 1]

        # Do the pooling
        if pool_type=='maxpool':
            pool_func = tf.nn.max_pool
        else:
            pool_func = tf.nn.avg_pool
        self.output = pool_func(
                in_layer,
                ksize=ksizes,
                strides=strides,
                padding=padding,
                name='pool',
                )

        # Set params, return the value
        self.params = {
                'input':in_layer.name,
                'type':pool_type,
                'kernel_size': (ksize1, ksize2),
                'stride': stride,
                'padding': padding}
        return self.output

    def activation(self, kind='relu', in_layer=None):
        if in_layer is None:
            in_layer = self.output
        if kind == 'relu':
            out = tf.nn.relu(in_layer, name=in_layer.name.replace(':', '__') + '_relu')
        else:
            raise ValueError("Activation '{}' not defined".format(kind))
        self.output = out
        return out

    def dropout(self, dropout=.5, in_layer=None, **kwargs):
        if in_layer is None:
            in_layer = self.output
        self.output = tf.nn.dropout(
                in_layer, dropout, 
                seed=self.seed, name=in_layer.name.replace(':', '__') + '_dropout', 
                **kwargs)
        return self.output


def mnist(inputs, train=True, seed=0):
    m = ConvNet()
    fc_kwargs = {
            'init': 'xavier',
            'dropout': None,
            }
    m.fc(128, layer='hidden1', in_layer=inputs, **fc_kwargs)
    m.fc(32, layer='hidden2', **fc_kwargs)
    m.fc(10, activation=None, layer='softmax_linear', **fc_kwargs)

    return m


def alexnet(inputs, train=True, norm=True, seed=0, **kwargs):
    # Define model class and default kwargs for different types of layers
    m = ConvNet(seed=seed)
    conv_kwargs = {
            'add_bn': False,
            'init': 'xavier',
            'weight_decay': .0001,
            }
    pool_kwargs = {
            'pool_type': 'maxpool',
            }
    fc_kwargs = {
            'init': 'trunc_norm',
            'weight_decay': .0001,
            'stddev': .01,
            }
    dropout = .5 if train else None

    # Actually define the network
    m.conv(
            96, 11, 4, padding='VALID', 
            layer='conv1', in_layer=inputs, **conv_kwargs)
    if norm:
        m.lrn(depth_radius=5, bias=1, alpha=.0001, beta=.75, layer='conv1')
    m.pool(3, 2, **pool_kwargs)

    m.conv(256, 5, 1, layer='conv2', **conv_kwargs)
    if norm:
        m.lrn(depth_radius=5, bias=1, alpha=.0001, beta=.75, layer='conv2')
    m.pool(3, 2, **pool_kwargs)

    m.conv(384, 3, 1, layer='conv3', **conv_kwargs)
    m.conv(384, 3, 1, layer='conv4', **conv_kwargs)

    m.conv(256, 3, 1, layer='conv5', **conv_kwargs)
    m.pool(3, 2, **pool_kwargs)

    m.fc(4096, dropout=dropout, bias=.1, layer='fc6', **fc_kwargs)
    m.fc(4096, dropout=dropout, bias=.1, layer='fc7', **fc_kwargs)
    m.fc(1000, activation=None, dropout=None, bias=0, layer='fc8', **fc_kwargs)

    return m


def mnist_tfutils(inputs, train=True, **kwargs):
    m = mnist(inputs['images'], train=train)
    return m.output, m.params


def alexnet_tfutils(inputs, **kwargs):
    m = alexnet(inputs['images'], **kwargs)
    return m.output, m.params
