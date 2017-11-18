from __future__ import division, print_function, absolute_import
import os, sys
from collections import OrderedDict
import numpy as np
import pymongo as pm

import tensorflow as tf

from tfutils import base, data, model, optimizer, utils

import copy

NUM_GPUS = 1
if not isinstance(NUM_GPUS, list):
    DEVICES = ['/gpu:' + str(i) for i in range(NUM_GPUS)]
else:
    DEVICES = ['/gpu:' + str(i) for i in range(len(NUM_GPUS))]

MODEL_PREFIX = 'model_0'
MB_SIZE = 2000
# Data parameters
INPUT_BATCH_SIZE = 1024 # queue size
IMAGE_SIZE_RESIZE = 50

WN_DATA_PATH = '/datasets/deepretina_data/tf_records/whitenoise'
NS_DATA_PATH = '/datasets/deepretina_data/tf_records/naturalscene'

# data provider
class retinaTF(data.TFRecordsParallelByFileProvider):

  def __init__(self,
               source_dirs,
               resize=IMAGE_SIZE_RESIZE,
               **kwargs
               ):

    if resize is None:
      self.resize = 50
    else:
      self.resize = resize

    postprocess = {'images': [], 'labels': []}
    postprocess['images'].insert(0, (tf.decode_raw, (tf.float32, ), {})) 
    postprocess['images'].insert(1, (tf.reshape, ([-1] + [50, 50, 40], ), {}))
    postprocess['images'].insert(2, (self.postproc_imgs, (), {})) 

    postprocess['labels'].insert(0, (tf.decode_raw, (tf.float32, ), {})) 
    postprocess['labels'].insert(1, (tf.reshape, ([-1] + [5], ), {}))

    super(retinaTF, self).__init__(
      source_dirs,
      postprocess=postprocess,
      **kwargs
    )


  def postproc_imgs(self, ims):
    def _postprocess_images(im):
        im = tf.image.resize_images(im, [self.resize, self.resize])
        return im
    return tf.map_fn(lambda im: _postprocess_images(im), ims, dtype=tf.float32)

def ln(inputs, train=True, prefix=MODEL_PREFIX, devices=DEVICES, num_gpus=NUM_GPUS, seed=0, cfg_final=None):
    params = OrderedDict()
    batch_size = inputs['images'].get_shape().as_list()[0]
    params['train'] = train
    params['batch_size'] = batch_size

    # implement your LN model here
    flat = tf.contrib.layers.flatten(inputs['images'])
    num_units = 5
    out = tf.layers.dense(
                flat, 
                num_units, 
                activation=tf.nn.softplus, 
                kernel_regularizer=tf.contrib.layers.l2_regularizer(1E-3))

    return out, params

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

def gaussian_noise_(input_layer, std):
    noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32) 
    return input_layer + noise

def relu_(inp):
    return tf.nn.relu(inp)

def cnn(inputs, train=True, prefix=MODEL_PREFIX, devices=DEVICES, num_gpus=NUM_GPUS, seed=0, cfg_final=None):
    params = OrderedDict()
    batch_size = inputs['images'].get_shape().as_list()[0]
    params['train'] = train
    params['batch_size'] = batch_size

    with tf.variable_scope('conv1'):
        temp, c1_k = conv_(inputs['images'],[15,15,40,16],1,padding='VALID')
        if train:
            conv1 = relu_(gaussian_noise_(temp,std=0.1))
        else:
            conv1 = relu_(temp)
            
    with tf.variable_scope('conv2'):
        temp, c2_k = conv_(conv1,[9,9,16,8],1,padding='VALID',reg=1e-3)
        if train:
            conv2 = relu_(gaussian_noise_(temp,std=0.1))
        else:
            conv2 = relu_(temp)
    
    with tf.variable_scope('fc'):
        flat_len = np.product(conv2.shape.as_list()[1:])
        flatten = tf.reshape(conv2, [-1, flat_len]) 
        out = dense_(flatten,[flat_len,5])

    return out, params

def pearson_agg(results):
    # concatenate results along batch dimension
    true_rates = np.concatenate(results['labels'], axis=0)
    pred_rates = np.concatenate(results['pred'], axis=0)

    true_std = np.std(true_rates, axis=0)
    pred_std = np.std(pred_rates, axis=0)

    true_mean = np.mean(true_rates, axis=0)
    pred_mean = np.mean(pred_rates, axis=0)

    r = np.mean( (true_rates - true_mean) * (pred_rates - pred_mean), axis=0 ) / (true_std * pred_std)
    return {'pearson' : r}

def online_agg(agg_res, res, step):
    if agg_res is None:
        agg_res = {k: [] for k in res}
    for k, v in res.items():
        agg_res[k].append(v)
    return agg_res

def return_outputs(inputs, outputs, targets, **kwargs):
    """
    Illustrates how to extract desired targets from the model
    """
    retval = {}
    retval['labels'] = inputs['labels']
    retval['pred'] = outputs
    return retval

# model parameters

default_params = {
    'save_params': {
        'host': '35.199.154.71',
        'port': 24444,
        'dbname': 'deepretina',
        'exp_id': 'trainval0',
    },

    'load_params': {
        'host': '35.199.154.71',
        'port': 24444,
        'do_restore': True,
        'query': None
    },

    'model_params': {
        'func': ln,
        'num_gpus': NUM_GPUS,
        'devices': DEVICES,
        'prefix': MODEL_PREFIX
    },

    'validation_params': {
        'whitenoise_pearson': {
            'data_params': {
                'func': retinaTF,
                'source_dirs': [os.path.join(WN_DATA_PATH, 'images'), os.path.join(WN_DATA_PATH, 'labels')],
                'resize': IMAGE_SIZE_RESIZE,
                'batch_size': INPUT_BATCH_SIZE,
                'file_pattern': 'test*.tfrecords',
                'n_threads': 4
            },
            'targets': {
                'func': return_outputs,
                'targets': ['labels'],
            },
            'queue_params': {
                'queue_type': 'fifo',
                'batch_size': MB_SIZE,
                'capacity': 11*INPUT_BATCH_SIZE,
                'min_after_dequeue': 10*INPUT_BATCH_SIZE,
                'seed': 0,
            },
            'num_steps': 5957 // MB_SIZE + 1,
            'agg_func': pearson_agg,
            'online_agg_func': online_agg
        },
        'naturalscene_pearson': {
            'data_params': {
                'func': retinaTF,
                'source_dirs': [os.path.join(NS_DATA_PATH, 'images'), os.path.join(NS_DATA_PATH, 'labels')],
                'resize': IMAGE_SIZE_RESIZE,
                'batch_size': INPUT_BATCH_SIZE,
                'file_pattern': 'test*.tfrecords',
                'n_threads': 4
            },
            'targets': {
                'func': return_outputs,
                'targets': ['labels'],
            },
            'queue_params': {
                'queue_type': 'fifo',
                'batch_size': MB_SIZE,
                'capacity': 11*INPUT_BATCH_SIZE,
                'min_after_dequeue': 10*INPUT_BATCH_SIZE,
                'seed': 0,
            },
            'num_steps': 5956 // MB_SIZE + 1,
            'agg_func': pearson_agg,
            'online_agg_func': online_agg
        }
    },
    'log_device_placement': False,  # if variable placement has to be logged
}

def test_ln(steps=None, train_stimulus='whitenoise'):
    params = copy.deepcopy(default_params)
    for param in ['save_params', 'load_params']:
        params[param]['dbname'] = 'ln_model'
        params[param]['collname'] = train_stimulus
        params[param]['exp_id'] = 'trainval0'
    params['model_params']['func'] = ln
    
     # determine time steps
    if steps is None:
        conn = pm.MongoClient(port=params['load_params']['port'])
        coll = conn[params['load_params']['dbname']][train_stimulus + '.files']
        steps = [i['step'] for i in coll.find({'exp_id': 'trainval0', 
                                               'train_results': {'$exists': True}}, projection=['step'])]
    for step in steps:
        print("Running Step %s" % step)
        params['load_params']['query'] = {'step': step}
        params['save_params']['exp_id'] = 'testval_step%s' % step
        base.test_from_params(**params)
        
def test_cnn(steps=None, train_stimulus='whitenoise'):
    params = copy.deepcopy(default_params)
    params['model_params']['func'] = cnn
    for param in ['save_params', 'load_params']:
        params[param]['dbname'] = 'cnn'
        params[param]['collname'] = train_stimulus
        params[param]['exp_id'] = 'trainval0'
    
    if steps is None:
        conn = pm.MongoClient(port=params['load_params']['port'])
        coll = conn[params['load_params']['dbname']][train_stimulus + '.files']
        steps = [i['step'] for i in coll.find({'exp_id': 'trainval0', 
                                               'train_results': {'$exists': True}}, projection=['step'])]
    print(params['load_params'])
    for step in steps:
        # determine time steps
        #print("Running Step %s" % step)
        params['load_params']['query'] = {'step': step}
        params['save_params']['exp_id'] = 'testval_step%s' % step
        #base.test_from_params(**params)
 
if __name__ == '__main__':
    # Set stim_type (at the top of this file) to change the data input to the models.
    # Set the stimulus paSram below to load the model trained on [stimulus].
    # i.e. stim_type = whitenoise, stimulus = naturalscene means calculating the correlation coefficient
    # for whitenoise data on the model trained on naturalscene data.
    # Set the step below to change the model checkpoint.
    for stimulus in ['whitenoise','naturalscene']:
        test_cnn(train_stimulus=stimulus)
        #test_ln(train_stimulus=stimulus)
