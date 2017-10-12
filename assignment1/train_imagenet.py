"""
Welcome to CS375! This is the first part of assignment 1 in which we will
setup an experiment to train AlexNet on ImageNet from scratch.
This document will walk you through setting up an experiment using tfutils 
a tool that we have developed to facilitate using tensorflow.
tfutils biggest advantage is that it provides an easy interface for storing
and loading experiments from a mongodb database. It also provides other useful
features such as data providers for reading tfrecord files or simple ways to
parallelize models across GPU's.
The goal of this tutorial is to learn how to setup an experiment that trains
a standard AlexNet on the ImageNet classification task.

By the time you are reading this document you should have setup your working
environment as described in the Wiki of this repository:
    https://github.com/neuroailab/cs375/wiki
If you haven't yet done so please follow the instructions to setup your
environment!

In the following, we will introduce you to some of tfutils' functionality
that you will need to successfully complete this class.
As you can see at the very end of this file, all you need to do to start 
training a model with tfutils is to call
    base.train_from_params(**params)
This function takes a single argument: A dictionary of parameters that describes
    - the model saving ('save_params')
    - the model loading/restoring ('load_params')
    - the training procedure ('train_params')
    - the validation procedure ('validation_params')
    - the optimizer ('optimizer_params')
    - the learning rate policy ('learning_rate_params')
    - and the model definition ('model_params')

So let's get started with configuring our experiment. For this, you will be 
mostly working in the setup_params function. Please scroll down to 
    def setup_params(self):
to continue the tutorial. Get ready for some coding!

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
from tfutils import base, data, model, optimizer, utils
from dataprovider import ImageNetDataProvider
from models import alexnet_model


class ImageNetExperiment():
    """
    Defines the ImageNet training experiment
    """
    class Config():
        """
        Holds model hyperparams and data information.
        The config class is used to store various hyperparameters and dataset
        information parameters.
        Please set the seed to your group number. You can also change the batch
        size and n_epochs if you want but please do not change the rest.
        """
        batch_size = 256
        data_path = '/datasets/TFRecord_Imagenet_standard'
        seed = 0
        crop_size = 227
        thres_loss = 1000
        n_epochs = 90
        train_steps = ImageNetDataProvider.N_TRAIN / batch_size * n_epochs
        val_steps = np.ceil(ImageNetDataProvider.N_VAL / batch_size).astype(int)


    def setup_params(self):
        """
        This function illustrates how to setup up the parameters for 
        train_from_params. 
        """
        params = {}

        """
        train_params defines the training parameters consisting of 
            - the data provider that reads the data, preprocesses it and enqueues it into
              the data queue
            - the data queue that batches and if specified shuffles the data and provides 
              the input to the model
            - other configuration parameters like the number of training steps
        It's arguments are
            data_params: defines how the data is read in.
            queue_params: defines how the data is presented to the model, i.e.
            if it is shuffled or not and how big of a batch size is used.
            targets: the targets to be extracted and evaluated in the tensorflow session
            num_steps: number of training steps
            thres_loss: if the loss exceeds thres_loss the training will be stopped
            validate_first: run validation before starting the training
        """
        params['train_params'] = {
            'data_params': {
                # ImageNet data provider arguments
                'func': ImageNetDataProvider,
                'data_path': self.Config.data_path,
                'group': 'train',
                'crop_size': self.Config.crop_size,
                # TFRecords (super class) data provider arguments
                'file_pattern': 'train*.tfrecords',
                'batch_size': self.Config.batch_size,
                'shuffle': False,
                'shuffle_seed': self.Config.seed,
                'file_grab_func': self.subselect_tfrecords,
                'n_threads': 4,
            },
            'queue_params': {
                'queue_type': 'random',
                'batch_size': self.Config.batch_size,
                'seed': self.Config.seed,
                'capacity': self.Config.batch_size * 10,
                'min_after_dequeue': self.Config.batch_size * 5,
            },
            'targets': {
                'func': self.return_outputs,
                'targets': [],
            },
            'num_steps': self.Config.train_steps,
            'thres_loss': self.Config.thres_loss,
            'validate_first': False,
        }

        """
        validation_params similar to train_params defines the validation parameters.
        It has the same arguments as train_params and additionally
            agg_func: function that aggregates the validation results across batches,
                e.g. to calculate the mean of across batch losses
            online_agg_func: function that aggregates the validation results across
                batches in an online manner, e.g. to calculate the RUNNING mean across
                batch losses
        """

        params['validation_params'] = {
            'topn_val': {
                'data_params': {
                    # ImageNet data provider arguments
                    'func': ImageNetDataProvider,
                    'data_path': self.Config.data_path,
                    'group': 'val',
                    'crop_size': self.Config.crop_size,
                    # TFRecords (super class) data provider arguments
                    'file_pattern': 'validation*.tfrecords',
                    'batch_size': self.Config.batch_size,
                    'shuffle': False,
                    'shuffle_seed': self.Config.seed,
                    'file_grab_func': self.subselect_tfrecords,
                    'n_threads': 4,
                },
                'queue_params': {
                    'queue_type': 'fifo',
                    'batch_size': self.Config.batch_size,
                    'seed': self.Config.seed,
                    'capacity': self.Config.batch_size * 10,
                    'min_after_dequeue': self.Config.batch_size * 5,
                },
                'targets': {
                    'func': self.in_top_k,
                },
                'num_steps': self.Config.val_steps,
                'agg_func': self.agg_mean, 
                'online_agg_func': self.online_agg_mean,
            }
        }

        """
        model_params defines the model i.e. the architecture that 
        takes the output of the data provider as input and outputs 
        the prediction of the model.

        You will need to EDIT alexnet_model in models.py. alexnet_model 
        is supposed to define a standard AlexNet model in tensorflow. 
        Please open models.py and fill out the missing parts in the alexnet_model 
        function. Once you start working with different models you will need to
        switch out alexnet_model with your model function.
        """
        params['model_params'] = {
            'func': alexnet_model,
        }

        """
        loss_params defines your training loss.

        You will need to EDIT 'loss_per_case_func'. 
        Implement a softmax cross-entropy loss. You can use tensorflow's 
        tf.nn.sparse_softmax_cross_entropy_with_logits function.
        
        Note: 
        1.) loss_per_case_func is called with
                loss_per_case_func(inputs, outputs)
            by tfutils.
        2.) labels = outputs['labels']
            logits = outputs['pred']
        """
        def loss_wrapper(inputs, outputs):
            labels = outputs['labels']
            logits = outputs['pred']
            return tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        
        params['loss_params'] = {
            'targets': ['labels'],
            'agg_func': tf.reduce_mean,
            'loss_per_case_func': loss_wrapper,
            'loss_per_case_func_params' : {'_outputs': 'outputs', 
                '_targets_$all': 'inputs'},
            'loss_func_kwargs' : {},
        }

        """
        learning_rate_params defines the learning rate, decay and learning function.

        You will need to EDIT this part. Replace the exponential decay 
        learning rate policy with a piecewise constant learning policy.
        ATTENTION: 
        1.) 'learning_rate', 'decay_steps', 'decay_rate' and 'staircase' are not
        arguments of tf.train.piecewise_constant! You will need to replace
        them with the appropriate keys. 
        2.) 'func' passes global_step as input to your learning rate policy 
        function. Set the 'x' argument of tf.train.piecewise_constant to
        global_step.
        3.) set 'values' to [0.01, 0.005, 0.001, 0.0005] and
            'boundaries' to [150000, 300000, 450000] for a batch size of 256
        4.) You will need to delete all keys except for 'func' and replace them
        with the input arguments to 
        """
        def lr_wrapper(global_step, boundaries, values):
            boundaries = list(np.array(boundaries,dtype=np.int64))
            return tf.train.piecewise_constant(x=global_step, boundaries=boundaries, values=values)
        
        params['learning_rate_params'] = {	
            'func': lr_wrapper,
            'boundaries': [150000, 300000, 450000],
            'values': [0.01, 0.005, 0.001, 0.0005]
        }

        """
        optimizer_params defines the optimizer.

        You will need to EDIT the optimizer class. Replace the Adam optimizer
        with a momentum optimizer after switching the learning rate policy to
        piecewise constant.
        """
        params['optimizer_params'] = {
            'func': optimizer.ClipOptimizer,
            'optimizer_class': tf.train.MomentumOptimizer,
            'clip': False,
            'momentum': .9,
        }

        """
        save_params defines how, where and when your training results are saved
        in the database.

        You will need to EDIT this part. Set your 'host' (set it to 'localhost',
        or to IP if using remote mongodb), 'port' (set it to 24444, unless you 
        have changed mongodb.conf), 'dbname', 'collname', and 'exp_id'. 
        """
        params['save_params'] = {
            'host': 'localhost',
            'port': 24444,
            'dbname': 'assignment1',
            'collname': 'alexnet',
            'exp_id': '1st_experiment',
            'save_valid_freq': 10000,
            'save_filters_freq': 30000,
            'cache_filters_freq': 50000,
            'save_metrics_freq': 200,
            'save_initial_filters' : False,
            'save_to_gfs': [],
        }

        """
        load_params defines how and if a model should be restored from the database.

        You will need to EDIT this part. Set your 'host' (set it to 'localhost',
        or to IP if using remote mongodb), 'port' (set it to 24444, unless you 
        have changed mongodb.conf), 'dbname', 'collname', and 'exp_id'. 

        If you want to restore your training these parameters should be the same 
        as in 'save_params'.
        """
        params['load_params'] = {
            'host': 'localhost',
            'port': 24444,
            'dbname': 'assignment1',
            'collname': 'alexnet',
            'exp_id': '1st_experiment',
            'do_restore': True,
            'load_query': None,
        }

        return params


    def agg_mean(self, x):
        return {k: np.mean(v) for k, v in x.items()}


    def in_top_k(self, inputs, outputs):
        """
        Implements top_k loss for validation

        You will need to EDIT this part. Implement the top1 and top5 functions
        in the respective dictionary entry.
        """
        return {'top1': lambda outputs, inputs: tf.nn.in_top_k(predictions, targets, k=1),
                'top5': lambda outputs, inputs: tf.nn.in_top_k(predictions, targets, k=5)}


    def subselect_tfrecords(self, path):
        """
        Illustrates how to subselect files for training or validation
        """
        all_filenames = os.listdir(path)
        rng = np.random.RandomState(seed=SEED)
        rng.shuffle(all_filenames)
        return [os.path.join(path, fn) for fn in all_filenames
                if fn.endswith('.tfrecords')]


    def return_outputs(self, inputs, outputs, targets, **kwargs):
        """
        Illustrates how to extract desired targets from the model
        """
        retval = {}
        for target in targets:
            retval[target] = outputs[target]
        return retval


    def online_agg_mean(self, agg_res, res, step):
        """
        Appends the mean value for each key
        """
        if agg_res is None:
            agg_res = {k: [] for k in res}
        for k, v in res.items():
            agg_res[k].append(np.mean(v))
        return agg_res


if __name__ == '__main__':
    """
    Illustrates how to run the configured model using tfutils
    """
    base.get_params()
    m = ImageNetExperiment()
    params = m.setup_params()
    base.train_from_params(**params)
