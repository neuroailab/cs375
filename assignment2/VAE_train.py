"""
Welcome to CS375! This is the starter file for assignment 2 in which you will 
train unsupervised networks. Since you should be familiar with tfutils by now
from assignment 1 the only thing that we provide is the config for the 
dataprovider and the dataproviders themselves as you will be also training 
and testing on CIFAR 10 in this assignment. You should be able to setup 
the rest of the code yourself.
You can find the instructions in assignment2.pdf in this folder.

Good luck with assigment 2!
"""

import os
import numpy as np
from VAE_models import VAE, VAE_loss, VAE_validation
import tensorflow as tf
from tfutils import base, data, model, optimizer, utils
from dataprovider import CIFAR10DataProvider

class CIFAR10Experiment():
    """
    Defines the CIFAR10 training experiment
    """
    class Config():
        """
        Holds model hyperparams and data information.
        The config class is used to store various hyperparameters and dataset
        information parameters. 
        """
        batch_size = 256
        data_path = '/datasets/cifar10/tfrecords'
        seed = 5
        crop_size = 24
        thres_loss = 1000000000000000
        n_epochs = 60
        train_steps = CIFAR10DataProvider.N_TRAIN / batch_size * n_epochs
        val_steps = np.ceil(CIFAR10DataProvider.N_VAL / batch_size).astype(int)


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
                # Cifar 10 data provider arguments
                'func': CIFAR10DataProvider,
                'data_path': self.Config.data_path,
                'group': 'train',
                'crop_size': self.Config.crop_size,
                # TFRecords (super class) data provider arguments
                'file_pattern': 'train*.tfrecords',
                'batch_size': self.Config.batch_size,
                'shuffle': False,
                'shuffle_seed': self.Config.seed,
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
            'valid0': {
                'data_params': {
                    # Cifar 10 data provider arguments
                    'func': CIFAR10DataProvider,
                    'data_path': self.Config.data_path,
                    'group': 'val',
                    'crop_size': self.Config.crop_size,
                    # TFRecords (super class) data provider arguments
                    'file_pattern': 'test*.tfrecords',
                    'batch_size': self.Config.batch_size,
                    'shuffle': False,
                    'shuffle_seed': self.Config.seed,
                    'n_threads': 4,
                },
                'queue_params': {
                    'queue_type': 'fifo',
                    'batch_size': self.Config.batch_size,
                    'seed': self.Config.seed,
                    'capacity': self.Config.batch_size * 10,
                    'min_after_dequeue': self.Config.batch_size * 5,
                },
                'targets': { 'func': VAE_validation},
                'num_steps': self.Config.val_steps,
                'agg_func': self.agg_mean, 
                'online_agg_func': self.online_agg_mean,
            }
        }

        """
        model_params defines the model i.e. the architecture that 
        takes the output of the data provider as input and outputs 
        the prediction of the model.
        """
        params['model_params'] = {
            'func': VAE
        }

        """
        loss_params defines your training loss.
        """
        params['loss_params'] = {
            'targets': ['images'],
            'loss_per_case_func': VAE_loss,
            'loss_per_case_func_params' : {'_outputs': 'outputs', 
                '_targets_$all': 'inputs'},
            'agg_func': tf.reduce_mean
        }

        """
        learning_rate_params defines the learning rate, decay and learning function.
        """
        params['learning_rate_params'] = {
            'learning_rate': 5e-4,
            'decay_steps': 100,
            'decay_rate': 0.95,
            'staircase': True,
        }

        """
        optimizer_params defines the optimizer.
        """
        params['optimizer_params'] = {
            'func': optimizer.ClipOptimizer,
            'optimizer_class': tf.train.AdamOptimizer,
            'clip': False,
        }

        """
        save_params defines how, where and when your training results are saved
        in the database.
        """
        params['save_params'] = {
            'host': 'localhost',
            'port': 24444,
            'dbname': 'assignment2',
            'collname': 'VAE',
            'exp_id': '1st_experiment',
            'save_valid_freq': 1000,
            'save_filters_freq': 2000,
            'cache_filters_freq': 5000,
            'save_metrics_freq': 2000,
            'save_initial_filters' : False,
            'save_to_gfs': []
        }

        """
        load_params defines how and if a model should be restored from the database.
        """
        params['load_params'] = {
            'host': 'localhost',
            'port': 24444,
            'dbname': 'assignment2',
            'collname': 'VAE',
            'exp_id': '1st_experiment',
            'do_restore': False,
            'load_query': None,
        }

        return params

    def agg_mean(self, results):
        for k in results:
            if k in ['pred', 'gt']:
                results[k] = results[k][0]
            elif k is 'total_loss':
                results[k] = np.mean(results[k])
            else:
                raise KeyError('Unknown target')
        return results

    
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
            if k in ['pred', 'gt']:
                value = v
            else:
                value = np.mean(v)
            agg_res[k].append(value)
        return agg_res
    
if __name__ == '__main__':
    """
    Illustrates how to run the configured model using tfutils
    """
    base.get_params()
    m = CIFAR10Experiment()
    params = m.setup_params()
    base.train_from_params(**params)
