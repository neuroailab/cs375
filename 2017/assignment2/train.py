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
        seed = 0
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
            }
        }

        """
        model_params defines the model i.e. the architecture that 
        takes the output of the data provider as input and outputs 
        the prediction of the model.
        """
        params['model_params'] = {
        }

        """
        loss_params defines your training loss.
        """
        params['loss_params'] = {
        }

        """
        learning_rate_params defines the learning rate, decay and learning function.
        """
        params['learning_rate_params'] = {
        }

        """
        optimizer_params defines the optimizer.
        """
        params['optimizer_params'] = {
        }

        """
        save_params defines how, where and when your training results are saved
        in the database.
        """
        params['save_params'] = {
        }

        """
        load_params defines how and if a model should be restored from the database.
        """
        params['load_params'] = {
        }

        return params

if __name__ == '__main__':
    """
    Illustrates how to run the configured model using tfutils
    """
    base.get_params()
    m = CIFAR10Experiment()
    params = m.setup_params()
    base.train_from_params(**params)
