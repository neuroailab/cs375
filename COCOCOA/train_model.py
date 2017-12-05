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

        params['model_params'] = {
            'func': brain_model,
        }


        params['loss_params'] = {
            'targets': ['labels'],
            'agg_func': tf.reduce_mean,
            'loss_per_case_func': loss_wrapper,
            'loss_per_case_func_params' : {'_outputs': 'outputs', 
                '_targets_$all': 'inputs'},
            'loss_func_kwargs' : {},
        }

        def lr_wrapper(global_step, boundaries, values):
            boundaries = list(np.array(boundaries,dtype=np.int64))
            return tf.train.piecewise_constant(x=global_step, boundaries=boundaries, values=values)
        
        params['learning_rate_params'] = {	
            'func': lr_wrapper,
            'boundaries': [150000, 300000, 450000],
            'values': [0.01, 0.005, 0.001, 0.0005]
        }

        params['optimizer_params'] = {
            'func': optimizer.ClipOptimizer,
            'optimizer_class': tf.train.MomentumOptimizer,
            'clip': False,
            'momentum': .9,
        }


        params['save_params'] = {
            'host': 'localhost',
            'port': 24444,
            'dbname': 'assignment1',
            'collname': 'alexnet',
            'exp_id': '2nd_experiment',
            'save_valid_freq': 10000,
            'save_filters_freq': 30000,
            'cache_filters_freq': 50000,
            'save_metrics_freq': 200,
            'save_initial_filters' : False,
            'save_to_gfs': [],
        }

        params['load_params'] = {
            'host': 'localhost',
            'port': 24444,
            'dbname': 'assignment1',
            'collname': 'alexnet',
            'exp_id': '2nd_experiment',
            'do_restore': True,
            'load_query': None,
        }

        return params


    def agg_mean(self, x):
        return {k: np.mean(v) for k, v in x.items()}



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
    base.get_params()
    m = ImageNetExperiment()
    params = m.setup_params()
    base.train_from_params(**params)
