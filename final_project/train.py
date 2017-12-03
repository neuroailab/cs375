"""
Final project
"""


import os
import numpy as np
import tensorflow as tf
from tfutils import base, data, model, optimizer, utils
from ImageNetDataProvider import ImageNetDataProvider
from yolo_tiny_net import YoloTinyNet

class ImageNetYOLO():
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
        common_params = {
            'image_size': crop_size,
            'num_classes': 20,
            'batch_size': 1
            }
        net_params = {
            'boxes_per_cell': 2,
            'weight_decay': 0.0005,
            'cell_size': 4,
            'object_scale':1,
            'noobject_scale':1,
            'class_scale':1,
            'coord_scale':1
            }
        ytn = YoloTinyNet(common_params,net_params,test=False)
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
        
        """
        Using combine worlds
        'data_params': {
            'func': Combine_world,
            'cfg_dataset': {'imagenet': 0}
            '
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
            'func': self.Config.ytn.inference,
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
            logits = outputs['logits']
            return tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits) + 0.0 * tf.reduce_sum(outputs['bboxes'])
        
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
        
        params['learning_rate_params'] = {	
            'func': tf.train.exponential_decay,
            'learning_rate': 0.0001,
            'decay_steps': ImageNetDataProvider.N_TRAIN / self.Config.batch_size,
            'decay_rate': 0.95,
            'staircase': True,
        }

        """
        optimizer_params defines the optimizer.

        You will need to EDIT the optimizer class. Replace the Adam optimizer
        with a momentum optimizer after switching the learning rate policy to
        piecewise constant.
        """
        params['optimizer_params'] = {
            'func': optimizer.ClipOptimizer,
            'optimizer_class': tf.train.AdamOptimizer,
            'clip': False,
        }

        """
        save_params defines how, where and when your training results are saved
        in the database.

        You will need to EDIT this part. Set your 'host' (set it to 'localhost',
        or to IP if using remote mongodb), 'port' (set it to 24444, unless you 
        have changed mongodb.conf), 'dbname', 'collname', and 'exp_id'. 
        """
        params['save_params'] = {
            'host': '35.199.154.71 ',
            'port': 24444,
            'dbname': 'final',
            'collname': 'alexnet',
            'exp_id': 'in_test2',
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
            'host': '35.199.154.71 ',
            'port': 24444,
            'dbname': 'final',
            'collname': 'alexnet',
            'exp_id': 'in_test2',
            'do_restore': False,
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
        def k_wrapper(inputs, outputs, k):
            return tf.nn.in_top_k(outputs['logits'], inputs['labels'], k)
                                   
        return {'top1': k_wrapper(inputs, outputs, 1),
                'top5': k_wrapper(inputs, outputs, 5)}


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
    m = ImageNetYOLO()
    params = m.setup_params()
    base.train_from_params(**params)
