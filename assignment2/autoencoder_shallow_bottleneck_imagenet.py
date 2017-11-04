"""
Welcome to the second part of the assignment 1! In this section, we will learn
how to analyze our trained model and evaluate its performance on predicting
neural data.
Mainly, you will first learn how to load your trained model from the database
and then how to use tfutils to evaluate your model on neural data using dldata.
The evaluation will be performed using the 'agg_func' in 'validation_params',
which operates on the aggregated validation results obtained from running the
model on the stimulus images. So let's get started!

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


from __future__ import division
import os
import numpy as np
import tensorflow as tf
import tabular as tb
import itertools

from scipy.stats import pearsonr, spearmanr
from dldata.metrics.utils import compute_metric_base
from tfutils import base, data, model, optimizer, utils

from dataprovider import ImageNetDataProvider
from autoencoder_shallow_bottleneck_model import ae_model, ae_model_sparse

class ImageNetClassificationExperiment():
    """
    Defines the neural data testing experiment
    """
    class Config():
        """
        Holds model hyperparams and data information.
        The config class is used to store various hyperparameters and dataset
        information parameters. You will need to change the target layers,
        exp_id, and might have to modify 'conv1_kernel' to the name of your
        first layer, once you start working with different models. Set the seed 
        number to your group number. But please do not change the rest. 

        You will have to EDIT this part. Please set your exp_id here.
        """
        target_layers=[ 'relu']
        extraction_step = None
        exp_id = '1st_experiment'
        data_path = '/datasets/TFRecord_Imagenet_standard'
        batch_size = 50
        seed = 5
        crop_size = 24
        gfs_targets = [] 
        extraction_targets = target_layers + ['labels']
        assert ImageNetDataProvider.N_VAL % batch_size == 0, \
                ('number of examples not divisible by batch size!')
        val_steps = int(ImageNetDataProvider.N_VAL / batch_size)

    def __init__(self):

        self.feature_masks = {}

    def setup_params(self):
        """
        This function illustrates how to setup up the parameters for train_from_params
        """
        params = {}

        """
        validation_params similar to train_params defines the validation parameters.
        It has the same arguments as train_params and additionally
            agg_func: function that aggregates the validation results across batches,
                e.g. to calculate the mean of across batch losses
            online_agg_func: function that aggregates the validation results across
                batches in an online manner, e.g. to calculate the RUNNING mean across
                batch losses

        Note: Note how we switched the data provider from the ImageNetDataProvider 
        to the NeuralDataProvider since we are now working with the neural data.

        """
        params['validation_params'] = {
            'imagenet': {
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
                    'func': self.return_outputs,
                    'targets': self.Config.extraction_targets,
                },
                'num_steps': self.Config.val_steps,
                'agg_func': self.imagenet_classification,
                'online_agg_func': self.online_agg,
            }
        }

        """
        model_params defines the model i.e. the architecture that 
        takes the output of the data provider as input and outputs 
        the prediction of the model.

        You will need to EDIT this part. Switch out the model 'func' as 
        needed when running experiments on different models. The default
        is set to the alexnet model you implemented in the first part of the
        assignment.
        """
        params['model_params'] = {
            'func': ae_model_sparse,
        }

        """
        save_params defines how, where and when your training results are saved
        in the database.

        You will need to EDIT this part. Set your own 'host' ('localhost' if local,
        mongodb IP if remote mongodb), 'port', 'dbname', and 'collname' if you want
        to evaluate on a different model than the pretrained alexnet model.
        'exp_id' has to be set in Config.
        """
        params['save_params'] = {
            'host': '35.199.154.71',
            'port': 24444,
            'dbname': 'assignment2',
            'collname': 'ae_sb_sparse',
            'exp_id': self.Config.exp_id, 
            'save_to_gfs': self.Config.gfs_targets,
        }

        """
        load_params defines how and if a model should be restored from the database.

        You will need to EDIT this part. Set your own 'host' ('localhost' if local,
        mongodb IP if remote mongodb), 'port', 'dbname', and 'collname' if you want
        to evaluate on a different model than the pretrained alexnet model.
        'exp_id' has to be set in Config.
        """
        params['load_params'] = {
            'host': '35.199.154.71',
            'port': 24444,
            'dbname': 'assignment2',
            'collname': 'ae_sb_sparse',
            'exp_id': self.Config.exp_id,
            'do_restore': True,
            'query': {'step': self.Config.extraction_step} \
                    if self.Config.extraction_step is not None else None,
        }

        params['inter_op_parallelism_threads'] = 500

        return params


    def return_outputs(self, inputs, outputs, targets, **kwargs):
        """
        Illustrates how to extract desired targets from the model
        """
        retval = {}
        for target in targets:
            retval[target] = outputs[target]
        return retval


    def online_agg(self, agg_res, res, step):
        """
        Appends the value for each key
        """
        if agg_res is None:
            agg_res = {k: [] for k in res}
    
            # Generate the feature masks
            for k, v in res.items():
                if k in self.Config.target_layers:
                    num_feats = np.product(v.shape[1:])
                    mask = np.random.RandomState(0).permutation(num_feats)[:1024]
                    self.feature_masks[k] = mask

        for k, v in res.items():
            if 'kernel' in k:
                agg_res[k] = v
            elif k in self.Config.target_layers:
                feats = np.reshape(v, [v.shape[0], -1])
                feats = feats[:, self.feature_masks[k]]
                agg_res[k].append(feats)
            else:
                agg_res[k].append(v)
        return agg_res

    def subselect_tfrecords(self, path):
        """
        Illustrates how to subselect files for training or validation
        """
        all_filenames = os.listdir(path)
        rng = np.random.RandomState(seed=SEED)
        rng.shuffle(all_filenames)
        return [os.path.join(path, fn) for fn in all_filenames
                if fn.endswith('.tfrecords')]

    def parse_imagenet_meta_data(self, results):
        """
        Parses the meta data from tfrecords into a tabarray
        """
        meta_keys = ["labels"]
        meta = {}
        for k in meta_keys:
            if k not in results:
                raise KeyError('Attribute %s not loaded' % k)
            meta[k] = np.concatenate(results[k], axis=0)
        return tb.tabarray(columns=[list(meta[k]) for k in meta_keys], names = meta_keys)

    def get_imagenet_features(self, results, num_subsampled_features=None):
        features = {}
        for layer in self.Config.target_layers:
            feats = np.concatenate(results[layer], axis=0)
            feats = np.reshape(feats, [feats.shape[0], -1])
            if num_subsampled_features is not None:
                features[layer] = \
                        feats[:, np.random.RandomState(0).permutation(
                            feats.shape[1])[:num_subsampled_features]]

        return features

    def imagenet_classification(self, results):
        """
        Performs classification on ImageNet using a linear regression on
        feature data from each layer
        """
        retval = {}
        meta = self.parse_imagenet_meta_data(results)
        features = self.get_imagenet_features(results, num_subsampled_features=1024)

        # Subsample to 100 labels
        target_labels = np.unique(meta['labels'])[::10]
        mask = np.isin(meta['labels'], target_labels)
        for layer in features:
            features[layer] = features[layer][mask]
        meta = tb.tabarray(columns=[list(meta['labels'][mask])], names=['labels'])

        #print "Features:", features['bn1'].shape
        print "Labels:", np.unique(meta['labels']).shape

        for layer in features:
            layer_features = features[layer]

            print('%s Imagenet classification test...' % layer)

            category_eval_spec = {
                'npc_train': None,
                'npc_test': 5,
                'num_splits': 3,
                'npc_validate': 0,
                'metric_screen': 'classifier',
                'metric_labels': None,
                'metric_kwargs': {'model_type': 'svm.LinearSVC',
                                  'model_kwargs': {'C':5e-3}},
                'labelfunc': 'labels',
                'train_q': None,
                'test_q': None,
                'split_by': 'labels',
            }
            res = compute_metric_base(layer_features, meta, category_eval_spec)
            res.pop('split_results')
            retval['imagenet_%s' % layer] = res
        return retval

if __name__ == '__main__':
    """
    Illustrates how to run the configured model using tfutils
    """
    base.get_params()
    m = ImageNetClassificationExperiment()
    params = m.setup_params()
    base.test_from_params(**params)
    """
    exp='exp_reg'
    batch=50
    crop=224
    for iteration in [10000, 20000, 40000]: 
        print("Running imagenet model at step %s" % iteration)
        base.get_params()
        m = ImageNetClassificationExperiment('exp_reg', iteration, 32, 224)
        params = m.setup_params()
        base.test_from_params(**params)

    """
