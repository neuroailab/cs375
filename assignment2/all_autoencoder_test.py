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

from scipy.stats import spearmanr
from dldata.metrics.utils import compute_metric_base
from tfutils import base, data, model, optimizer, utils

from utils import post_process_neural_regression_msplit_preprocessed
from dataprovider import NeuralDataProvider, ImageNetDataProvider
from dataprovider import CIFAR10DataProvider
from autoencoder_shallow_bottleneck_model import ae_model, ae_model_sparse
from VAE_models import VAE
from pooledBottleneck_model import pBottleneck_model, pBottleneckSparse_model


class NeuralDataExperiment():
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
        target_layers = []
        extraction_step = None
        exp_id = '1st_experiment'
        data_path = '/datasets/neural_data/tfrecords_with_meta'
        noise_estimates_path = '/datasets/neural_data/noise_estimates.npy'
        batch_size = 128
        seed = 5
        crop_size = 24
        gfs_targets = [] 
        extraction_targets = [attr[0] for attr in NeuralDataProvider.ATTRIBUTES]
        assert NeuralDataProvider.N_VAL % batch_size == 0, \
                ('number of examples not divisible by batch size!')
        val_steps = int(NeuralDataProvider.N_VAL / batch_size)
        # for imagenet classification
        imagenet_data_path = '/datasets/TFRecord_Imagenet_standard'
        imagenet_extraction_targets = ['labels']
    
    def __init__(self, target_layers=None, conv_kernel=None):
        self.Config.target_layers = target_layers
        if target_layers is not None:
            self.Config.extraction_targets += target_layers
            self.Config.imagenet_extraction_targets += target_layers
        if conv_kernel is not None:
            self.Config.extraction_targets += [conv_kernel]
            self.Config.conv_kernel = conv_kernel
        # imagenet 
        self.feature_masks = {}

    def setup_params(self, collection, model):
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
            'VAll': {
                'data_params': {
                    # ImageNet data provider arguments
                    'func': NeuralDataProvider,
                    'data_path': self.Config.data_path,
                    'crop_size': self.Config.crop_size,
                    # TFRecords (super class) data provider arguments
                    'file_pattern': '*.tfrecords',
                    'batch_size': self.Config.batch_size,
                    'shuffle': False,
                    'shuffle_seed': self.Config.seed, 
                    'n_threads': 1,
                },
                'queue_params': {
                    'queue_type': 'fifo',
                    'batch_size': self.Config.batch_size,
                    'seed': self.Config.seed,
                    'capacity': self.Config.batch_size * 10,
                    'min_after_dequeue': self.Config.batch_size * 1,
                },
                'targets': {
                    'func': self.return_outputs,
                    'targets': self.Config.extraction_targets,
                },
                'num_steps': self.Config.val_steps,
                'agg_func': self.neural_analysis,
                'online_agg_func': self.online_agg,
            },
            'V6': {
                'data_params': {
                    # ImageNet data provider arguments
                    'func': NeuralDataProvider,
                    'data_path': self.Config.data_path,
                    'crop_size': self.Config.crop_size,
                    # TFRecords (super class) data provider arguments
                    'file_pattern': '*.tfrecords',
                    'batch_size': self.Config.batch_size,
                    'shuffle': False,
                    'shuffle_seed': self.Config.seed, 
                    'n_threads': 1,
                },
                'queue_params': {
                    'queue_type': 'fifo',
                    'batch_size': self.Config.batch_size,
                    'seed': self.Config.seed,
                    'capacity': self.Config.batch_size * 10,
                    'min_after_dequeue': self.Config.batch_size * 1,
                },
                'targets': {
                    'func': self.return_outputs,
                    'targets': self.Config.extraction_targets,
                },
                'num_steps': self.Config.val_steps,
                'agg_func': self.neural_analysisV6,
                'online_agg_func': self.online_agg,
            },
#             'imagenet': {
#                 'data_params': {
#                     # ImageNet data provider arguments
#                     'func': ImageNetDataProvider,
#                     'data_path': self.Config.imagenet_data_path,
#                     'group': 'val',
#                     'crop_size': self.Config.crop_size,
#                     # TFRecords (super class) data provider arguments
#                     'file_pattern': 'validation*.tfrecords',
#                     'batch_size': self.Config.batch_size,
#                     'shuffle': False,
#                     'shuffle_seed': self.Config.seed,
#                     'file_grab_func': self.subselect_tfrecords,
#                     'n_threads': 4,
#                 },
#                 'queue_params': {
#                     'queue_type': 'fifo',
#                     'batch_size': self.Config.batch_size,
#                     'seed': self.Config.seed,
#                     'capacity': self.Config.batch_size * 10,
#                     'min_after_dequeue': self.Config.batch_size * 5,
#                 },
#                 'targets': {
#                     'func': self.return_outputs,
#                     'targets': self.Config.imagenet_extraction_targets,
#                 },
#                 'num_steps': self.Config.val_steps,
#                 'agg_func': self.imagenet_classification,
#                 'online_agg_func': self.imagenet_online_agg,
#             }
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
            'func': model,
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
            'host': 'localhost',
            'port': 24444,
            'dbname': 'assignment2',
            'collname': collection,
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
            'host': 'localhost',
            'port': 24444,
            'dbname': 'assignment2',
            'collname': collection,
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
        print('Return Outputs Targets', targets)
        retval = {}
        for target in targets:
            retval[target] = outputs[target]
        return retval


    def parse_meta_data(self, results):
        """
        Parses the meta data from tfrecords into a tabarray
        """
        meta_keys = [attr[0] for attr in NeuralDataProvider.ATTRIBUTES \
                if attr[0] not in ['images', 'it_feats']]
        meta = {}
        for k in meta_keys:
            if k not in results:
                raise KeyError('Attribute %s not loaded' % k)
            meta[k] = np.concatenate(results[k], axis=0)
        return tb.tabarray(columns=[list(meta[k]) for k in meta_keys], names = meta_keys)


    def categorization_test(self, features, meta, variability=None):
        """
        Performs a categorization test using dldata

        You will need to EDIT this part. Define the specification to
        do a categorization on the neural stimuli using 
        compute_metric_base from dldata.
        """
        print('Categorization test...')
        if variability is None:
            selection={},
        else:
            selection = {'var': variability}
        category_eval_spec = {
            'npc_train': None,
            'npc_test': 2,
            'num_splits': 20,
            'npc_validate': 0,
            'metric_screen': 'classifier',
            'metric_labels': None,
            'metric_kwargs': {'model_type': 'svm.LinearSVC',
                              'model_kwargs': {'C':5e-3}
                             },
            'labelfunc': 'category',
            'train_q': selection,
            'test_q': selection,
            'split_by': 'obj'
        }
        res = compute_metric_base(features, meta, category_eval_spec)
        res.pop('split_results')
        return res

    def within_categorization_test(self, features, meta, variability=None):
        """
        Performs a categorization test using dldata

        You will need to EDIT this part. Define the specification to
        do a categorization on the neural stimuli using 
        compute_metric_base from dldata.
        """
        print('Within Categorization test...')
        if variability is None:
            selection={},
        else:
            selection = {'var': variability}
        results = {}
        for category in sorted(np.unique(meta['category'])):
            selection['category'] = category
            category_eval_spec = {
                'npc_train': None,
                'npc_test': 2,
                'num_splits': 20,
                'npc_validate': 0,
                'metric_screen': 'classifier',
                'metric_labels': None,
                'metric_kwargs': {'model_type': 'svm.LinearSVC',
                                  'model_kwargs': {'C':5e-3}
                                 },
                'labelfunc': 'obj',
                'train_q': selection,
                'test_q': selection,
                'split_by': 'obj'
            }
            res = compute_metric_base(features, meta, category_eval_spec)
            res.pop('split_results')
            results[category] = res
        return results
    
    def regression_test(self, features, IT_features, meta, variability=None):
        """
        Illustrates how to perform a regression test using dldata

        You will need to EDIT this part. Define the specification to
        do a regression on the IT neurons using compute_metric_base from dldata.
        """
        print('Regression test...')
        if variability is None:
            selection={},
        else:
            selection = {'var': variability}
        it_reg_eval_spec = {
            'npc_train': None,
            'npc_test': 2,
            'num_splits': 20,
            'npc_validate': 0,
            'metric_screen': 'regression',
            'metric_labels': None,
            'metric_kwargs': {'model_type': 'pls.PLSRegression',
                              'model_kwargs': {'n_components':25,'scale':False}
                             },
            'labelfunc': lambda x: (IT_features, None),
            'train_q': selection,
            'test_q': selection,
            'split_by': 'obj'
        }
        res = compute_metric_base(features, meta, it_reg_eval_spec)
        espec = (('all','','IT_regression'), it_reg_eval_spec)
        post_process_neural_regression_msplit_preprocessed(
                res, self.Config.noise_estimates_path)
        res.pop('split_results')
        return res

    def meta_regression_test(self, features, meta, variability=None):
        """
        Illustrates how to perform a regression test using dldata

        You will need to EDIT this part. Define the specification to
        do a regression on the IT neurons using compute_metric_base from dldata.
        """
        print('Meta Regression test...')
        if variability is None:
            selection={},
        else:
            selection = {'var': variability}
        reg_eval_spec = {
            'npc_train': None,
            'npc_test': 2,
            'num_splits': 20,
            'npc_validate': 0,
            'metric_screen': 'regression',
            'metric_labels': None,
            'metric_kwargs': {'model_type': 'linear_model.LassoCV',
                              'model_kwargs': {}
                             },
            'labelfunc': 'rxz',
            'train_q': selection,
            'test_q': selection,
            'split_by': 'obj'
        }
        res = compute_metric_base(features, meta, reg_eval_spec)
        res.pop('split_results')
        return res
    
    def compute_rdm(self, features, meta, mean_objects=False):
        """
        Computes the RDM of the input features

        You will need to EDIT this part. Compute the RDM of features which is a
        [N_IMAGES x N_FEATURES] matrix. The features are then averaged across
        images of the same category which creates a [N_CATEGORIES x N_FEATURES]
        matrix that you have to work with.
        """
        print('Computing RDM...')
        if mean_objects:
            object_list = list(itertools.chain(
                *[np.unique(meta[meta['category'] == c]['obj']) \
                        for c in np.unique(meta['category'])]))
            features = np.array([features[(meta['obj'] == o.rstrip('_'))].mean(0) \
                    for o in object_list])
        ### YOUR CODE HERE
        rdm = 1 - np.corrcoef(features)
        ### END OF YOUR CODE
        return rdm
    
    def compute_rdmV6(self, features, meta, mean_objects=False):
        """
        Computes the RDM of the input features

        You will need to EDIT this part. Compute the RDM of features which is a
        [N_IMAGES x N_FEATURES] matrix. The features are then averaged across
        images of the same category which creates a [N_CATEGORIES x N_FEATURES]
        matrix that you have to work with.
        """
        print('Computing RDM V6...')
        if mean_objects:
            object_list = list(itertools.chain(
                *[np.unique(meta[(meta['var'] == 'V6') & (meta['category'] == c)]['obj']) \
                        for c in np.unique(meta['category'])]))
            features = np.array([features[(meta['obj'] == o.rstrip('_'))].mean(0) \
                    for o in object_list])
        ### YOUR CODE HERE
        rdm = 1 - np.corrcoef(features)
        ### END OF YOUR CODE
        return rdm
    
    def online_agg(self, agg_res, res, step):
        """
        Appends the value for each key
        """
        if agg_res is None:
            agg_res = {k: [] for k in res}
        for k, v in res.items():
            if 'kernel' in k:
                agg_res[k] = v
            else:
                agg_res[k].append(v)
        return agg_res


    # Imagnet Classification
    def subselect_tfrecords(self, path):
        """
        Illustrates how to subselect files for training or validation
        """
        all_filenames = os.listdir(path)
        rng = np.random.RandomState(seed=SEED)
        rng.shuffle(all_filenames)
        return [os.path.join(path, fn) for fn in all_filenames
                if fn.endswith('.tfrecords')]
    
    def imagenet_online_agg(self, agg_res, res, step):
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
    
    # IT Neural Analysis
    
    def get_features(self, results, num_subsampled_features=None):
        """
        Extracts, preprocesses and subsamples the target features
        and the IT features
        """
        features = {}
        for layer in self.Config.target_layers:
            feats = np.concatenate(results[layer], axis=0)
            feats = np.reshape(feats, [feats.shape[0], -1])
            if num_subsampled_features is not None:
                features[layer] = \
                        feats[:, np.random.RandomState(0).permutation(
                            feats.shape[1])[:num_subsampled_features]]

        IT_feats = np.concatenate(results['it_feats'], axis=0)

        return features, IT_feats
    
    def neural_analysis(self, results):
        """
        Performs an analysis of the results from the model on the neural data.
        This analysis includes:
            - saving the conv1 kernels
            - computing a RDM
            - a categorization test
            - and an IT regression.

        You will need to EDIT this function to fully complete the assignment.
        Add the necessary analyses as specified in the assignment pdf.
        """
        print(results.keys())
        retval = {'conv_kernel': results[self.Config.conv_kernel]}
        print('Performing neural analysis...')
        meta = self.parse_meta_data(results)
        features, IT_feats = self.get_features(results, num_subsampled_features=1024)

        print('IT:')
        retval['rdm_it'] = \
                self.compute_rdm(IT_feats, meta, mean_objects=True)

        for layer in features:
            print('Layer: %s' % layer)
            # RDM
            retval['rdm_%s' % layer] = \
                    self.compute_rdm(features[layer], meta, mean_objects=True)
            # RDM correlation
            retval['spearman_corrcoef_%s' % layer] = \
                    spearmanr(
                            np.reshape(retval['rdm_%s' % layer], [-1]),
                            np.reshape(retval['rdm_it'], [-1])
                            )[0]
            # categorization test
            retval['categorization_%s' % layer] = \
                    self.categorization_test(features[layer], meta, ['V0','V3','V6'])
            # within-categorization test
            retval['within_categorization_%s' % layer] = \
                    self.within_categorization_test(features[layer], meta, ['V0','V3','V6'])
            # IT regression test
            retval['it_regression_%s' % layer] = \
                    self.regression_test(features[layer], IT_feats, meta, ['V0','V3','V6'])
            # meta regression test
#             retval['meta_regression_%s' % layer] = \
#                     self.meta_regression_test(features[layer], meta, ['V0','V3','V6'])
        return retval
    
    def neural_analysisV6(self, results):
        """
        Performs an analysis of the results from the model on the neural data.
        This analysis includes:
            - saving the conv1 kernels
            - computing a RDM
            - a categorization test
            - and an IT regression.

        You will need to EDIT this function to fully complete the assignment.
        Add the necessary analyses as specified in the assignment pdf.
        """
        print('Results', results.keys())
        retval = {'conv_kernel': results[self.Config.conv_kernel]}
        print('Performing neural analysis...')
        meta = self.parse_meta_data(results)
        features, IT_feats = self.get_features(results, num_subsampled_features=1024)
        print('Features', features.keys())
        print('IT:')
        retval['rdm_it'] = \
                self.compute_rdmV6(IT_feats, meta, mean_objects=True)

        for layer in features:
            print('Layer: %s' % layer)
            # RDM
            retval['rdm_%s' % layer] = \
                    self.compute_rdmV6(features[layer], meta, mean_objects=True)
            # RDM correlation
            retval['spearman_corrcoef_%s' % layer] = \
                    spearmanr(
                            np.reshape(retval['rdm_%s' % layer], [-1]),
                            np.reshape(retval['rdm_it'], [-1])
                            )[0]
            # categorization test
            retval['categorization_%s' % layer] = \
                    self.categorization_test(features[layer], meta, ['V6'])
            # within-categorization test
            retval['within_categorization_%s' % layer] = \
                    self.within_categorization_test(features[layer], meta, ['V6'])
            # IT regression test
            retval['it_regression_%s' % layer] = \
                    self.regression_test(features[layer], IT_feats, meta, ['V6'])
            # meta regression test
#             retval['meta_regression_%s' % layer] = \
#                     self.meta_regression_test(features[layer], meta, ['V6'])
                
        return retval

if __name__ == '__main__':
    """
    Illustrates how to run the configured model using tfutils
    """
    models = [{'collection': 'ae_shallow_bottleneck',
              'model': ae_model,
              'target_layers': ['relu'],
              'conv_kernel': 'conv_kernel'} ,
               {'collection': 'VAE',
               'model': VAE,
               'target_layers': ['conv1', 'z'],
               'conv_kernel': 'conv1_weights'},
              {'collection': 'ae_sb_sparse',
              'model': ae_model_sparse,
              'target_layers': ['relu'],
              'conv_kernel': 'conv_kernel'},
              {'collection': 'pooled_bottleneck',
              'model': pBottleneck_model,
              'target_layers': ['deconv2'],
              'conv_kernel': 'conv1_kernel'} ,
              {'collection': 'pooled_bottleneckSparse',
              'model': pBottleneckSparse_model,
              'target_layers': ['deconv2'],
              'conv_kernel': 'conv1_kernel'}
              ]
    models=[models[2]]
    
    for input_dict in models:
        print('Collection:', input_dict['collection'])
        base.get_params()
        m = NeuralDataExperiment(input_dict['target_layers'], input_dict['conv_kernel'])
        params = m.setup_params(input_dict['collection'], input_dict['model'])
        base.test_from_params(**params)
