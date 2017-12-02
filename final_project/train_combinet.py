from __future__ import division, print_function, absolute_import
import os, sys
import numpy as np

import tensorflow as tf
import cPickle

from tfutils import base, data, optimizer

import json
import copy
import argparse

from utils import *
from data_provider import *

sys.path.append('../normal_pred/')
import normal_encoder_asymmetric_with_bypass
import combinet_builder

host = os.uname()[1]

BATCH_SIZE = 32
IMAGE_SIZE_CROP = 224
NUM_CHANNELS = 3

def get_parser():

    parser = argparse.ArgumentParser(description='The script to train the combine net')

    # General settings
    parser.add_argument('--nport', default = 27017, type = int, action = 'store', help = 'Port number of mongodb')
    parser.add_argument('--expId', default = "combinet_test", type = str, action = 'store', help = 'Name of experiment id')
    parser.add_argument('--loadexpId', default = None, type = str, action = 'store', help = 'Name of experiment id')
    parser.add_argument('--cacheDirPrefix', default = "/mnt/fs1/chengxuz", type = str, action = 'store', help = 'Prefix of cache directory')
    parser.add_argument('--innerargs', default = [], type = str, action = 'append', help = 'Arguments for every network')
    parser.add_argument('--with_rep', default = 0, type = int, action = 'store', help = 'Whether reporting other losses every batch')
    parser.add_argument('--with_grad', default = 0, type = int, action = 'store', help = 'Whether with gradients reporting')
    parser.add_argument('--with_train', default = 0, type = int, action = 'store', help = 'Whether with training dataset')
    parser.add_argument('--with_recdata', default = 0, type = int, action = 'store', help = 'Whether with second database setting')
    parser.add_argument('--nport_rec', default = 27007, type = int, action = 'store', help = 'Port for second database')
    parser.add_argument('--valid_first', default = 0, type = int, action = 'store', help = 'Whether validating first')
    parser.add_argument('--loadport', default = None, type = int, action = 'store', help = 'Port number of mongodb for loading')
    parser.add_argument('--with_feat', default = 1, type = int, action = 'store', help = 'Whether adding the feat validation')
    parser.add_argument('--loadstep', default = None, type = int, action = 'store', help = 'Number of steps for loading')

    # Network related
    parser.add_argument('--pathconfig', default = "normals_config_fcnvgg16_withdepth.cfg", type = str, action = 'store', help = 'Path to config file')
    parser.add_argument('--dataconfig', default = "dataset_config.cfg", type = str, action = 'store', help = 'Path to config file for dataset')
    parser.add_argument('--valdconfig', default = None, type = str, action = 'store', help = 'Validation dataset config, default to be None, and will copy to other configs below')
    parser.add_argument('--topndconfig', default = None, type = str, action = 'store', help = 'Path to config file for dataset, for topn validation')
    parser.add_argument('--featdconfig', default = None, type = str, action = 'store', help = 'Path to config file for dataset, for feats validation')
    parser.add_argument('--modeldconfig', default = None, type = str, action = 'store', help = 'Path to config file for dataset, for model in validation')
    parser.add_argument('--seed', default = 0, type = int, action = 'store', help = 'Random seed for model')
    parser.add_argument('--namefunc', default = "combine_normal_tfutils_new", type = str, action = 'store', help = 'Name of function to build the network')
    parser.add_argument('--valinum', default = -1, type = int, action = 'store', help = 'Number of validation steps, default is -1, which means all the validation')
    parser.add_argument('--cache_filter', default = 0, type = int, action = 'store', help = 'Whether cache the pretrained weights as tf tensors')
    parser.add_argument('--fix_pretrain', default = 0, type = int, action = 'store', help = 'Whether fix the pretrained weights')
    parser.add_argument('--extra_feat', default = 0, type = int, action = 'store', help = 'Whether to add normal and depth outputs for ImageNet and PlaceNet, default is 0, which means no')

    # Loss related
    parser.add_argument('--whichloss', default = 0, type = int, action = 'store', help = 'Whether to use new loss') # Deprecated for now
    parser.add_argument('--depth_norm', default = 8000, type = int, action = 'store', help = 'Coefficient for depth loss')
    parser.add_argument('--label_norm', default = 20, type = float, action = 'store', help = 'Coefficient for label loss')
    parser.add_argument('--depthloss', default = 0, type = int, action = 'store', help = 'Whether to use new depth loss')
    parser.add_argument('--normalloss', default = 0, type = int, action = 'store', help = 'Whether to use new normal loss')
    parser.add_argument('--multtime', default = 1, type = int, action = 'store', help = '1 means original, larger than 1 means multiple time points')
    parser.add_argument('--trainable', default = 0, type = int, action = 'store', help = 'Whether use trainable weights')
    parser.add_argument('--nfromd', default = 0, type = int, action = 'store', help = 'Whether calculating the normals from depth')
    parser.add_argument('--ret_dict', default = 0, type = int, action = 'store', help = '1 means returning dict for loss_withcfg, default is 0')
    parser.add_argument('--combine_dict', default = 0, type = int, action = 'store', help = '1 means combining 5 datasets to 2, default is 0')
    parser.add_argument('--self_order', default = None, type = str, action = 'store', help = 'None means default, otherwise, it should be separated by ","')
    parser.add_argument('--print_loss', default = 0, type = int, action = 'store', help = '1 means printing loss us tf.Print, default is 0')

    # Training related
    parser.add_argument('--batchsize', default = None, type = int, action = 'store', help = 'Batch size')
    parser.add_argument('--valbatchsize', default = None, type = int, action = 'store', help = 'Validation Batch size')
    parser.add_argument('--queuecap', default = None, type = int, action = 'store', help = 'Queue capacity')
    parser.add_argument('--init_stddev', default = .01, type = float, action = 'store', help = 'Init stddev for convs')
    parser.add_argument('--init_type', default = 'xavier', type = str, action = 'store', help = 'Init type')
    parser.add_argument('--n_threads', default = 4, type = int, action = 'store', help = 'Number of threads')
    parser.add_argument('--val_n_threads', default = 1, type = int, action = 'store', help = 'Number of threads for validation')

    ## Learning rate, optimizers
    parser.add_argument('--init_lr', default = .01, type = float, action = 'store', help = 'Init learning rate')
    parser.add_argument('--whichopt', default = 0, type = int, action = 'store', help = 'Choice of the optimizer, 0 means momentum, 1 means Adam')
    parser.add_argument('--adameps', default = 0.1, type = float, action = 'store', help = 'Epsilon for adam, only used when whichopt is 1')
    parser.add_argument('--adambeta1', default = 0.9, type = float, action = 'store', help = 'Beta1 for adam, only used when whichopt is 1')
    parser.add_argument('--adambeta2', default = 0.999, type = float, action = 'store', help = 'Beta2 for adam, only used when whichopt is 1')
    parser.add_argument('--withclip', default = 1, type = int, action = 'store', help = 'Whether do clip')

    ## Saving metric
    parser.add_argument('--fre_valid', default = 10000, type = int, action = 'store', help = 'Frequency of the validation')
    parser.add_argument('--fre_metric', default = 1000, type = int, action = 'store', help = 'Frequency of the saving metrics')
    parser.add_argument('--fre_filter', default = 10000, type = int, action = 'store', help = 'Frequency of the saving filters')
    
    ## Dataset related
    parser.add_argument('--whichscenenet', default = 0, type = int, action = 'store', help = 'Choice of the scenenet, 0 means all, 1 means the compressed version, 2 means the new png version')
    parser.add_argument('--whichscannet', default = 0, type = int, action = 'store', help = 'Choice of the scannet, 0 means original, 1 means the smaller new version')
    parser.add_argument('--whichimagenet', default = 0, type = int, action = 'store', help = 'Choice of the imagenet, 0 means original, 1 means the smaller new version')
    parser.add_argument('--whichcoco', default = 0, type = int, action = 'store', help = 'Which coco dataset to use, 0 means original, 1 means the one without 0 instance')
    parser.add_argument('--as_list', default = 0, type = int, action = 'store', help = 'Whether handling init_ops as dicts or not, if as dicts, enqueue and dequeue will be done separately to each dict')
    parser.add_argument('--which_place', default = 0, type = int, action = 'store', help = 'Which place dataset to use, 1 means only part')

    ## Preprocessing related
    parser.add_argument('--twonormals', default = 0, type = int, action = 'store', help = 'Whether having two normal readouts, 0 means no')
    parser.add_argument('--depthnormal', default = 0, type = int, action = 'store', help = 'Whether to normalize the depth input')
    parser.add_argument('--ignorebname', default = 0, type = int, action = 'store', help = 'Whether ignore the batch name')
    parser.add_argument('--categorymulti', default = 1, type = int, action = 'store', help = 'Whether use multiple batches for category inputs (imagenet and places)')
    parser.add_argument('--withflip', default = 0, type = int, action = 'store', help = 'Whether flip the input images horizontally, this will not flip the normals correctly, see flipnormal choice below')
    parser.add_argument('--onlyflip_cate', default = 0, type = int, action = 'store', help = 'Whether flip the input images only in categorization')
    parser.add_argument('--flipnormal', default = 0, type = int, action = 'store', help = 'Set this to 1 while withflip is 1, and it will flip the normals correctly')
    parser.add_argument('--with_noise', default = 0, type = int, action = 'store', help = 'Whether adding gaussian noise')
    parser.add_argument('--noise_level', default = 10, type = int, action = 'store', help = 'Level of gaussian noise added')
    parser.add_argument('--shuffle_seed', default = 0, type = int, action = 'store', help = 'Shuffle seed for the data provider')
    parser.add_argument('--weight_decay', default = None, type = float, action = 'store', help = 'Init learning rate')
    parser.add_argument('--no_shuffle', default = 0, type = int, action = 'store', help = 'Whether do the shuffling')
    parser.add_argument('--crop_each', default = 0, type = int, action = 'store', help = 'Whether do the crop to each image')
    parser.add_argument('--eliprep', default = 0, type = int, action = 'store', help = 'Whether to use Eli preprocessing for imagenet')
    parser.add_argument('--thprep', default = 0, type = int, action = 'store', help = 'Whether to use torch resnet preprocessing for imagenet')
    parser.add_argument('--no_prep', default = 0, type = int, action = 'store', help = 'Avoid the scaling in model function or not')
    parser.add_argument('--sub_mean', default = 0, type = int, action = 'store', help = 'Whether subtracting the mean')
    parser.add_argument('--mean_path', default = '/mnt/fs0/datasets/TFRecord_Imagenet_standard/ilsvrc_2012_mean.npy', type = str, action = 'store', help = 'Provided by Damian!')
    parser.add_argument('--with_color_noise', default = 0, type = int, action = 'store', help = 'Whether do color jittering')

    ## Related to kinetics
    parser.add_argument('--crop_time', default = 5, type = int, action = 'store', help = 'Crop time for kinetics dataset')
    parser.add_argument('--crop_rate', default = 5, type = int, action = 'store', help = 'Crop rate for kinetics dataset')
    parser.add_argument('--replace_folder_train', default = None, type = str, action = 'store', help = 'Replace_folder for train group')
    parser.add_argument('--replace_folder_val', default = None, type = str, action = 'store', help = 'Replace folder for val group')

    # GPU related
    parser.add_argument('--gpu', default = '0', type = str, action = 'store', help = 'Availabel GPUs')
    parser.add_argument('--n_gpus', default = None, type = int, action = 'store', help = 'Number of GPUs to use, default is None, to use length in gpu')
    parser.add_argument('--gpu_offset', default = 0, type = int, action = 'store', help = 'Offset of gpu index')
    parser.add_argument('--parallel', default = 0, type = int, action = 'store', help = 'Whether use paralleling method')
    parser.add_argument('--use_new_tfutils', default = 0, type = int, action = 'store', help = 'Whether use new interface')
    parser.add_argument('--minibatch', default = None, type = int, action = 'store', help = 'Minibatch to use, default to be None, not using')
    return parser

def get_params_from_arg(args):

    if args.topndconfig is None:
        args.topndconfig = args.valdconfig
    if args.featdconfig is None:
        args.featdconfig = args.valdconfig
    if args.modeldconfig is None:
        args.modeldconfig = args.valdconfig

    if args.n_gpus==None:
        args.n_gpus = len(args.gpu.split(','))

    pathconfig = args.pathconfig
    if not os.path.exists(pathconfig):
        pathconfig = os.path.join('network_configs', pathconfig)
    cfg_initial = postprocess_config(json.load(open(pathconfig)))

    dataconfig = args.dataconfig
    if not os.path.exists(dataconfig):
        dataconfig = os.path.join('dataset_configs', dataconfig)
    cfg_dataset = postprocess_config(json.load(open(dataconfig)))

    exp_id  = args.expId
    #cache_dir = os.path.join(args.cacheDirPrefix, '.tfutils', 'localhost:'+ str(args.nport), 'normalnet-test', 'normalnet', exp_id)
    cache_dir = os.path.join(args.cacheDirPrefix, '.tfutils', 'localhost:27017', 'normalnet-test', 'normalnet', exp_id)

    BATCH_SIZE  = normal_encoder_asymmetric_with_bypass.getBatchSize(cfg_initial)
    queue_capa  = normal_encoder_asymmetric_with_bypass.getQueueCap(cfg_initial)
    if args.batchsize!=None:
        BATCH_SIZE = args.batchsize

    if args.queuecap!=None:
        queue_capa = args.queuecap
    n_threads   = args.n_threads

    func_net = getattr(combinet_builder, args.namefunc)

    if args.depthnormal==1:
        assert args.depth_norm==1, "Depth norm needs to be 1!"

    data_param_base = {
                    'data_path': DATA_PATH,
                    'batch_size': 1,
                    'cfg_dataset': cfg_dataset,
                    'whichscenenet': args.whichscenenet,
                    'nfromd': args.nfromd,
                    'depthnormal': args.depthnormal,
                    'whichscannet': args.whichscannet,
                    'withflip': args.withflip,
                    'with_noise': args.with_noise,
                    'noise_level': args.noise_level,
                    'whichimagenet': args.whichimagenet,
                    'whichcoco': args.whichcoco,
                    'onlyflip_cate': args.onlyflip_cate,
                    'flipnormal': args.flipnormal,
                    'eliprep': args.eliprep,
                    'thprep': args.thprep,
                    'crop_time': args.crop_time,
                    'crop_rate': args.crop_rate,
                    'sub_mean': args.sub_mean,
                    'mean_path': args.mean_path,
                    'with_color_noise': args.with_color_noise,
                    'which_place': args.which_place,
            }

    train_data_param_base = {
                    'group': 'train',
                    'n_threads': n_threads,
                    'shuffle_seed': args.shuffle_seed,
                    'no_shuffle': args.no_shuffle,
                    'replace_folder': args.replace_folder_train,
                    'categorymulti': args.categorymulti,
                    'crop_each': args.crop_each,
                    'as_list': args.as_list,
            }
    train_data_param_base.update(data_param_base)


    train_queue_params = {
            'queue_type': 'random',
            'batch_size': BATCH_SIZE,
            'seed': 0,
            'capacity': queue_capa,
            'min_after_dequeue': BATCH_SIZE,
            # 'n_threads' : 4
        }
    if args.categorymulti==1 and args.as_list==0:
        train_data_param = {
                    'func': Combine_world,
                }
        train_data_param.update(train_data_param_base)

    else:
        train_data_param = {
                    'queue_params': train_queue_params,
                }
        train_data_param.update(train_data_param_base)
        train_world = Combine_world(**train_data_param)

        train_data_param = train_world.data_params_list
        train_queue_params = train_world.queue_params_list

    val_data_param = {
                'func': Combine_world,
                'group': 'val',
                'n_threads': args.val_n_threads,
                'replace_folder': args.replace_folder_val,
            }
    val_data_param.update(data_param_base)

    val_batch_size = BATCH_SIZE
    if not args.minibatch is None:
        val_batch_size = args.minibatch
    if not args.valbatchsize is None:
        val_batch_size = args.valbatchsize
    val_queue_params = {
                'queue_type': 'fifo',
                'batch_size': val_batch_size,
                'seed': 0,
                'capacity': val_batch_size*2,
                'min_after_dequeue': val_batch_size,
            }

    #val_target          = ['normal_scenenet', 'normal_pbrnet']
    #val_target          = ['normal_scenenet', 'normal_pbrnet', 'depth_pbrnet']
    #val_target          = ['normal_pbrnet', 'depth_pbrnet', 'depth_scannet', 'label_imagenet']
    #val_target          = ['normal_pbrnet', 'depth_pbrnet', 'depth_scannet']
    val_target = get_val_target(cfg_dataset, nfromd = args.nfromd)

    val_step_num = 1000 
    NUM_BATCHES_PER_EPOCH = 5000

    if args.valinum>-1:
        val_step_num = args.valinum

    #loss_func = loss_ave_l2
    #loss_func = loss_withdepth
    #loss_func = loss_withdepthlabel
    loss_func = loss_withcfg
    loss_func_kwargs = {
                'cfg_dataset': cfg_dataset,
                'depth_norm': args.depth_norm,
                'label_norm': args.label_norm,
                'depthloss': args.depthloss,
                'normalloss': args.normalloss,
                'nfromd': args.nfromd,
                'trainable': args.trainable,
                'multtime': args.multtime,
                'ret_dict': args.ret_dict,
                'combine_dict': args.combine_dict,
                'print_loss': args.print_loss,
                'extra_feat': args.extra_feat,
            }

    learning_rate_params = {
            'func': tf.train.exponential_decay,
            'learning_rate': args.init_lr,
            #'decay_rate': .95,
            'decay_rate': 1,
            'decay_steps': 6*NUM_BATCHES_PER_EPOCH,  # exponential decay each epoch
            'staircase': True
        }

    optimizer_class = tf.train.MomentumOptimizer

    model_params = {
            'func': func_net,
            'seed': args.seed,
            'cfg_initial': cfg_initial,
            'cfg_dataset': cfg_dataset,
            'init_stddev': args.init_stddev,
            'twonormals': args.twonormals,
            'ignorebname': args.ignorebname,
            'weight_decay': args.weight_decay,
            'init_type': args.init_type,
            'cache_filter': args.cache_filter,
            'fix_pretrain': args.fix_pretrain,
            'extra_feat': args.extra_feat,
        }
    if args.namefunc in ['combine_normal_tfutils_new', 'combine_tfutils_general'] and args.no_prep==1:
        model_params['no_prep'] = 1

    if not args.modeldconfig is None:
        dataconfig = args.modeldconfig
        if not os.path.exists(dataconfig):
            dataconfig = os.path.join('dataset_configs', dataconfig)
        cfg_dataset_model = postprocess_config(json.load(open(dataconfig)))

        model_params['cfg_dataset'] = cfg_dataset_model
    
    if args.parallel==1:
        if args.use_new_tfutils==0:
            model_params['model_func'] = model_params['func']
            model_params['func'] = combinet_builder.parallel_network_tfutils
            model_params['n_gpus'] = args.n_gpus
            model_params['gpu_offset'] = args.gpu_offset
        else:
            model_params['num_gpus'] = args.n_gpus
            model_params['devices'] = ['/gpu:%i' % (i + args.gpu_offset) for i in xrange(args.n_gpus)]

    if args.whichloss==1:
        loss_func = loss_ave_invdot
        learning_rate_params = {
                'func': tf.train.exponential_decay,
                'learning_rate': .001,
                'decay_rate': .5,
                'decay_steps': 5*NUM_BATCHES_PER_EPOCH,  # exponential decay each epoch
                'staircase': True
            }

        model_params['center_im']   = True

    dbname = 'combinet-test'
    collname = 'combinet'

    save_to_gfs = [
                'fea_image_scenenet', 'fea_normal_scenenet', 'fea_depth_scenenet', 'out_normal_scenenet', 'out_depth_scenenet',
                'fea_image_pbrnet', 'fea_normal_pbrnet', 'fea_depth_pbrnet',  'out_normal_pbrnet', 'out_depth_pbrnet',
                'fea_image_scannet', 'fea_depth_scannet', 'out_depth_scannet',
                'fea_instance_pbrnet', 'out_instance_pbrnet',
                'fea_instance_scenenet', 'out_instance_scenenet',
                'fea_instance_coco', 'out_instance_coco', 'fea_image_coco',
                'fea_image_nyuv2', 'fea_depth_nyuv2', 'out_depth_nyuv2',
                'fea_image_imagenet', 'out_normal_imagenet', 'out_depth_imagenet',
                'fea_image_place', 'out_normal_place', 'out_depth_place',
                ]

    if args.with_rep==1:
        save_to_gfs.extend(['loss_normal_scenenet', 'loss_depth_scenenet', 'loss_instance_scenenet',
            'loss_depth_scannet', 
            'loss_normal_pbrnet', 'loss_depth_pbrnet', 'loss_instance_pbrnet',
            'loss_top1_imagenet', 'loss_top5_imagenet', 
            'loss_instance_coco',
            'loss_top1_place', 'loss_top5_place', 'loss_depth_nyuv2',
            ])

    save_params = {
            'host': 'localhost',
            'port': args.nport,
            'dbname': dbname,
            'collname': collname,
            'exp_id': exp_id,

            'do_save': True,
            'save_initial_filters': True,
            'save_metrics_freq': args.fre_metric,  # keeps loss from every SAVE_LOSS_FREQ steps.
            'save_valid_freq': args.fre_valid,
            'save_filters_freq': args.fre_filter,
            'cache_filters_freq': args.fre_filter,
            'cache_dir': cache_dir,  # defaults to '~/.tfutils'
            'save_to_gfs': save_to_gfs,
        }

    if args.with_recdata==1:
        save_params['port_rec'] = args.nport_rec

    loadport = args.nport
    if not args.loadport is None:
        loadport = args.loadport
    loadexpId = exp_id
    if not args.loadexpId is None:
        loadexpId = args.loadexpId
    load_query = None
    if not args.loadstep is None:
        load_query = {'exp_id': loadexpId, 'saved_filters': True, 'step': args.loadstep}
    load_params = {
            'host': 'localhost',
            'port': loadport,
            'dbname': dbname,
            'collname': collname,
            'exp_id': loadexpId,
            'do_restore': True,
            'query': load_query,
        }

    loss_rep_targets_kwargs = {
                    'target': val_target,
                    'cfg_dataset': cfg_dataset,
                    'depth_norm': args.depth_norm,
                    'depthloss': args.depthloss,
                    'normalloss': args.normalloss,
                    'nfromd': args.nfromd,
                    'multtime': args.multtime,
                    'extra_feat': args.extra_feat,
            }
    topn_val_data_param = copy.deepcopy(val_data_param)

    if not args.topndconfig is None:
        dataconfig = args.topndconfig
        if not os.path.exists(dataconfig):
            dataconfig = os.path.join('dataset_configs', dataconfig)
        cfg_dataset_topn = postprocess_config(json.load(open(dataconfig)))
        val_target_topn = get_val_target(cfg_dataset_topn, nfromd = args.nfromd)

        loss_rep_targets_kwargs['cfg_dataset'] = cfg_dataset_topn
        loss_rep_targets_kwargs['target'] = val_target_topn
        topn_val_data_param['cfg_dataset'] = cfg_dataset_topn

    loss_rep_targets = {
                    'func': rep_loss_withcfg,
                }
    loss_rep_targets.update(loss_rep_targets_kwargs)

    train_params = {
            'validate_first': False,
            'data_params': train_data_param,
            'queue_params': train_queue_params,
            'thres_loss': float('Inf'),
            'num_steps': 1900 * NUM_BATCHES_PER_EPOCH,  # number of steps to train
        }

    if args.with_rep==1:
        train_params['targets'] = loss_rep_targets

    if args.valid_first==1:
        train_params['validate_first'] = True

    if args.ret_dict==1:
        curr_order = ['scene', 'pbr', 'imagenet', 'coco', 'place', 'kinetics']
        if args.combine_dict==1:
            curr_order = ['category', 'noncategory']
        if not args.self_order is None:
            curr_order = args.self_order.split(',')
        trainloop_class = Trainloop_class(curr_order)

        train_params['train_loop'] = {'func' : trainloop_class.train_loop}

    loss_params = {
            'targets': val_target,
            'agg_func': tf.reduce_mean,
            'loss_per_case_func': loss_func,
            'loss_per_case_func_params': {},
            'loss_func_kwargs': loss_func_kwargs,
        }

    if args.ret_dict==1:
        loss_params['agg_func'] = lambda x: {k:tf.reduce_mean(v) for k,v in x.items()}

    if args.parallel==1:
        if args.use_new_tfutils==0:
            loss_params['loss_per_case_func'] = parallel_loss_withcfg
            loss_params['loss_func_kwargs']['n_gpus'] = args.n_gpus
            loss_params['loss_func_kwargs']['gpu_offset'] = args.gpu_offset
            loss_params['agg_func'] = parallel_reduce_mean

    clip_flag = args.withclip==1

    optimizer_params = {
            'func': optimizer.ClipOptimizer,
            'optimizer_class': optimizer_class,
            'clip': clip_flag,
            'momentum': .9
        }

    if args.whichopt==1:
        optimizer_params = {
            'func': optimizer.ClipOptimizer,
            'optimizer_class': tf.train.AdamOptimizer,
            'clip': clip_flag,
            'epsilon': args.adameps,
            'beta1': args.adambeta1,
            'beta2': args.adambeta2,
        }

    if args.whichopt==2:
        optimizer_params = {
            'func': optimizer.ClipOptimizer,
            'optimizer_class': tf.train.AdagradOptimizer,
            'clip': clip_flag,
        }

    if args.whichopt==3:
        optimizer_params = {
                'func': optimizer.ClipOptimizer,
                'optimizer_class': optimizer_class,
                'clip': clip_flag,
                'momentum': .9,
                'use_nesterov': True
            }

    if args.whichopt==4:
        optimizer_params = {
            'func': optimizer.ClipOptimizer,
            'optimizer_class': tf.train.AdadeltaOptimizer,
            'clip': clip_flag,
        }

    if args.whichopt==5:
        optimizer_params = {
            'func': optimizer.ClipOptimizer,
            'optimizer_class': tf.train.RMSPropOptimizer,
            'clip': clip_flag,
        }

    if args.parallel==1:
        if args.use_new_tfutils==0:
            optimizer_params['func'] = ParallelClipOptimizer
            optimizer_params['gpu_offset'] = args.gpu_offset
            optimizer_params['n_gpus'] = args.n_gpus

    feats_target = {
                    'func': save_features,
                    'num_to_save': 5,
                    'targets' : val_target,
                    'cfg_dataset': cfg_dataset,
                    'depth_norm': args.depth_norm,
                    'normalloss': args.normalloss,
                    'nfromd': args.nfromd,
                    'depthnormal': args.depthnormal,
                    'extra_feat': args.extra_feat,
                }
    feats_val_data_param = copy.deepcopy(val_data_param)

    if not args.featdconfig is None:
        dataconfig = args.featdconfig
        if not os.path.exists(dataconfig):
            dataconfig = os.path.join('dataset_configs', dataconfig)
        cfg_dataset_feat = postprocess_config(json.load(open(dataconfig)))
        val_target_feat = get_val_target(cfg_dataset_feat, nfromd = args.nfromd)

        feats_target['target'] = val_target_feat
        feats_target['cfg_dataset'] = cfg_dataset_feat
        feats_val_data_param['cfg_dataset'] = cfg_dataset_feat

    feats_val_param = {
                'data_params': feats_val_data_param,
                'queue_params': val_queue_params,
                'targets': feats_target,
                'num_steps': 10,
                'agg_func': mean_losses_keep_rest,
            }
    feats_train_param = {
                'data_params': train_data_param,
                'queue_params': val_queue_params,
                'targets': feats_target,
                'num_steps': 10,
                'agg_func': mean_losses_keep_rest,
            }

    topn_val_param = {
                'data_params': topn_val_data_param,
                'queue_params': val_queue_params,
                'targets': loss_rep_targets,
                'num_steps': val_step_num,
                'agg_func': lambda x: {k:np.mean(v) for k,v in x.items()},
                'online_agg_func': online_agg
            }

    topn_train_param = {
                'data_params': train_data_param,
                'queue_params': val_queue_params,
                'targets': loss_rep_targets,
                'num_steps': val_step_num,
                'agg_func': lambda x: {k:np.mean(v) for k,v in x.items()},
                'online_agg_func': online_agg
            }

    extra_loss_func_kwargs = {
            'label_norm': args.label_norm, 
            'top_or_loss': 1,
            }
    extra_loss_func_kwargs.update(loss_rep_targets_kwargs)
    grad_val_param = {
                'data_params': train_data_param,
                'queue_params': val_queue_params,
                'targets': {
                    'func': report_grad,
                    'loss_func': rep_loss_withcfg,
                    'loss_func_kwargs': extra_loss_func_kwargs,
                    'var_filter': encode_var_filter,
                },
                'num_steps': 100,
                'agg_func': mean_losses_keep_rest,
                #'agg_func': online_agg,
            }

    validation_params = {
            'topn': topn_val_param,
        }
    if args.with_feat==1:
        validation_params['feats'] = feats_val_param

    if args.with_grad==1:
        #validation_params['grad'] = grad_val_param
        validation_params = {'grad': grad_val_param}
        train_params['validate_first'] = True
        save_params['save_valid_freq'] = 1000

    if args.with_train==1:
        validation_params['topn_train'] = topn_train_param
        validation_params['feats_train'] = feats_train_param

    if args.parallel==1:
        if args.use_new_tfutils==0:
            validation_params['feats']['targets']['n_gpus'] = args.n_gpus
            validation_params['feats']['targets']['func'] = parallel_save_features

            validation_params['topn']['targets']['n_gpus'] = args.n_gpus
            validation_params['topn']['targets']['func'] = parallel_rep_loss_withcfg

    params = {
        'save_params': save_params,

        'load_params': load_params,

        'model_params': model_params,

        'train_params': train_params,

        'loss_params': loss_params,

        'learning_rate_params': learning_rate_params,

        'optimizer_params': optimizer_params,

        'log_device_placement': False,  # if variable placement has to be logged

        'validation_params': validation_params,
        #'inter_op_parallelism_threads': 1,
    }

    if not args.minibatch is None:
        params['train_params']['minibatch_size'] = args.minibatch

    return params

def main():

    parser = get_parser()

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    params = get_params_from_arg(args)

    #base.train_from_params(**params)

    if len(args.innerargs)==0:
        params = get_params_from_arg(args)

        if not params is None:
            base.train_from_params(**params)
    else:
        params = {
                'save_params': [],

                'load_params': [],

                'model_params': [],

                'train_params': None,

                'loss_params': [],

                'learning_rate_params': [],

                'optimizer_params': [],

                'log_device_placement': False,  # if variable placement has to be logged
                'validation_params': [],
            }

        list_names = [
                "save_params", "load_params", "model_params", 
                "validation_params", "loss_params", "learning_rate_params", 
                "optimizer_params"
                ]
        
        for curr_arg in args.innerargs:

            args = parser.parse_args(curr_arg.split())
            curr_params = get_params_from_arg(args)
            
            for tmp_key in list_names:
                params[tmp_key].append(curr_params[tmp_key])

            params['train_params'] = curr_params['train_params']

        base.train_from_params(**params)

if __name__ == '__main__':
    main()
