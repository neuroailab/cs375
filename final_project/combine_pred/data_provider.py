from __future__ import division, print_function, absolute_import
import os, sys
import numpy as np

import tensorflow as tf
import cPickle

from tfutils import base, data, optimizer

import json
import copy
import argparse

import coco_provider
import kinetics_provider

sys.path.append('../no_tfutils/')
from vgg_preprocessing import preprocess_image
from resnet_th_preprocessing import preprocessing_th

class Combine_world:

    def __init__(self,
                 data_path,
                 group='train',
                 batch_size=1,
                 n_threads=4,
                 crop_size=None,
                 cfg_dataset={},
                 whichscenenet=0,
                 nfromd=0,
                 depthnormal=0,
                 whichscannet=0,
                 categorymulti=1,
                 queue_params=None,
                 withflip=0, 
                 with_noise=0, 
                 noise_level=10,
                 whichimagenet=0,
                 no_shuffle=0,
                 crop_each=0,
                 whichcoco=0,
                 onlyflip_cate=0,
                 flipnormal=0,
                 eliprep=0,
                 thprep=0,
                 crop_time=5,
                 crop_rate=5,
                 replace_folder=None,
                 as_list=0,
                 sub_mean=0,
                 mean_path=None,
                 with_color_noise=0,
                 which_place=0,
                 *args, **kwargs
                 ):
        self.group = group
        self.batch_size = batch_size
        self.categorymulti = categorymulti
        self.ret_list = categorymulti>1 or as_list==1
        self.queue_params = queue_params
        self.withflip = withflip
        self.crop_each = crop_each

        self.shuffle_flag = group=='train'

        if no_shuffle==1:
            self.shuffle_flag = False

        if self.ret_list:
            assert not self.queue_params==None, "Must send queue params in"

            self.queue_params_list = []
            self.data_params_list = []

        self.crop_size = 224
        if not crop_size==None:
            self.crop_size = crop_size

        self.all_providers = []

        if cfg_dataset.get('scenenet', 0)==1:
            # Keys for scenenet, (240, 320), raw
            self.image_scenenet = 'image_scenenet'
            self.depth_scenenet = 'depth_scenenet'
            self.normal_scenenet = 'normal_scenenet'
            self.instance_scenenet = 'instance_scenenet'

            #postproc_scenenet = lambda x: self.postproc_flag(x, NOW_SIZE1 = 240, NOW_SIZE2 = 320, seed_random = 0)
            postproc_scenenet = (self.postproc_flag, (), {'NOW_SIZE1' : 240, 'NOW_SIZE2' : 320, 'seed_random' : 0})

            if whichscenenet<2:
                #postprocess_scenenet = {self.image_scenenet: [(postproc_scenenet, (), {})]}
                postprocess_scenenet = {self.image_scenenet: [postproc_scenenet]}
            else:
                #postprocess_scenenet = {self.image_scenenet: [(lambda x: self.postprocess_images(x, dtype_now = tf.uint8, shape_now = (240, 320, 3)), (), {}), (postproc_scenenet, (), {})]
                postprocess_scenenet = {self.image_scenenet: [(self.postprocess_images , (), {'dtype_now' : tf.uint8, 'shape_now' : (240, 320, 3)}), postproc_scenenet]
                                  }

            need_normal = cfg_dataset.get('scene_normal', 1)==1 and nfromd==0
            need_depth = cfg_dataset.get('scene_depth', 1)==1 or (cfg_dataset.get('scene_normal', 1)==1 and nfromd!=0)
            need_instance = cfg_dataset.get('scene_instance', 0)==1

            if need_normal:
                postproc_scenenet_normal = postproc_scenenet
                if flipnormal==1:
                    postproc_scenenet_normal = (self.postproc_flag, (), {'NOW_SIZE1' : 240, 'NOW_SIZE2' : 320, 'seed_random' : 0, 'is_normal': 1})

                if whichscenenet<2:
                    #postprocess_scenenet[self.normal_scenenet] = [(postproc_scenenet, (), {})]
                    postprocess_scenenet[self.normal_scenenet] = [postproc_scenenet_normal]
                else:
                    #postprocess_scenenet[self.normal_scenenet] = [(lambda x: self.postprocess_images(x, dtype_now = tf.uint8, shape_now = (240, 320, 3)), (), {}), (postproc_scenenet, (), {})]
                    postprocess_scenenet[self.normal_scenenet] = [(self.postprocess_images, (), { 'dtype_now' : tf.uint8, 'shape_now' : (240, 320, 3)}), postproc_scenenet_normal]

            if need_depth:
                if whichscenenet<2:
                    #postprocess_scenenet[self.depth_scenenet] = [(self.postprocess_rawdepth, (), {}), (postproc_scenenet, (), {})]
                    postprocess_scenenet[self.depth_scenenet] = [(self.postprocess_rawdepth, (), {}), postproc_scenenet]
                else:
                    #postprocess_scenenet[self.depth_scenenet] = [(lambda x: self.postprocess_images(x, dtype_now = tf.uint16, shape_now = (240, 320, 1)), (), {}), (postproc_scenenet, (), {})]
                    postprocess_scenenet[self.depth_scenenet] = [(self.postprocess_images, (), {'dtype_now' : tf.uint16, 'shape_now' : (240, 320, 1)}), postproc_scenenet]

                if depthnormal==1:
                    postprocess_scenenet[self.depth_scenenet].append((self.postprocess_normalize, (), {}))

            if need_instance:
                assert whichscenenet>=2, "Use the new scenenet, old one is not supported"
                #postprocess_scenenet[self.instance_scenenet] = [(lambda x: self.postprocess_images(x, dtype_now = tf.uint8, shape_now = (240, 320, 1)), (), {}), (postproc_scenenet, (), {})]
                #postprocess_scenenet[self.instance_scenenet] = [(lambda x: self.postprocess_images(x, dtype_now = tf.uint16, shape_now = (240, 320, 1)), (), {}), (postproc_scenenet, (), {})]
                postprocess_scenenet[self.instance_scenenet] = [(self.postprocess_images, (), {'dtype_now' : tf.uint16, 'shape_now' : (240, 320, 1)}), postproc_scenenet]

            source_prefix = 'scenenet'
            if whichscenenet==1:
                source_prefix = 'scenenet_compress'
            elif whichscenenet==2:
                source_prefix = 'scenenet_new'

            source_dirs_scenenet = [data_path["%s/%s/images" % (source_prefix, group)]]
            if need_normal:
                source_dirs_scenenet.append(data_path["%s/%s/normals" % (source_prefix, group)])
            if need_depth:
                source_dirs_scenenet.append(data_path["%s/%s/depths" % (source_prefix, group)])
            if need_instance:
                source_dirs_scenenet.append(data_path["%s/%s/instances" % (source_prefix, group)])

            if whichscenenet<2:
                trans_dicts_scenenet = [{'image_raw': self.image_scenenet}]
            else:
                trans_dicts_scenenet = [{'photo': self.image_scenenet}]

            if need_normal:
                if whichscenenet<2:
                    trans_dicts_scenenet.append({'image_raw': self.normal_scenenet})
                else:
                    trans_dicts_scenenet.append({'normals': self.normal_scenenet})
            if need_depth:
                if whichscenenet<2:
                    trans_dicts_scenenet.append({'image_raw': self.depth_scenenet})
                else:
                    trans_dicts_scenenet.append({'depth': self.depth_scenenet})
            if need_instance:
                #trans_dicts_scenenet.append({'instance': self.instance_scenenet})
                trans_dicts_scenenet.append({'classes': self.instance_scenenet})

            #trans_dicts_scenenet = [{'image_raw': self.image_scenenet}, {'image_raw': self.normal_scenenet}]
            if self.ret_list:
                curr_data_params = {
                        'func': data.TFRecordsParallelByFileProvider,
                        'source_dirs': source_dirs_scenenet,
                        'trans_dicts': trans_dicts_scenenet,
                        'postprocess': postprocess_scenenet,
                        'batch_size': batch_size,
                        'n_threads': n_threads,
                        'shuffle': self.shuffle_flag,
                        }
                curr_data_params.update(kwargs)
                self.data_params_list.append(curr_data_params)
                self.queue_params_list.append(self.queue_params)
            else:
                self.all_providers.append(data.TFRecordsParallelByFileProvider(source_dirs = source_dirs_scenenet, 
                                                trans_dicts = trans_dicts_scenenet, 
                                                postprocess = postprocess_scenenet, 
                                                batch_size = batch_size, 
                                                n_threads=n_threads,
                                                shuffle = self.shuffle_flag,
                                                *args, **kwargs
                                                ))

        if cfg_dataset.get('scannet', 0)==1:
            # Keys for scannet, (480, 640), png
            self.image_scannet = 'image_scannet'
            self.depth_scannet = 'depth_scannet'

            SIZE_1 = 480
            SIZE_2 = 640

            if whichscannet==1:
                SIZE_1 = 240
                SIZE_2 = 320

            #postproc_scannet = lambda x: self.postproc_flag(x, NOW_SIZE1 = SIZE_1, NOW_SIZE2 = SIZE_2, seed_random = 1)
            postproc_scannet = (self.postproc_flag, (), {'NOW_SIZE1': SIZE_1, 'NOW_SIZE2' : SIZE_2, 'seed_random' : 1})
            if whichscannet<2:
                #postprocess_scannet = {self.image_scannet: [(lambda x: self.postprocess_images(x, dtype_now = tf.uint8, shape_now = (SIZE_1, SIZE_2, 3)), (), {}), (postproc_scannet, (), {})], 
                #        self.depth_scannet: [(lambda x: self.postprocess_images(x, dtype_now = tf.uint16, shape_now = (SIZE_1, SIZE_2, 1)), (), {}), (postproc_scannet, (), {})]}
                postprocess_scannet = {self.image_scannet: [(self.postprocess_images, (), {'dtype_now' : tf.uint8, 'shape_now' : (SIZE_1, SIZE_2, 3)}), postproc_scannet], 
                        self.depth_scannet: [(self.postprocess_images, (), {'dtype_now' : tf.uint16, 'shape_now' : (SIZE_1, SIZE_2, 1)}), postproc_scannet]}
            else:
                #postprocess_scannet = {self.image_scannet: [(lambda x: self.postprocess_images(x, dtype_now = tf.uint8, shape_now = (240, 320, 3)), (), {}), (postproc_scannet, (), {})], 
                #        self.depth_scannet: [(lambda x: self.postprocess_images(x, dtype_now = tf.uint16, shape_now = (480, 640, 1)), (), {}), (lambda x: self.postprocess_resize(x), (), {}), (postproc_scannet, (), {})]}
                postprocess_scannet = {self.image_scannet: [(self.postprocess_images, (), {'dtype_now' : tf.uint8, 'shape_now' : (240, 320, 3)}), postproc_scannet], 
                        self.depth_scannet: [(self.postprocess_images, (), {'dtype_now' : tf.uint16, 'shape_now' : (480, 640, 1)}), (self.postprocess_resize, (), {}), postproc_scannet]}

            if whichscannet<2:
                source_prefix = 'scannet'
                if whichscannet==1:
                    source_prefix = 'scannet_re'
                #source_dirs_scannet = [data_path["scannet/%s/images" % group], data_path["scannet/%s/depths" % group]]
                source_dirs_scannet = [data_path["%s/%s/images" % (source_prefix, group)], data_path["%s/%s/depths" % (source_prefix, group)]]
            else:
                source_dirs_scannet = [data_path["scannet_re/%s/images" % group], data_path["scannet/%s/depths" % group]]

            trans_dicts_scannet = [{'image': self.image_scannet}, {'depth': self.depth_scannet}]
            if self.ret_list:
                curr_data_params = {
                        'func': data.TFRecordsParallelByFileProvider,
                        'source_dirs': source_dirs_scannet,
                        'trans_dicts': trans_dicts_scannet,
                        'postprocess': postprocess_scannet,
                        'batch_size': batch_size,
                        'n_threads': n_threads,
                        'shuffle': self.shuffle_flag,
                        }
                curr_data_params.update(kwargs)
                self.data_params_list.append(curr_data_params)
                self.queue_params_list.append(self.queue_params)
            else:
                self.all_providers.append(data.TFRecordsParallelByFileProvider(source_dirs = source_dirs_scannet, 
                                                trans_dicts = trans_dicts_scannet, 
                                                postprocess = postprocess_scannet, 
                                                batch_size = batch_size, 
                                                n_threads=n_threads,
                                                shuffle = self.shuffle_flag,
                                                *args, **kwargs
                                                ))

        if cfg_dataset.get('pbrnet', 0)==1:
            # Keys for pbrnet, (480, 640), png, TODO: valid?
            self.image_pbrnet = 'image_pbrnet'
            self.depth_pbrnet = 'depth_pbrnet'
            self.normal_pbrnet = 'normal_pbrnet'
            self.instance_pbrnet = 'instance_pbrnet'

            #postproc_pbrnet = lambda x: self.postproc_flag(x, NOW_SIZE1 = 480, NOW_SIZE2 = 640, seed_random = 2)
            postproc_pbrnet = (self.postproc_flag, (), { 'NOW_SIZE1' : 480, 'NOW_SIZE2' : 640, 'seed_random' : 2})
            #postprocess_pbrnet = {self.image_pbrnet: [(lambda x: self.postprocess_images(x, dtype_now = tf.uint8, shape_now = (480, 640, 3)), (), {}), (postproc_pbrnet, (), {})]
            postprocess_pbrnet = {self.image_pbrnet: [(self.postprocess_images, (), {'dtype_now' : tf.uint8, 'shape_now' : (480, 640, 3)}), postproc_pbrnet]
                                  }

            need_normal = cfg_dataset.get('pbr_normal', 1)==1 and nfromd==0
            need_depth = cfg_dataset.get('pbr_depth', 1)==1 or (cfg_dataset.get('pbr_normal', 1)==1 and nfromd!=0)
            need_instance = cfg_dataset.get('pbr_instance', 0)==1

            if need_normal:
                #postprocess_pbrnet[self.normal_pbrnet] = [(lambda x: self.postprocess_images(x, dtype_now = tf.uint8, shape_now = (480, 640, 3)), (), {}), (postproc_pbrnet, (), {})]
                postproc_pbrnet_normal = postproc_pbrnet
                if flipnormal==1:
                    postproc_pbrnet_normal = (self.postproc_flag, (), { 'NOW_SIZE1' : 480, 'NOW_SIZE2' : 640, 'seed_random' : 2, 'is_normal': 1})
                postprocess_pbrnet[self.normal_pbrnet] = [(self.postprocess_images, (), {'dtype_now' : tf.uint8, 'shape_now' : (480, 640, 3)}), postproc_pbrnet_normal]
            if need_depth:
                #postprocess_pbrnet[self.depth_pbrnet] = [(lambda x: self.postprocess_images(x, dtype_now = tf.uint16, shape_now = (480, 640, 1)), (), {}), (postproc_pbrnet, (), {})]
                postprocess_pbrnet[self.depth_pbrnet] = [(self.postprocess_images, (), {'dtype_now' : tf.uint16, 'shape_now' : (480, 640, 1)}), postproc_pbrnet]
                if depthnormal==1:
                    postprocess_pbrnet[self.depth_pbrnet].append((self.postprocess_normalize, (), {}))

            if need_instance:
                #postprocess_pbrnet[self.instance_pbrnet] = [(lambda x: self.postprocess_images(x, dtype_now = tf.uint8, shape_now = (480, 640, 1)), (), {}), (postproc_pbrnet, (), {})]
                postprocess_pbrnet[self.instance_pbrnet] = [(self.postprocess_images, (), {'dtype_now' : tf.uint8, 'shape_now' : (480, 640, 1)}), postproc_pbrnet]

            source_dirs_pbrnet = [data_path["pbrnet/%s/images" % group]]
            if need_normal:
                source_dirs_pbrnet.append(data_path["pbrnet/%s/normals" % group])
            if need_depth:
                source_dirs_pbrnet.append(data_path["pbrnet/%s/depths" % group])
            if need_instance:
                source_dirs_pbrnet.append(data_path["pbrnet/%s/instances" % group])

            trans_dicts_pbrnet = [{'mlt': self.image_pbrnet}]
            if need_normal:
                trans_dicts_pbrnet.append({'normal': self.normal_pbrnet})
            if need_depth:
                trans_dicts_pbrnet.append({'depth': self.depth_pbrnet})
            if need_instance:
                trans_dicts_pbrnet.append({'category': self.instance_pbrnet})

            if self.ret_list:
                curr_data_params = {
                        'func': data.TFRecordsParallelByFileProvider,
                        'source_dirs': source_dirs_pbrnet,
                        'trans_dicts': trans_dicts_pbrnet,
                        'postprocess': postprocess_pbrnet,
                        'batch_size': batch_size,
                        'n_threads': n_threads,
                        'shuffle': self.shuffle_flag,
                        }
                curr_data_params.update(kwargs)
                self.data_params_list.append(curr_data_params)
                self.queue_params_list.append(self.queue_params)
            else:
                self.all_providers.append(data.TFRecordsParallelByFileProvider(source_dirs = source_dirs_pbrnet, 
                                                trans_dicts = trans_dicts_pbrnet, 
                                                postprocess = postprocess_pbrnet, 
                                                batch_size = batch_size, 
                                                n_threads=n_threads,
                                                shuffle = self.shuffle_flag,
                                                *args, **kwargs
                                                ))

        if cfg_dataset.get('imagenet', 0)==1:
            postproc_imagenet = [self.postproc_flag, (), { 
                'NOW_SIZE1':256, 
                'NOW_SIZE2':256, 
                'seed_random':3, 
                'with_noise':with_noise, 
                'noise_level':noise_level, 
                'with_flip':onlyflip_cate, 
                'eliprep':eliprep, 
                'thprep':thprep,
                'sub_mean':sub_mean, 
                'mean_path':mean_path,
                'with_color_noise':with_color_noise,
                }]
            if whichimagenet==0:
                # Keys for imagenet, (256, 256), raw
                self.image_imagenet = 'image_imagenet'
                self.label_imagenet = 'label_imagenet'

                postprocess_imagenet = {self.image_imagenet: [postproc_imagenet],
                                        self.label_imagenet: [(self.postproc_label, (), {})]
                                        }

                source_dirs_imagenet = [data_path["imagenet/images"], data_path["imagenet/labels"]]
                trans_dicts_imagenet = [{'images': self.image_imagenet}, {'labels': self.label_imagenet}]
                if group=='train':
                    file_pattern = 'train*.tfrecords'
                else:
                    file_pattern = 'val*.tfrecords'

                curr_batch_size = batch_size

                if self.ret_list:
                    curr_data_params = {
                            'func': data.TFRecordsParallelByFileProvider,
                            'source_dirs': source_dirs_imagenet,
                            'trans_dicts': trans_dicts_imagenet,
                            'postprocess': postprocess_imagenet,
                            'batch_size': batch_size,
                            'n_threads': n_threads,
                            'shuffle': self.shuffle_flag,
                            'file_pattern': file_pattern,
                            }
                    curr_data_params.update(kwargs)
                    self.data_params_list.append(curr_data_params)

                    curr_queue_params = copy.deepcopy(self.queue_params)
                    curr_queue_params['batch_size'] = curr_queue_params['batch_size']*self.categorymulti
                    self.queue_params_list.append(curr_queue_params)
                else:
                    self.all_providers.append(data.TFRecordsParallelByFileProvider(source_dirs = source_dirs_imagenet, 
                                                    trans_dicts = trans_dicts_imagenet, 
                                                    postprocess = postprocess_imagenet, 
                                                    batch_size = curr_batch_size, 
                                                    n_threads=n_threads,
                                                    file_pattern=file_pattern,
                                                    shuffle = self.shuffle_flag,
                                                    *args, **kwargs
                                                    ))
            elif whichimagenet==1:

                self.image_imagenet = 'image_imagenet'
                self.label_imagenet = 'label_imagenet'

                postprocess_imagenet = {self.image_imagenet: [(self.postprocess_images, (), {'dtype_now' : tf.uint8, 'shape_now' : (256, 256, 3)}), postproc_imagenet],
                                    self.label_imagenet: [(self.postproc_label, (), {})]
                                    }

                source_dirs_imagenet = [data_path["imagenet/%s/images" % group], data_path["imagenet/%s/labels" % group]]
                trans_dicts_imagenet = [{'image': self.image_imagenet}, {'label': self.label_imagenet}]

                curr_batch_size = batch_size
                shuffle_flag = self.shuffle_flag

                if self.ret_list:
                    curr_data_params = {
                            'func': data.TFRecordsParallelByFileProvider,
                            'source_dirs': source_dirs_imagenet,
                            'trans_dicts': trans_dicts_imagenet,
                            'postprocess': postprocess_imagenet,
                            'batch_size': batch_size,
                            'n_threads': n_threads,
                            'shuffle': shuffle_flag,
                            }
                    curr_data_params.update(kwargs)
                    self.data_params_list.append(curr_data_params)

                    curr_queue_params = copy.deepcopy(self.queue_params)
                    curr_queue_params['batch_size'] = curr_queue_params['batch_size']*self.categorymulti
                    self.queue_params_list.append(curr_queue_params)
                else:
                    self.all_providers.append(data.TFRecordsParallelByFileProvider(source_dirs = source_dirs_imagenet, 
                                                    trans_dicts = trans_dicts_imagenet, 
                                                    postprocess = postprocess_imagenet, 
                                                    batch_size = curr_batch_size,
                                                    n_threads=n_threads,
                                                    shuffle = shuffle_flag,
                                                    *args, **kwargs
                                                    ))
            elif whichimagenet==2 or whichimagenet==3 or whichimagenet==4:
                # whichimagenet==3 should not be used, it's a wrong dataset
                # Keys for imagenet, (256, 256), raw
                self.image_imagenet = 'image_imagenet'
                self.label_imagenet = 'label_imagenet'

                postprocess_imagenet = {self.image_imagenet: [postproc_imagenet],
                                        self.label_imagenet: [(self.postproc_label, (), {})]
                                        }

                if whichimagenet==2:
                    source_dirs_imagenet = [data_path["imagenet/image_label"]]
                elif whichimagenet==3:
                    source_dirs_imagenet = [data_path["imagenet/image_label_hdf5"]]
                elif whichimagenet==4:
                    source_dirs_imagenet = [data_path["imagenet/image_label_part"]]

                trans_dicts_imagenet = [{'images': self.image_imagenet, 'labels': self.label_imagenet}]
                if group=='train':
                    file_pattern = 'train*.tfrecords'
                else:
                    file_pattern = 'val*.tfrecords'

                curr_batch_size = batch_size

                if self.ret_list:
                    curr_data_params = {
                            'func': data.TFRecordsParallelByFileProvider,
                            'source_dirs': source_dirs_imagenet,
                            'trans_dicts': trans_dicts_imagenet,
                            'postprocess': postprocess_imagenet,
                            'batch_size': batch_size,
                            'n_threads': n_threads,
                            'shuffle': self.shuffle_flag,
                            'file_pattern': file_pattern,
                            }
                    curr_data_params.update(kwargs)
                    self.data_params_list.append(curr_data_params)

                    curr_queue_params = copy.deepcopy(self.queue_params)
                    curr_queue_params['batch_size'] = curr_queue_params['batch_size']*self.categorymulti
                    self.queue_params_list.append(curr_queue_params)
                else:
                    self.all_providers.append(data.TFRecordsParallelByFileProvider(source_dirs = source_dirs_imagenet, 
                                                    trans_dicts = trans_dicts_imagenet, 
                                                    postprocess = postprocess_imagenet, 
                                                    batch_size = curr_batch_size, 
                                                    n_threads=n_threads,
                                                    file_pattern=file_pattern,
                                                    shuffle = self.shuffle_flag,
                                                    *args, **kwargs
                                                    ))
            elif whichimagenet==5:
                # Use full dataset
                self.image_imagenet = 'image_imagenet'
                self.label_imagenet = 'label_imagenet'

                postprocess_imagenet = {self.image_imagenet: [(tf.reshape, (), {'shape':[]}), (tf.image.decode_image, (), {'channels':3}), postproc_imagenet],
                                        self.label_imagenet: [(self.postproc_label, (), {})]
                                        }

                source_dirs_imagenet = [data_path["imagenet/image_label_full"]]
                trans_dicts_imagenet = [{'images': self.image_imagenet, 'labels': self.label_imagenet}]
                if group=='train':
                    file_pattern = 'train*'
                else:
                    file_pattern = 'val*'

                curr_batch_size = batch_size

                if self.ret_list:
                    curr_data_params = {
                            'func': data.TFRecordsParallelByFileProvider,
                            'source_dirs': source_dirs_imagenet,
                            'trans_dicts': trans_dicts_imagenet,
                            'postprocess': postprocess_imagenet,
                            'batch_size': batch_size,
                            'n_threads': n_threads,
                            'shuffle': self.shuffle_flag,
                            'file_pattern': file_pattern,
                            }
                    curr_data_params.update(kwargs)
                    self.data_params_list.append(curr_data_params)

                    curr_queue_params = copy.deepcopy(self.queue_params)
                    curr_queue_params['batch_size'] = curr_queue_params['batch_size']*self.categorymulti
                    self.queue_params_list.append(curr_queue_params)
                else:
                    self.all_providers.append(data.TFRecordsParallelByFileProvider(source_dirs = source_dirs_imagenet, 
                                                    trans_dicts = trans_dicts_imagenet, 
                                                    postprocess = postprocess_imagenet, 
                                                    batch_size = curr_batch_size, 
                                                    n_threads=n_threads,
                                                    file_pattern=file_pattern,
                                                    shuffle = self.shuffle_flag,
                                                    *args, **kwargs
                                                    ))

        if cfg_dataset.get('coco', 0)==1:
            key_list = ['height', 'images', 'labels', 'num_objects', \
                    'segmentation_masks', 'width']
            BYTES_KEYs = ['images', 'labels', 'segmentation_masks']

            if whichcoco==0:
                source_dirs = [data_path['coco/%s/%s' % (self.group, v)] for v in key_list]
            else:
                source_dirs = [data_path['coco_no0/%s/%s' % (self.group, v)] for v in key_list]

            meta_dicts = [{v : {'dtype': tf.string, 'shape': []}} if v in BYTES_KEYs else {v : {'dtype': tf.int64, 'shape': []}} for v in key_list]
            if self.ret_list:
                curr_data_params = {
                        'func': coco_provider.COCO,
                        'source_dirs': source_dirs,
                        'meta_dicts': meta_dicts,
                        'group': group,
                        'batch_size': batch_size,
                        'n_threads': n_threads,
                        'image_min_size': 240,
                        'crop_height': 224,
                        'crop_width': 224,
                        }
                curr_data_params.update(kwargs)
                self.data_params_list.append(curr_data_params)
                self.queue_params_list.append(self.queue_params)
            else:
                self.all_providers.append(coco_provider.COCO(source_dirs = source_dirs,
                                            meta_dicts = meta_dicts,
                                            group = group,
                                            batch_size = batch_size,
                                            n_threads = n_threads,
                                            image_min_size = 240,
                                            crop_height = 224,
                                            crop_width = 224,
                                            *args, **kwargs
                                            ))

        if cfg_dataset.get('place', 0)==1:

            self.image_place = 'image_place'
            self.label_place = 'label_place'

            postproc_place = [self.postproc_flag, (), { 
                'NOW_SIZE1':256, 
                'NOW_SIZE2':256, 
                'seed_random':4, 
                'with_noise':with_noise, 
                'noise_level':noise_level, 
                'with_flip':onlyflip_cate,
                'with_color_noise':with_color_noise,
                }]
            postprocess_place = {self.image_place: [(self.postprocess_images, (), {'dtype_now' : tf.uint8, 'shape_now' : (256, 256, 3)}), postproc_place],
                                self.label_place: [(self.postproc_label, (), {})]
                                }

            if which_place==0:
                source_dirs_place = [
                        data_path["place/%s/images" % group], 
                        data_path["place/%s/labels" % group]
                        ]
            else:
                source_dirs_place = [
                        data_path["place/%s/images_part" % group], 
                        data_path["place/%s/labels_part" % group],
                        ]
            trans_dicts_place = [{'image': self.image_place}, {'label': self.label_place}]

            curr_batch_size = batch_size

            if self.ret_list:
                curr_data_params = {
                        'func': data.TFRecordsParallelByFileProvider,
                        'source_dirs': source_dirs_place,
                        'trans_dicts': trans_dicts_place,
                        'postprocess': postprocess_place,
                        'batch_size': batch_size,
                        'n_threads': n_threads,
                        'shuffle': self.shuffle_flag,
                        }
                curr_data_params.update(kwargs)
                self.data_params_list.append(curr_data_params)

                curr_queue_params = copy.deepcopy(self.queue_params)
                curr_queue_params['batch_size'] = curr_queue_params['batch_size']*self.categorymulti
                self.queue_params_list.append(curr_queue_params)
            else:
                self.all_providers.append(data.TFRecordsParallelByFileProvider(source_dirs = source_dirs_place, 
                                                trans_dicts = trans_dicts_place, 
                                                postprocess = postprocess_place, 
                                                batch_size = curr_batch_size,
                                                n_threads=n_threads,
                                                shuffle = self.shuffle_flag,
                                                *args, **kwargs
                                                ))

        if cfg_dataset.get('kinetics', 0)==1:
            #key_list = ['path', 'label_p', 'height_p', 'width_p']
            key_list = ['path', 'label_p']

            source_dirs = [data_path['kinetics/%s/%s' % (self.group, v)] for v in key_list]

            if self.ret_list:
                curr_data_params = {
                        'func': kinetics_provider.Kinetics,
                        'source_dirs': source_dirs,
                        'group': group,
                        'batch_size': batch_size,
                        'n_threads': n_threads,
                        'crop_height': 224,
                        'crop_width': 224,
                        'crop_time': crop_time,
                        'crop_rate': crop_rate,
                        'shuffle': self.shuffle_flag,
                        'replace_folder': replace_folder,
                        'sub_mean': sub_mean,
                        }
                curr_data_params.update(kwargs)
                self.data_params_list.append(curr_data_params)
                self.queue_params_list.append(self.queue_params)
            else:
                self.all_providers.append(kinetics_provider.Kinetics(source_dirs = source_dirs,
                                            group = group,
                                            batch_size = batch_size,
                                            n_threads = n_threads,
                                            crop_height = 224,
                                            crop_width = 224,
                                            crop_time = crop_time,
                                            crop_rate = crop_rate,
                                            replace_folder = replace_folder,
                                            shuffle = self.shuffle_flag,
                                            sub_mean =  sub_mean,
                                            *args, **kwargs
                                            ))

        if cfg_dataset.get('nyuv2', 0)==1:

            # Keys for nyuv2, (480, 640), png
            self.image_nyuv2 = 'image_nyuv2'
            self.depth_nyuv2 = 'depth_nyuv2'

            SIZE_1 = 480
            SIZE_2 = 640

            #postproc_nyuv2 = lambda x: self.postproc_flag(x, NOW_SIZE1 = SIZE_1, NOW_SIZE2 = SIZE_2, seed_random = 5)
            postproc_nyuv2 = [self.postproc_flag, (), {'NOW_SIZE1' : SIZE_1, 'NOW_SIZE2' : SIZE_2, 'seed_random' : 5}]
            #postprocess_nyuv2 = {self.image_nyuv2: [(lambda x: self.postprocess_images(x, dtype_now = tf.uint8, shape_now = (SIZE_1, SIZE_2, 3)), (), {}), (postproc_nyuv2, (), {})], 
            postprocess_nyuv2 = {self.image_nyuv2: [(self.postprocess_images, (), {'dtype_now' : tf.uint8, 'shape_now' : (SIZE_1, SIZE_2, 3)}), postproc_nyuv2], 
                    #self.depth_nyuv2: [(lambda x: self.postprocess_images(x, dtype_now = tf.uint16, shape_now = (SIZE_1, SIZE_2, 1)), (), {}), (postproc_nyuv2, (), {})]}
                    self.depth_nyuv2: [(self.postprocess_images, (), {'dtype_now' : tf.uint16, 'shape_now' : (SIZE_1, SIZE_2, 1)}), postproc_nyuv2]}

            if depthnormal==1:
                postprocess_nyuv2[self.depth_nyuv2].append((self.postprocess_normalize, (), {}))

            source_prefix = 'nyuv2'
            source_dirs_nyuv2 = [data_path["%s/%s/images" % (source_prefix, group)], data_path["%s/%s/depths" % (source_prefix, group)]]

            trans_dicts_nyuv2 = [{'image': self.image_nyuv2}, {'depth': self.depth_nyuv2}]
            if self.ret_list:
                curr_data_params = {
                        'func': data.TFRecordsParallelByFileProvider,
                        'source_dirs': source_dirs_nyuv2,
                        'trans_dicts': trans_dicts_nyuv2,
                        'postprocess': postprocess_nyuv2,
                        'batch_size': batch_size,
                        'n_threads': n_threads,
                        'shuffle': self.shuffle_flag,
                        }
                curr_data_params.update(kwargs)
                self.data_params_list.append(curr_data_params)
                self.queue_params_list.append(self.queue_params)
            else:
                self.all_providers.append(data.TFRecordsParallelByFileProvider(source_dirs = source_dirs_nyuv2, 
                                                trans_dicts = trans_dicts_nyuv2, 
                                                postprocess = postprocess_nyuv2, 
                                                batch_size = batch_size, 
                                                n_threads=n_threads,
                                                shuffle = self.shuffle_flag,
                                                *args, **kwargs
                                                ))


    def postproc_label(self, labels):

        curr_batch_size = self.batch_size

        labels.set_shape([curr_batch_size])
        
        if curr_batch_size==1:
            labels = tf.squeeze(labels, axis = [0])

        return labels

    def postproc_flag(self, images, 
            NOW_SIZE1=256, 
            NOW_SIZE2=256, 
            seed_random=0, 
            curr_batch_size=None, 
            with_noise=0, 
            noise_level=10, 
            with_flip=0, 
            is_normal=0, 
            eliprep=0, 
            thprep=0,
            sub_mean=0, 
            mean_path=None, 
            with_color_noise=0):

        if curr_batch_size==None:
            curr_batch_size = self.batch_size
            
        orig_dtype = images.dtype
        norm = tf.cast(images, tf.float32)
        #print(norm.get_shape().as_list())

        if eliprep==1:
            def prep_each(norm_):
                _RESIZE_SIDE_MIN = 256
                _RESIZE_SIDE_MAX = 512

                if self.group == 'train':
                    im = preprocess_image(norm_, self.crop_size, self.crop_size, is_training=True,
                                             resize_side_min=_RESIZE_SIDE_MIN,
                                             resize_side_max=_RESIZE_SIDE_MAX)
                else:
                    im = preprocess_image(norm_, self.crop_size, self.crop_size, is_training=False,
                                             resize_side_min=_RESIZE_SIDE_MIN,
                                             resize_side_max=_RESIZE_SIDE_MAX)

                return im
            crop_images = tf.map_fn(prep_each, norm)
        elif thprep==1:
            def prep_each(norm_):
                im = preprocessing_th(norm_, self.crop_size, self.crop_size, 
                        is_training=self.group=='train', seed_random=seed_random)
                return im
            #crop_images = tf.map_fn(prep_each, images)
            crop_images = prep_each(images)
        else:
            if with_color_noise==1 and self.group=='train':
                order_temp = tf.constant([0,1,2], dtype=tf.int32)
                order_rand = tf.random_shuffle(order_temp, seed=seed_random)

                fn_pred_fn_pairs = lambda x, image: [
                        (tf.equal(x, order_temp[0]), lambda :tf.image.random_saturation(image, 0.6, 1.4, seed=seed_random)),
                        (tf.equal(x, order_temp[1]), lambda :tf.image.random_brightness(image, 0.4, seed=seed_random)),
                        ]
                default_fn = lambda image: tf.image.random_contrast(image, 0.6, 1.4, seed=seed_random)
                def _color_jitter_one(_norm):
                    orig_shape = _norm.get_shape().as_list()
                    _norm = tf.case(fn_pred_fn_pairs(order_rand[0], _norm), default = lambda : default_fn(_norm))
                    _norm = tf.case(fn_pred_fn_pairs(order_rand[1], _norm), default = lambda : default_fn(_norm))
                    _norm = tf.case(fn_pred_fn_pairs(order_rand[2], _norm), default = lambda : default_fn(_norm))
                    _norm.set_shape(orig_shape)

                    return _norm
                norm = tf.map_fn(_color_jitter_one, norm)

            if sub_mean==1:
                IMAGENET_MEAN = tf.constant(np.load(mean_path).swapaxes(0,1).swapaxes(1,2)[:,:,::-1], dtype = tf.float32)
                orig_dtype = tf.float32
                norm = norm - IMAGENET_MEAN

            if self.group=='train':

                shape_tensor = norm.get_shape().as_list()
                if self.crop_each==0:

                    crop_images = tf.random_crop(norm, [curr_batch_size, self.crop_size, self.crop_size, shape_tensor[3]], seed=seed_random)
                else:
                    off_sta_x = tf.random_uniform(shape = [curr_batch_size, 1], maxval = NOW_SIZE1 - self.crop_size, dtype = tf.int32, seed = 2*seed_random)
                    off_end_x = off_sta_x + self.crop_size
                    off_sta_x = off_sta_x / (NOW_SIZE1 - 1)
                    off_end_x = off_end_x / (NOW_SIZE1 - 1)

                    off_sta_y = tf.random_uniform(shape = [curr_batch_size, 1], maxval = NOW_SIZE2 - self.crop_size, dtype = tf.int32, seed = 2*seed_random + 1)
                    off_end_y = off_sta_y + self.crop_size
                    off_sta_y = off_sta_y / (NOW_SIZE2 - 1)
                    off_end_y = off_end_y / (NOW_SIZE2 - 1)

                    off = tf.concat([off_sta_x, off_sta_y, off_end_x, off_end_y], axis = 1)
                    off = tf.cast(off, tf.float32)

                    box_ind    = tf.constant(range(curr_batch_size))

                    crop_images = tf.image.crop_and_resize(norm, off, box_ind, tf.constant([self.crop_size, self.crop_size]))

                if self.withflip==1 or with_flip==1:
                    def _postprocess_flip(im):
                        # Original way of flipping, changing to random_uniform way to be more controllable
                        #im = tf.image.random_flip_left_right(im, seed = seed_random)
                        #return im
                        do_flip = tf.random_uniform(shape = [1], minval=0, maxval=1, dtype=tf.float32, seed = seed_random)

                        def __left_right_flip(im):
                            flipped = tf.image.flip_left_right(im)
                            if is_normal==1:
                                #flipped = 256 - flipped
                                flipped_x, flipped_y, flipped_z = tf.unstack(flipped, axis = 2)
                                flipped = tf.stack([256 - flipped_x, flipped_y, flipped_z], axis = 2)
                            return flipped

                        return tf.cond(tf.less(do_flip[0], 0.5), fn1 = lambda: __left_right_flip(im), fn2 = lambda: im)

                    crop_images = tf.map_fn(_postprocess_flip, crop_images, dtype = crop_images.dtype)

                if with_noise==1:
                    def _postprocess_noise(im):
                        do_noise = tf.random_uniform(shape = [1], minval=0, maxval=1, dtype=tf.float32, seed = None)

                        def __add_noise(im):
                            curr_level = tf.random_uniform(shape = [1], minval=0, maxval=noise_level, dtype=tf.float32, seed = None)
                            curr_noise = tf.random_normal(shape = tf.shape(im), mean=0.0, stddev=curr_level, dtype=tf.float32)

                            return tf.add(im, curr_noise)
                        
                        #return tf.cond(tf.less(do_noise[0], 0.5), true_fn = lambda: __add_noise(im), false_fn = lambda: im)
                        return tf.cond(tf.less(do_noise[0], 0.5), fn1 = lambda: __add_noise(im), fn2 = lambda: im)
                    crop_images = tf.map_fn(_postprocess_noise, crop_images, dtype = crop_images.dtype)

            else:

                off = np.zeros(shape = [curr_batch_size, 4])
                off[:, 0] = int((NOW_SIZE1 - self.crop_size)/2)
                off[:, 1] = int((NOW_SIZE2 - self.crop_size)/2)
                off[:, 2:4] = off[:, :2] + self.crop_size
                off[:, 0] = off[:, 0]*1.0/(NOW_SIZE1 - 1)
                off[:, 2] = off[:, 2]*1.0/(NOW_SIZE1 - 1)

                off[:, 1] = off[:, 1]*1.0/(NOW_SIZE2 - 1)
                off[:, 3] = off[:, 3]*1.0/(NOW_SIZE2 - 1)

                box_ind    = tf.constant(range(curr_batch_size))

                crop_images = tf.image.crop_and_resize(norm, off, box_ind, tf.constant([self.crop_size, self.crop_size]))

            crop_images = tf.cast(crop_images, orig_dtype)
        if curr_batch_size==1 and thprep==0:
            crop_images = tf.squeeze(crop_images, axis=[0]) 

        return crop_images

    def postprocess_images(self, ims, dtype_now, shape_now):
        def _postprocess_images(im):
            im = tf.image.decode_png(im, dtype = dtype_now)
            im.set_shape(shape_now)
            if dtype_now==tf.uint16:
                im = tf.cast(im, tf.int32)
            return im
        if dtype_now==tf.uint16:
            write_dtype = tf.int32
        else:
            write_dtype = dtype_now
        return tf.map_fn(lambda im: _postprocess_images(im), ims, dtype=write_dtype)

    def postprocess_normalize(self, ims):
        def _postprocess_normalize(im):
            im = tf.cast(im, tf.float32)
            mean, var = tf.nn.moments(im, axes = list(range(len(im.get_shape().as_list()) - 1)))
            #var = tf.Print(var, [var], message = 'Var')
            #print(var.get_shape().as_list())
            #print(mean.get_shape().as_list())
            mean = tf.Print(mean, [mean], message = 'Mean')
            im = im - mean
            im = im / (var + 0.001)
            mean, var = tf.nn.moments(im, axes = list(range(len(im.get_shape().as_list()))))
            mean = tf.Print(mean, [mean], message = 'Mean after')
            var = tf.Print(var, [var], message = 'Var after')
            im = im - mean
            im = im / (var + 0.001)
            return im

        def _postprocess_normalize_2(im):
            im = tf.cast(im, tf.float32)
            #print(im.get_shape().as_list())
            im = tf.image.per_image_standardization(im)
            return im

        #return tf.map_fn(lambda im: _postprocess_normalize(im), ims, dtype = tf.float32)
        #return tf.map_fn(lambda im: _postprocess_normalize_2(im), ims, dtype = tf.float32)
        if self.batch_size==1:
            return _postprocess_normalize_2(ims)
        else:
            return tf.map_fn(lambda im: _postprocess_normalize_2(im), ims, dtype = tf.float32)
    
    def postprocess_resize(self, ims, newsize_1=240, newsize_2=320):
        return tf.image.resize_images(ims, (newsize_1, newsize_2))

    def postprocess_rawdepth(self, ims):
        def _postprocess_images(im):
            im = tf.decode_raw(im, tf.int32)
            im = tf.reshape(im, [240, 320, 1])
            return im
        return tf.map_fn(lambda im: _postprocess_images(im), ims, dtype=tf.int32)

    def init_ops(self):
        all_init_ops = [data_temp.init_ops() for data_temp in self.all_providers]
        num_threads = len(all_init_ops[0])

        self.ret_init_ops = []

        for indx_t in xrange(num_threads):
            curr_dict = {}
            for curr_init_ops in all_init_ops:
                curr_dict.update(curr_init_ops[indx_t])
            self.ret_init_ops.append(curr_dict)

        return self.ret_init_ops
