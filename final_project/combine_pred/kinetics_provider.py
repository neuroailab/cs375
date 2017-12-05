from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
 
import functools
import os, sys
import time
import numpy as np
import tensorflow as tf

from tfutils import data

import json
import copy

from tensorflow.python.ops import control_flow_ops

sys.path.append('../other_dataset/kinetics')

from test_str_tfr import get_frames

# Build data provider for kinetics dataset
class Kinetics(data.TFRecordsParallelByFileProvider):
    def __init__(self,
                 source_dirs,
                 group='train',
                 batch_size=1,
                 n_threads=4,
                 crop_height = 224,
                 crop_width = 224,
                 crop_time = 5,
                 crop_rate = 5,
                 replace_folder = None,
                 seed = None,
                 sub_mean = 0,
                 *args,
                 **kwargs):
        self.group = group
        self.batch_size = batch_size
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.crop_time = crop_time
        self.crop_rate = crop_rate
        self.replace_folder = replace_folder
        self.means = [123.68, 116.779, 103.939]
        self.sub_mean = sub_mean

        super(Kinetics, self).__init__(
            source_dirs = source_dirs,
            batch_size=batch_size,
            n_threads=n_threads,
            *args, **kwargs)

    def prep_data(self, data):
        for i in range(len(data)):
            inputs = data[i]
            local_get_frame = lambda vd_path: get_frames(vd_path, crop_time = self.crop_time, crop_rate  = self.crop_rate, crop_hei = self.crop_height, crop_wid = self.crop_width, replace_folder = self.replace_folder)
            curr_path = inputs['path']
            #curr_path = tf.Print(curr_path, [curr_path], message = 'Current path')
            frames_ten = tf.py_func(local_get_frame, [curr_path[0]], tf.uint8)
            #print(frames_ten)

            frame_num = self.crop_time * self.crop_rate
            frames_ten.set_shape([frame_num, self.crop_height, self.crop_width, 3])

            if self.sub_mean==1:
                frames_ten = tf.cast(frames_ten, tf.float32)
                frames_ten = frames_ten - tf.constant(self.means, tf.float32)

            #frames_ten = tf.random_crop(value = frames_ten, size = [frame_num, self.crop_height, self.crop_width, 3], seed = seed)

            labels = inputs['label_p']
            labels.set_shape([self.batch_size])
            #labels = tf.Print(labels, [labels], message = 'Current label')
            
            if self.batch_size==1:
                labels = tf.squeeze(labels, axis = [0])

            data[i] = {'image_kinetics': frames_ten, 'label_kinetics': labels}

        return data

    def init_ops(self):
        self.input_ops = super(Kinetics, self).init_ops()

        self.input_ops = self.prep_data(self.input_ops)

        return self.input_ops
