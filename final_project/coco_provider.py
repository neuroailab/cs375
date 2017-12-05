from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
 
import functools
import os, sys
import time
import numpy as np
import tensorflow as tf

from tfutils import base, data, optimizer

import json
import copy

from tensorflow.python.ops import control_flow_ops

original_labels = [1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 13, 14, 15, 16, 17, 18,
       19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38,
       39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56,
       57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77,
       78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]

max_objects = 64

labels_dict = {}
for idx, lab in enumerate(original_labels):
    labels_dict[lab] = idx + 1 # 0 is the background class

def _smallest_size_at_least(height, width, smallest_side):
    smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)

    height = tf.to_float(height)
    width = tf.to_float(width)
    smallest_side = tf.to_float(smallest_side)

    scale = tf.cond(tf.greater(height, width),
                  lambda: smallest_side / width,
                  lambda: smallest_side / height)
    new_height = tf.to_int32(height * scale)
    new_width = tf.to_int32(width * scale)
    return new_height, new_width

def preprocess_for_training(image, image_min_size = 240):
    
    ih, iw = tf.shape(image)[0], tf.shape(image)[1]

    ## min size resizing
    new_ih, new_iw = _smallest_size_at_least(ih, iw, image_min_size)
    image = tf.expand_dims(image, 0)
    image = tf.image.resize_bilinear(image, [new_ih, new_iw], align_corners=False)
    image = tf.squeeze(image, axis=[0])

    # gt_masks = tf.expand_dims(gt_masks, -1)
    # gt_masks = tf.cast(gt_masks, tf.float32)
    # gt_masks = tf.image.resize_nearest_neighbor(gt_masks, [new_ih, new_iw], align_corners=False)
    # gt_masks = tf.cast(gt_masks, tf.int32)
    # gt_masks = tf.squeeze(gt_masks, axis=[-1])

    return image

def combine_masks(gt_masks, labels):
    def _time_label(input_elem):
        gt_mask, label = input_elem
        gt_mask = tf.cast(gt_mask, tf.int32)
        label = tf.cast(label, tf.int32)
        label = label + 1
        return gt_mask*label

    return tf.add_n(tf.map_fn(_time_label, (gt_masks, labels), dtype = tf.int32))

def combine_masks_pyfunc(instance_cats, gt_mask, gt_map):
    #print(instance_cats)
    instance_cats = instance_cats[:, 0]
    sorted_indices = np.argsort(instance_cats)
    sorted_instance_cats = instance_cats[sorted_indices]
    unique_cats, reduce_idx = np.unique(sorted_instance_cats, return_index=True)
    sorted_gt_mask = gt_mask[sorted_indices, :, :]
    processed_mask = np.add.reduceat(gt_mask, reduce_idx, axis=0)
    processed_mask = np.minimum(processed_mask, 1) # ensure values are 0 and 1
    for cat_idx, cat in enumerate(unique_cats):
        gt_map = gt_map + processed_mask[cat_idx]*(cat + 1)

    print(gt_map.shape)
    #print(processed_mask.shape)

    return gt_map

# preprocesses the ground truth map
def preprocess_gtmap(instance_cats, gt_mask, gt_map):
    instance_cats = instance_cats[:, 0]
    #instance_cats = np.array([labels_dict[cat] for cat in instance_cats]) # map them to be 1-80
    sorted_indices = np.argsort(instance_cats)
    sorted_instance_cats = instance_cats[sorted_indices]
    unique_cats, reduce_idx = np.unique(sorted_instance_cats, return_index=True)
    sorted_gt_mask = gt_mask[sorted_indices, :, :]
    processed_mask = np.add.reduceat(gt_mask, reduce_idx, axis=0)
    processed_mask = np.minimum(processed_mask, 1) # ensure values are 0 and 1
    for cat_idx, cat in enumerate(unique_cats):
        gt_map[0, :, :, cat] = processed_mask[cat_idx, :, :]
    return gt_map

def _crop(image, offset_height, offset_width, crop_height, crop_width):

    original_shape = tf.shape(image)

    rank_assertion = tf.Assert(
      tf.equal(tf.rank(image), 3),
      ['Rank of image must be equal to 3.'])
    cropped_shape = control_flow_ops.with_dependencies(
      [rank_assertion],
      tf.stack([crop_height, crop_width, original_shape[2]]))

    size_assertion = tf.Assert(
      tf.logical_and(
          tf.greater_equal(original_shape[0], crop_height),
          tf.greater_equal(original_shape[1], crop_width)),
      ['Crop size greater than the image size.'])

    offsets = tf.to_int32(tf.stack([offset_height, offset_width, 0]))

    # Use tf.slice instead of crop_to_bounding box as it accepts tensors to
    # define the crop size.
    image = control_flow_ops.with_dependencies(
      [size_assertion],
      tf.slice(image, offsets, cropped_shape))
    return tf.reshape(image, cropped_shape)

def _central_crop(image, crop_height, crop_width):

    image_height = tf.shape(image)[0]
    image_width = tf.shape(image)[1]

    offset_height = (image_height - crop_height) / 2
    offset_width = (image_width - crop_width) / 2

    output_image = _crop(image, offset_height, offset_width,
                               crop_height, crop_width)
    return output_image

def _random_crop(image, label, crop_height, crop_width):
    
    #image = tf.Print(image, [tf.shape(image)], message = 'Shape of image')
    #label = tf.Print(label, [tf.shape(label)], message = 'Shape of label')

    image_list = [image]
    label_list = [label]

    # Compute the rank assertions.
    rank_assertions = []
    for i in range(len(image_list)):
        image_rank = tf.rank(image_list[i])
        rank_assert = tf.Assert(
            tf.equal(image_rank, 3),
            ['Wrong rank for tensor  %s [expected] [actual]',
             image_list[i].name, 3, image_rank])
        rank_assertions.append(rank_assert)

    image_shape = control_flow_ops.with_dependencies(
        [rank_assertions[0]],
        tf.shape(image_list[0]))

    image_height = image_shape[0]
    image_width = image_shape[1]
    crop_size_assert = tf.Assert(
        tf.logical_and(
            tf.greater_equal(image_height, crop_height),
            tf.greater_equal(image_width, crop_width)),
        ['Crop size greater than the image size.', image_height, image_width, crop_height, crop_width])

    asserts = [rank_assertions[0], crop_size_assert]

    # Create a random bounding box.
    #
    # Use tf.random_uniform and not numpy.random.rand as doing the former would
    # generate random numbers at graph eval time, unlike the latter which
    # generates random numbers at graph definition time.
    max_offset_height = control_flow_ops.with_dependencies(
        asserts, tf.reshape(image_height - crop_height + 1, []))
    max_offset_width = control_flow_ops.with_dependencies(
        asserts, tf.reshape(image_width - crop_width + 1, []))
    offset_height = tf.random_uniform(
        [], maxval=max_offset_height, dtype=tf.int32)
    offset_width = tf.random_uniform(
        [], maxval=max_offset_width, dtype=tf.int32)

    cropped_images = [_crop(image, offset_height, offset_width,
                          crop_height, crop_width) for image in image_list]
    cropped_labels = [_crop(label, offset_height, offset_width,
                          crop_height, crop_width) for label in label_list]
    return cropped_images[0], cropped_labels[0]

# Build data provider for COCO dataset
class COCO(data.TFRecordsParallelByFileProvider):

    def __init__(self,
                 # data_path,
                 # source_dirs=['/mnt/data/mscoco/train_tfrecords'],
                 #key_list,
                 # meta_dicts,
                 group='train',
                 batch_size=1,
                 n_threads=4,
                 image_min_size = 240,
                 crop_height = 224,
                 crop_width = 224,
                 *args,
                 **kwargs):
        self.group = group
        self.batch_size = batch_size
        self.image_min_size = image_min_size
        self.crop_height = crop_height
        self.crop_width = crop_width

        #source_dirs = [data_path['%s/%s' % (self.group, v)] for v in key_list]
        #meta_dicts = [{v : {'dtype': tf.string, 'shape': []}} if v in BYTES_KEYs else {v : {'dtype': tf.int64, 'shape': []}} for v in key_list]
        key_list = ['height', 'images', 'labels', 'num_objects', \
                    'segmentation_masks', 'width', 'bboxes']

        # key_list = ['images']
        source_dirs = ['/mnt/data/mscoco/train_tfrecords/{}/' .format(v) for v in key_list]
        
        BYTES_KEYs = ['images', 'labels', 'segmentation_masks', 'bboxes']

        meta_dicts = [{v : {'dtype': tf.string, 'shape': []}} if v in BYTES_KEYs else {v : {'dtype': tf.int64, 'shape': []}} for v in key_list]

        super(COCO, self).__init__(
            source_dirs = source_dirs,
            meta_dicts = meta_dicts,
            batch_size=batch_size,
            n_threads=n_threads,
            # shuffle = True,
            shuffle=False,
            *args, 
            **kwargs)

    def prep_data(self, data):
        for i in range(len(data)):
            inputs = data[i]

            image = inputs['images']
            image = tf.decode_raw(image, tf.uint8)
            ih = inputs['height']
            iw = inputs['width']
            ih = tf.cast(ih, tf.int32)
            iw = tf.cast(iw, tf.int32)
            inputs['height'] = ih
            inputs['width'] = iw
            
            bboxes = tf.decode_raw(inputs['bboxes'], tf.float64)
            
            imsize = tf.size(image)

            #image = tf.Print(image, [imsize, ih, iw], message = 'Imsize')

            image = tf.cond(tf.equal(imsize, ih * iw), \
                  lambda: tf.image.grayscale_to_rgb(tf.reshape(image, (ih, iw, 1))), \
                  lambda: tf.reshape(image, (ih, iw, 3)))
            
            image_height = ih
            image_width = iw
            num_instances = inputs['num_objects']
            num_instances = tf.cast(num_instances, tf.int32)
            inputs['num_objects'] = num_instances
            
            # labels = tf.decode_raw(inputs['labels'], tf.int32)
            # labels = tf.reshape(labels, [num_instances, 1])
            #labels = tf.Print(labels, [labels], message = 'Labels')
            # inputs['labels'] = labels
            # single_label = labels[0]

            # gt_boxes = tf.reshape(bboxes, [num_instances, 4])
            image = preprocess_for_training(image, self.image_min_size)
            image = _central_crop(image, self.crop_height, self.crop_width)
            # x_shift, y_shift = (iw - self.crop_width)/2, (ih - self.crop_height)/2
            # boxes = gt_boxes - [x_shift, y_shift, x_shift, y_shift]

            # import pdb; pdb.set_trace()
            # bboxes = tf.reshape(bboxes, [num_instances, 4])
            # bboxes.set_shape([num_instances, 4])
            box_vector = tf.reshape(bboxes, [-1])
            zero_pad = tf.zeros([max_objects*4] - tf.shape(box_vector), dtype=box_vector.dtype)
            padded_boxes = tf.concat([box_vector, zero_pad], axis=0)
            padded_boxes = tf.reshape(padded_boxes, [-1, 4])
            # ones = tf.ones([tf.shape(padded_boxes)[0], 1], dtype=padded_boxes.dtype)
            padded_boxes_with_conf = tf.pad(padded_boxes, tf.constant([[0,0],[0,1]]), constant_values=1.0)#tf.concat([padded_boxes, ones], 1)
            data[i] = {'images': image, 'boxes': padded_boxes_with_conf, 'num_objects': num_instances}#, 'multiple_labels': labels}
            # data[i]['mask_coco'].set_shape([self.crop_height, self.crop_width, 1])
            data[i]['images'].set_shape([self.crop_height, self.crop_width, 3])
        
            # data[i] = {
            #     'images': tf.random_normal([224, 224, 3]),
            #     'boxes': bboxes,
            #     'garbage_image': image,
            #     'num_objects': tf.constant(1, dtype=tf.int32),
            # }
            data[i]['ih'] = ih
            data[i]['iw'] = iw
            # data[i]['prints'] = [p1, p2]
            
        return data

    def set_data_shapes_none(self, data):
        for i in range(len(data)):
            for k in data[i]:
                data[i][k] = tf.squeeze(data[i][k], axis=[-1])
        return data

    def init_ops(self):
        self.input_ops = super(COCO, self).init_ops()

        self.input_ops = self.set_data_shapes_none(self.input_ops)
        self.input_ops = self.prep_data(self.input_ops)

        return self.input_ops
