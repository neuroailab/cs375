"""
Provide a dataset_func which builds a tensorflow dataset object for neural data
See below for an example about how to use this function
"""
import tensorflow as tf
from tfutils.imagenet_data import color_normalize

import os, sys
import numpy as np
import pdb
import h5py


class Generator(object):
    """
    Callable generator loading from hdf5
    """
    NUM_IMGS = 5760

    def __init__(
            self, data_path, 
            index_list=None,
            filter_func=None):
        assert os.path.isfile(data_path), "%s does not exist!" % data_path
        self.data_path = data_path

        # index_list specifies image indexes that will be looped over
        # if it's not provided, will loop over all images 
        if not index_list:
            index_list = range(self.NUM_IMGS)
        self.index_list = index_list

        # filter_func is supposed to be a function which receives image index
        # and hdf5 data and then returns True of False
        self.filter_func = filter_func

    def __call__(self):
        with h5py.File(self.data_path, 'r') as hf:
            for im_indx in self.index_list:
                if not self.filter_func or self.filter_func(im_indx, hf):
                    yield hf['images'][im_indx], \
                          hf['image_meta']['category'][im_indx], \
                          hf['image_meta']['object_name'][im_indx], \
                          hf['image_meta']['rotation_xy'][im_indx], \
                          hf['image_meta']['rotation_xz'][im_indx], \
                          hf['image_meta']['rotation_yz'][im_indx], \
                          hf['image_meta']['size'][im_indx], \
                          hf['image_meta']['translation_y'][im_indx], \
                          hf['image_meta']['translation_z'][im_indx], \
                          hf['image_meta']['variation_level'][im_indx]


def dataset_func(
    batch_size,
    crop_size=224,
    **generator_kwargs
    ):
    gen = Generator(**generator_kwargs)
    ds = tf.data.Dataset.from_generator(
            gen, 
            (tf.uint8, tf.string, tf.string, 
                tf.float32, tf.float32, tf.float32,
                tf.float32, tf.float32, tf.float32,
                tf.string
                ), 
            )

    # Change content in dataset to a dict format
    def _tuple_to_dict(*tuple_value):
        dict_value = {
                'image': tuple_value[0],
                'category': tuple_value[1],
                'object_name': tuple_value[2],
                'rotation_xy': tuple_value[3],
                'rotation_xz': tuple_value[4],
                'rotation_yz': tuple_value[5],
                'size': tuple_value[6],
                'translation_y': tuple_value[7],
                'translation_z': tuple_value[8],
                'variation_level': tuple_value[9],
                }
        return dict_value
    ds = ds.map(
            _tuple_to_dict,
            num_parallel_calls=48)

    # Resize the image to 224*224, and color normalize it
    def _resize_normalize_image(value):
        image = value['image']
        image.set_shape([256, 256])
        image = tf.tile(
                tf.expand_dims(image, axis=2),
                [1, 1, 3]
                )
        image = tf.image.resize_bilinear(
                [image], 
                [crop_size, crop_size])[0]
        image.set_shape([crop_size, crop_size, 3])
        image = color_normalize(image)
        value['image'] = image
        return value
    ds = ds.map(
            _resize_normalize_image,
            num_parallel_calls=48)

    # Make the iterator
    ds = ds.apply(
            tf.contrib.data.batch_and_drop_remainder(batch_size))
    value = ds.make_one_shot_iterator().get_next()
    return value


if __name__=="__main__":
    # Example codes about how to use this data provider
    ## Example of filter_func
    def _filter_func(idx, hf):
        return hf['image_meta']['variation_level'][idx] in ['V3', 'V6']

    # here data_path should be the path to your neural data hdf5 file
    data_iter = dataset_func(
            batch_size=64, 
            data_path='/data/chengxuz/class/ventral_neural_data.hdf5',
            filter_func=_filter_func,
            )
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

    sess = tf.Session()
    test_image = sess.run(data_iter)
    print(test_image.keys())
    print(test_image['image'].shape)
    print(test_image['variation_level'])
