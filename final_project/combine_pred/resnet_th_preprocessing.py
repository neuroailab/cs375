from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import os, sys
import numpy as np

sys.path.append('../no_tfutils/')
from vgg_preprocessing import preprocess_image, _aspect_preserving_resize, _central_crop

def RandomSizedCrop(
        image, 
        out_height, 
        out_width, 
        seed_random=0
        ):
    shape = tf.shape(image)
    height = tf.cast(shape[0], tf.float32)
    width = tf.cast(shape[1], tf.float32)

    area = height*width
    rnd_area = tf.random_uniform(shape = [1], minval=0.08, maxval=1.0, dtype=tf.float32, seed = seed_random)[0] * area
    asp_ratio = tf.random_uniform(shape = [1], minval=3.0/4, maxval=4.0/3, dtype=tf.float32, seed = seed_random)[0]

    crop_height = tf.sqrt(rnd_area * asp_ratio)
    crop_width = tf.sqrt(rnd_area / asp_ratio)

    div_ratio = tf.maximum(crop_height/height, tf.constant(1, dtype=tf.float32))
    div_ratio = tf.maximum(crop_width/width, div_ratio)

    #image = tf.Print(image, [image], message='In rand')

    crop_height = tf.cast(crop_height/div_ratio, tf.int32)
    crop_width = tf.cast(crop_width/div_ratio, tf.int32)

    #image = tf.Print(image, [crop_height, crop_width, height, width], message='Height, width')
    crop_image = tf.random_crop(image, [crop_height, crop_width, 3])
    #crop_image = tf.Print(crop_image, [crop_image], message='After rand_crop')
    image = tf.image.resize_images(crop_image, [out_height, out_width])
    #crop_image = tf.Print(crop_image, [crop_image], message='After rand_size')

    image.set_shape([out_height, out_width, 3])

    return image

def ColorJitter(image, seed_random=0):
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

    image = _color_jitter_one(image)
    return image

def ColorLighting(image, seed_random=0):
    alphastd = 0.1
    eigval = np.array([0.2175, 0.0188, 0.0045], dtype=np.float32)
    eigvec = np.array([[-0.5675,  0.7192,  0.4009],
		     [-0.5808, -0.0045, -0.8140],
		     [-0.5836, -0.6948,  0.4203]], dtype=np.float32)

    alpha = tf.random_normal([3, 1], mean=0.0, stddev=alphastd, seed=seed_random)
    rgb = alpha * (eigval.reshape([3, 1]) * eigvec)
    image = image + tf.reduce_sum(rgb, axis=0)

    return image

def ColorNormalize(image):
    imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    imagenet_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = (image - imagenet_mean) / imagenet_std

    return image

def preprocessing_train(
        image, 
        out_height, 
        out_width, 
        seed_random=0
        ):
    #image = tf.Print(image, [tf.shape(image)], message='Init')
    #image = tf.Print(image, [image], message='Convert')

    image = RandomSizedCrop(
            image=image, 
            out_height=out_height,
            out_width=out_width,
            seed_random=seed_random,
            )
    #image = tf.Print(image, [image], message='Rand')
    image = ColorJitter(image, seed_random)

    image = tf.image.convert_image_dtype(image, dtype = tf.float32)
    #image = tf.Print(image, [image], message='Jitter')
    image = ColorLighting(image, seed_random)
    #image = tf.Print(image, [image], message='Light')
    image = ColorNormalize(image)
    #image = tf.Print(image, [image], message='Norm')
    image = tf.image.random_flip_left_right(image, seed = seed_random)
    #image = tf.Print(image, [image], message='Flip')

    return image

def preprocessing_val(
        image, 
        out_height, 
        out_width, 
        ):

    image = _aspect_preserving_resize(image, 256)
    image = _central_crop([image], out_height, out_width)[0]
    image.set_shape([out_height, out_width, 3])

    image = tf.image.convert_image_dtype(image, dtype = tf.float32)
    image = ColorNormalize(image)

    return image

def preprocessing_th(image, 
        out_height, 
        out_width, 
        seed_random=0,
        is_training=False,
        ):
    if is_training:
        return preprocessing_train(image, out_height, out_width, seed_random)
    else:
        return preprocessing_val(image, out_height, out_width)
