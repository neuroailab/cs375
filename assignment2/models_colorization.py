"""
Please implement a standard AlexNet model here as defined in the paper
    https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf

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

def colorization_model(inputs, train=True, norm=True, **kwargs):
    outputs = inputs
    images = inputs['images']

    # Preprocess into Lab colorspace and generate groundtruth labels
    ## Downsample the images with a convolution
    lab_images = rgb_to_lab(images)
    L_images = lab_images[:, :, :, :1]
    ab_images = lab_images[:, 0:-1:4, 0:-1:4, 1:3]
    outputs['L'] = L_images
    outputs['ab'] = ab_images

    gt_Q, _ = ab_to_Q(ab_images)
    outputs['gt_Q'] = gt_Q
    print(gt_Q.get_shape().as_list())

    # conv1
    kernel1 = tf.get_variable("conv1_1/kernel", [3, 3, 1, 64], tf.float32, initializer=tf.contrib.layers.xavier_initializer())
    bias1 = tf.get_variable("conv1_1/bias", [64], tf.float32, initializer=tf.zeros_initializer())

    conv1 = tf.nn.conv2d(L_images, kernel1, [1, 1, 1, 1], padding='SAME')
    conv1 = tf.nn.bias_add(conv1, bias1)
    pred = tf.nn.relu(conv1)

    print(pred.get_shape().as_list())
    outputs['conv1_1'] = pred
    outputs['conv1_1_kernel'] = kernel1 
    pred = tf.layers.conv2d(pred, 64, (3, 3),
                            strides=(2, 2),
                            padding='same',
                            dilation_rate=(1, 1),
                            activation=tf.nn.relu,
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                            name='conv1_2')
    print(pred.get_shape().as_list())
    outputs['conv1_2'] = pred
    pred = tf.layers.batch_normalization(pred, training=train, name='bn1')

    # conv2
    pred = tf.layers.conv2d(pred, 128, (3, 3),
                            strides=(1, 1),
                            padding='same',
                            dilation_rate=(1, 1),
                            activation=tf.nn.relu,
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                            name='conv2_1')
    outputs['conv2_1'] = pred
    print(pred.get_shape().as_list())
    pred = tf.layers.conv2d(pred, 128, (3, 3),
                            strides=(2, 2),
                            padding='same',
                            dilation_rate=(1, 1),
                            activation=tf.nn.relu,
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                            name='conv2_2')
    outputs['conv2_2'] = pred
    print(pred.get_shape().as_list())
    pred = tf.layers.batch_normalization(pred, training=train, name='bn2')

    # conv3
    pred = tf.layers.conv2d(pred, 256, (3, 3),
                            strides=(1, 1),
                            padding='same',
                            dilation_rate=(1, 1),
                            activation=tf.nn.relu,
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                            name='conv3_1')
    outputs['conv3_1'] = pred
    print(pred.get_shape().as_list())
    pred = tf.layers.conv2d(pred, 256, (3, 3),
                            strides=(1, 1),
                            padding='same',
                            dilation_rate=(1, 1),
                            activation=tf.nn.relu,
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                            name='conv3_2')
    outputs['conv3_2'] = pred
    print(pred.get_shape().as_list())
    pred = tf.layers.conv2d(pred, 256, (3, 3),
                            strides=(2, 2),
                            padding='same',
                            dilation_rate=(1, 1),
                            activation=tf.nn.relu,
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                            name='conv3_3')
    outputs['conv3_3'] = pred
    print(pred.get_shape().as_list())
    pred = tf.layers.batch_normalization(pred, training=train, name='bn3')

    # conv4
    pred = tf.layers.conv2d(pred, 512, (3, 3),
                            strides=(1, 1),
                            padding='same',
                            dilation_rate=(1, 1),
                            activation=tf.nn.relu,
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                            name='conv4_1')
    outputs['conv4_1'] = pred
    print(pred.get_shape().as_list())
    pred = tf.layers.conv2d(pred, 512, (3, 3),
                            strides=(1, 1),
                            padding='same',
                            dilation_rate=(1, 1),
                            activation=tf.nn.relu,
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                            name='conv4_2')
    outputs['conv4_2'] = pred
    print(pred.get_shape().as_list())
    pred = tf.layers.conv2d(pred, 512, (3, 3),
                            strides=(1, 1),
                            padding='same',
                            dilation_rate=(1, 1),
                            activation=tf.nn.relu,
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                            name='conv4_3')
    outputs['conv4_3'] = pred
    print(pred.get_shape().as_list())
    pred = tf.layers.batch_normalization(pred, training=train, name='bn4')

    # conv5
    pred = tf.layers.conv2d(pred, 512, (3, 3),
                            strides=(1, 1),
                            padding='same',
                            dilation_rate=(2, 2),
                            activation=tf.nn.relu,
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                            name='conv5_1')
    outputs['conv5_1'] = pred
    print(pred.get_shape().as_list())
    pred = tf.layers.conv2d(pred, 512, (3, 3),
                            strides=(1, 1),
                            padding='same',
                            dilation_rate=(2, 2),
                            activation=tf.nn.relu,
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                            name='conv5_2')
    outputs['conv5_2'] = pred
    print(pred.get_shape().as_list())
    pred = tf.layers.conv2d(pred, 512, (3, 3),
                            strides=(1, 1),
                            padding='same',
                            dilation_rate=(2, 2),
                            activation=tf.nn.relu,
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                            name='conv5_3')
    outputs['conv5_3'] = pred
    print(pred.get_shape().as_list())
    pred = tf.layers.batch_normalization(pred, training=train, name='bn5')

    # conv6
    pred = tf.layers.conv2d(pred, 512, (3, 3),
                            strides=(1, 1),
                            padding='same',
                            dilation_rate=(2, 2),
                            activation=tf.nn.relu,
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                            name='conv6_1')
    outputs['conv6_1'] = pred
    print(pred.get_shape().as_list())
    pred = tf.layers.conv2d(pred, 512, (3, 3),
                            strides=(1, 1),
                            padding='same',
                            dilation_rate=(2, 2),
                            activation=tf.nn.relu,
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                            name='conv6_2')
    outputs['conv6_2'] = pred
    print(pred.get_shape().as_list())
    pred = tf.layers.conv2d(pred, 512, (3, 3),
                            strides=(1, 1),
                            padding='same',
                            dilation_rate=(2, 2),
                            activation=tf.nn.relu,
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                            name='conv6_3')
    outputs['conv6_3'] = pred
    print(pred.get_shape().as_list())
    pred = tf.layers.batch_normalization(pred, training=train, name='bn6')

    # conv7
    pred = tf.layers.conv2d(pred, 512, (3, 3),
                            strides=(1, 1),
                            padding='same',
                            dilation_rate=(1, 1),
                            activation=tf.nn.relu,
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                            name='conv7_1')
    outputs['conv7_1'] = pred
    print(pred.get_shape().as_list())
    pred = tf.layers.conv2d(pred, 512, (3, 3),
                            strides=(1, 1),
                            padding='same',
                            dilation_rate=(1, 1),
                            activation=tf.nn.relu,
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                            name='conv7_2')
    outputs['conv7_2'] = pred
    print(pred.get_shape().as_list())
    pred = tf.layers.conv2d(pred, 512, (3, 3),
                            strides=(1, 1),
                            padding='same',
                            dilation_rate=(1, 1),
                            activation=tf.nn.relu,
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                            name='conv7_3')
    outputs['conv7_3'] = pred
    print(pred.get_shape().as_list())
    pred = tf.layers.batch_normalization(pred, training=train, name='bn7')

    # conv8
    pred = tf.layers.conv2d_transpose(pred, 256, (4, 4),
                                      strides=(2, 2),
                                      padding='same',
                                      activation=tf.nn.relu,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                      name='conv8_1')
    outputs['conv8_1'] = pred
    print(pred.get_shape().as_list())
    pred = tf.layers.conv2d(pred, 256, (3, 3),
                            strides=(1, 1),
                            padding='same',
                            dilation_rate=(1, 1),
                            activation=tf.nn.relu,
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                            name='conv8_2')
    outputs['conv8_2'] = pred
    print(pred.get_shape().as_list())
    pred = tf.layers.conv2d(pred, 256, (3, 3),
                            strides=(1, 1),
                            padding='same',
                            dilation_rate=(1, 1),
                            activation=tf.nn.relu,
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                            name='conv8_3')
    print(pred.get_shape().as_list())
    outputs['conv8_3'] = pred

    # prediction
    pred = tf.layers.conv2d(pred, 313, (1, 1),
                            strides=(1, 1),
                            padding='same',
                            dilation_rate=(1, 1),
                            activation=None,
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                            name='conv8_313')
    outputs['pred'] = pred
    return outputs, {}

def rgb_to_lab(srgb):
    with tf.name_scope("rgb_to_lab"):
        #srgb = check_image(srgb)
        srgb_pixels = tf.reshape(srgb, [-1, 3])

        with tf.name_scope("srgb_to_xyz"):
            linear_mask = tf.cast(srgb_pixels <= 0.04045, dtype=tf.float32)
            exponential_mask = tf.cast(srgb_pixels > 0.04045, dtype=tf.float32)
            rgb_pixels = (srgb_pixels / 12.92 * linear_mask) + (((srgb_pixels + 0.055) / 1.055) ** 2.4) * exponential_mask
            rgb_to_xyz = tf.constant([
                #    X        Y          Z
                [0.412453, 0.212671, 0.019334], # R
                [0.357580, 0.715160, 0.119193], # G
                [0.180423, 0.072169, 0.950227], # B
            ])
            xyz_pixels = tf.matmul(rgb_pixels, rgb_to_xyz)

        # https://en.wikipedia.org/wiki/Lab_color_space#CIELAB-CIEXYZ_conversions
        with tf.name_scope("xyz_to_cielab"):
            # convert to fx = f(X/Xn), fy = f(Y/Yn), fz = f(Z/Zn)

            # normalize for D65 white point
            xyz_normalized_pixels = tf.multiply(xyz_pixels, [1/0.950456, 1.0, 1/1.088754])

            epsilon = 6/29
            linear_mask = tf.cast(xyz_normalized_pixels <= (epsilon**3), dtype=tf.float32)
            exponential_mask = tf.cast(xyz_normalized_pixels > (epsilon**3), dtype=tf.float32)
            fxfyfz_pixels = (xyz_normalized_pixels / (3 * epsilon**2) + 4/29) * linear_mask + (xyz_normalized_pixels ** (1/3)) * exponential_mask

            # convert to lab
            fxfyfz_to_lab = tf.constant([
                #  l       a       b
                [  0.0,  500.0,    0.0], # fx
                [116.0, -500.0,  200.0], # fy
                [  0.0,    0.0, -200.0], # fz
            ])
            lab_pixels = tf.matmul(fxfyfz_pixels, fxfyfz_to_lab) + tf.constant([-16.0, 0.0, 0.0])

        return tf.reshape(lab_pixels, tf.shape(srgb))

def lab_to_rgb(lab):
    with tf.name_scope("lab_to_rgb"):
        #lab = check_image(lab)
        lab_pixels = tf.reshape(lab, [-1, 3])

        # https://en.wikipedia.org/wiki/Lab_color_space#CIELAB-CIEXYZ_conversions
        with tf.name_scope("cielab_to_xyz"):
            # convert to fxfyfz
            lab_to_fxfyfz = tf.constant([
                #   fx      fy        fz
                [1/116.0, 1/116.0,  1/116.0], # l
                [1/500.0,     0.0,      0.0], # a
                [    0.0,     0.0, -1/200.0], # b
            ])
            fxfyfz_pixels = tf.matmul(lab_pixels + tf.constant([16.0, 0.0, 0.0]), lab_to_fxfyfz)

            # convert to xyz
            epsilon = 6/29
            linear_mask = tf.cast(fxfyfz_pixels <= epsilon, dtype=tf.float32)
            exponential_mask = tf.cast(fxfyfz_pixels > epsilon, dtype=tf.float32)
            xyz_pixels = (3 * epsilon**2 * (fxfyfz_pixels - 4/29)) * linear_mask + (fxfyfz_pixels ** 3) * exponential_mask

            # denormalize for D65 white point
            xyz_pixels = tf.multiply(xyz_pixels, [0.950456, 1.0, 1.088754])

        with tf.name_scope("xyz_to_srgb"):
            xyz_to_rgb = tf.constant([
                #     r           g          b
                [ 3.2404542, -0.9692660,  0.0556434], # x
                [-1.5371385,  1.8760108, -0.2040259], # y
                [-0.4985314,  0.0415560,  1.0572252], # z
            ])
            rgb_pixels = tf.matmul(xyz_pixels, xyz_to_rgb)
            # avoid a slightly negative number messing up the conversion
            rgb_pixels = tf.clip_by_value(rgb_pixels, 0.0, 1.0)
            linear_mask = tf.cast(rgb_pixels <= 0.0031308, dtype=tf.float32)
            exponential_mask = tf.cast(rgb_pixels > 0.0031308, dtype=tf.float32)
            srgb_pixels = (rgb_pixels * 12.92 * linear_mask) + ((rgb_pixels ** (1/2.4) * 1.055) - 0.055) * exponential_mask

        return tf.reshape(srgb_pixels, tf.shape(lab))

def ab_to_Q(ab_images):
    pts_in_hull = tf.constant(np.load('pts_in_hull.npy').T, dtype=tf.float32)
    rA = tf.reduce_sum(ab_images**2, axis=-1, keep_dims=True)
    rB = tf.reduce_sum(pts_in_hull**2, axis=0, keep_dims=True)
    distances = rA - 2 * tf.tensordot(ab_images, pts_in_hull, [[3], [0]]) + rB
    return tf.argmin(distances, axis=-1), distances

if __name__ == "__main__":
    # check the output shape of each layer
    X = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))
    data = np.random.rand(2, 224, 224, 3)
    inputs = {}
    inputs['images'] = X
    outputs, _ = colorization_model(inputs)
    """
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        preds, gt = sess.run([outputs['pred'], outputs['gt_Q']], feed_dict={X : data})
        print "Predictions"
        print preds
        print "Ground truth"
        print gt
    """
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    print(total_parameters)
