import os
import numpy as np
import tensorflow as tf
import tfutils.data as data

class ImageNetDataProvider(data.TFRecordsParallelByFileProvider):
    """
    Implements the data provider that reads the ImageNet .tfrecords files,
    preprocesses and augments the image data (cropping, random cropping and
    flipping for training) and enqueues the data onto the data
    queue in which the data gets shuffled before being passed on to the model
    """

    N_TRAIN = 1281167
    N_VAL = 50000
    IMAGENET_MEAN = np.load(
            '/datasets/TFRecord_Imagenet_standard/ilsvrc_2012_mean.npy')\
            .swapaxes(0,1).swapaxes(1,2)[:,:,::-1] / 255.0

    def __init__(self,
                 data_path,
                 group='train',
                 crop_size=256,
                 use_mean_subtraction=False,
                 **kwargs
                 ):

        self.crop_size = crop_size
        self.group = group
        self.use_mean_subtraction = use_mean_subtraction

        source_dirs = [os.path.join(data_path, attr)
                for attr in ['images', 'labels_0']]

        postprocess = {'images': [(self.postproc_imgs, (), {})]}

        super(ImageNetDataProvider, self).__init__(
            source_dirs,
            postprocess=postprocess,
            **kwargs
        )

    def postproc_imgs(self, ims):
        """
        Image preprocessing function that performs a random crop and flip for training
        and crop or pad for validation
        """
        dtype = tf.float32
        shape = [self.crop_size, self.crop_size, 3]

        if self.group == 'train':
            ims = tf.map_fn(lambda img: tf.image.convert_image_dtype(
                img, dtype=dtype), ims, dtype=dtype)
            if self.use_mean_subtraction:
                ims = ims - self.IMAGENET_MEAN
            ims = tf.map_fn(lambda img: tf.random_crop(
                img, shape), ims)
            ims = tf.map_fn(lambda img: tf.image.random_flip_left_right(
                img), ims)

        elif self.group == 'val':
            ims = tf.map_fn(lambda img: tf.image.convert_image_dtype(
                img, dtype=dtype), ims, dtype=dtype)
            if self.use_mean_subtraction:
                ims = ims - self.IMAGENET_MEAN
            ims = tf.map_fn(lambda img: tf.image.resize_image_with_crop_or_pad(
                img, shape[0], shape[1]), ims)

        else:
            raise NotImplementedError(
                    'group not valid -  please choose train or val')
        return ims

    def postproc_labels(self, labels):
        """
        Label preprocessing function
        """
        labels = tf.cast(labels, dtype=tf.int64)
        return labels


class CIFAR10DataProvider(data.TFRecordsParallelByFileProvider):
    """
    Implements the data provider that reads the CIFAR10 .tfrecords files,
    preprocesses and augments the image data (cropping, random cropping and 
    flipping for training) and enqueues the data onto the data
    queue in which the data gets shuffled before being passed on to the model
    """

    N_TRAIN = 50000
    N_VAL = 10000

    def __init__(self,
                 data_path,
                 group='train',
                 crop_size=24,
                 **kwargs
                 ):

        self.crop_size = crop_size
        self.group = group
        
        source_dirs = [os.path.join(data_path, attr) 
                for attr in ['images', 'labels']]

        postprocess = {
                'images': [(self.postproc_imgs, (), {})],
                'labels': [(self.postproc_labels, (), {})],
                }

        super(CIFAR10DataProvider, self).__init__(
            source_dirs,
            postprocess=postprocess,
            **kwargs
        )

    def postproc_imgs(self, ims):
        """
        Image preprocessing function that performs a random crop and flip for training
        and crop or pad for validation
        """
        ims = tf.decode_raw(ims, self.meta_dict['images']['rawtype'])
        ims = tf.reshape(ims, [-1] + self.meta_dict['images']['rawshape'])

        dtype = tf.float32
        shape = [self.crop_size, self.crop_size, 3]

        if self.group == 'train':
            ims = tf.map_fn(lambda img: tf.image.convert_image_dtype(
                img, dtype=dtype), ims, dtype=dtype)
            ims = tf.map_fn(lambda img: tf.random_crop(
                img, shape), ims)
            ims = tf.map_fn(lambda img: tf.image.random_flip_left_right(
                img), ims)

        elif self.group == 'val':
            ims = tf.map_fn(lambda img: tf.image.convert_image_dtype(
                img, dtype=dtype), ims, dtype=dtype)
            ims = tf.map_fn(lambda img: tf.image.resize_image_with_crop_or_pad(
                img, shape[0], shape[1]), ims)

        else:
            raise NotImplementedError(
                    'group not valid -  please choose train or val')
        return ims

    def postproc_labels(self, labels):
        """
        Label preprocessing function
        """
        labels = tf.decode_raw(labels, self.meta_dict['labels']['rawtype'])
        labels = tf.reshape(labels, [-1] + self.meta_dict['labels']['rawshape'])
        labels = tf.cast(labels, dtype=tf.int64)
        return labels


class NeuralDataProvider(data.TFRecordsParallelByFileProvider):
    """
    Implements the data provider that reads the neural data .tfrecords files,
    and enqueues the data onto the data queue 
    """

    N_VAL = 5760

    IMAGENET_MEAN = np.load(
            '/datasets/TFRecord_Imagenet_standard/ilsvrc_2012_mean.npy')\
            .swapaxes(0,1).swapaxes(1,2)[:,:,::-1] / 255.0

    ATTRIBUTES = list(np.load('/datasets/neural_data/attributes.npy'))
    ATTRIBUTES.extend([('images', 'float32'), ('it_feats', 'float32')])

    def __init__(self,
                 data_path,
                 crop_size=256,
                 use_mean_subtraction=False,
                 **kwargs
                 ):

        self.crop_size = crop_size
        self.use_mean_subtraction = use_mean_subtraction

        source_dirs = [os.path.join(data_path, attr[0]) 
                for attr in self.ATTRIBUTES]

        postprocess = dict([(attr[0], [
                (self.decode_and_reshape, ([attr[0]]), {}),
            ]) for attr in self.ATTRIBUTES])
        postprocess['images'].insert(1, (self.postproc_imgs, (), {}))      
      
        super(NeuralDataProvider, self).__init__(
                source_dirs,
                postprocess=postprocess,
                **kwargs
                )

    def decode_and_reshape(self, data, attr, *args, **kwargs):
        if self.meta_dict[attr]['rawtype'] != tf.string:
            data = tf.decode_raw(data, self.meta_dict[attr]['rawtype'])
        data = tf.reshape(data, [-1] + self.meta_dict[attr]['rawshape'])
        return data
        
    def postproc_imgs(self, ims):
        """
        Image preprocessing function that resizes the image
        """
        def _postprocess_images(im):
            if self.use_mean_subtraction:
                im = im - self.IMAGENET_MEAN
            im = tf.image.resize_image_with_crop_or_pad(
                    im, self.crop_size, self.crop_size)
            return im
        return tf.map_fn(lambda im: _postprocess_images(im), ims, dtype=tf.float32)
