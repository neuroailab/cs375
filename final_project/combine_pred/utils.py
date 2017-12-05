from __future__ import division, print_function, absolute_import
import os, sys
import numpy as np

import tensorflow as tf
import copy

host = os.uname()[1]

# Pathes for all the datasets
DATA_PATH = {}

# ThreedWorld, not used now
DATA_PATH['threed/train/images'] = '/mnt/fs0/datasets/one_world_dataset/tfdata/images'
DATA_PATH['threed/train/normals'] = '/mnt/fs0/datasets/one_world_dataset/tfdata/normals'
DATA_PATH['threed/val/images'] = '/mnt/fs0/datasets/one_world_dataset/tfvaldata/images'
DATA_PATH['threed/val/normals'] = '/mnt/fs0/datasets/one_world_dataset/tfvaldata/normals'

# Scenenet, set which_scenenet to be 2
DATA_PATH['scenenet/train/images'] = '/mnt/fs1/Dataset/scenenet_combine/photo'
DATA_PATH['scenenet/train/normals'] = '/mnt/fs1/Dataset/scenenet_combine/normal_new'
DATA_PATH['scenenet/train/depths'] = '/mnt/fs1/Dataset/scenenet_combine/depth_new'
#DATA_PATH['scenenet/val/images'] = '/mnt/fs1/Dataset/scenenet_combine_val/photo'
#DATA_PATH['scenenet/val/normals'] = '/mnt/fs1/Dataset/scenenet_combine_val/normal_new'
#DATA_PATH['scenenet/val/depths'] = '/mnt/fs1/Dataset/scenenet_combine_val/depth_new'
DATA_PATH['scenenet/val/images'] = '/mnt/fs1/Dataset/scenenet_combine/photo'
DATA_PATH['scenenet/val/normals'] = '/mnt/fs1/Dataset/scenenet_combine/normal_new'
DATA_PATH['scenenet/val/depths'] = '/mnt/fs1/Dataset/scenenet_combine/depth_new'

DATA_PATH['scenenet_compress/train/images'] = '/mnt/fs0/chengxuz/Data/scenenet_compress/photo'
DATA_PATH['scenenet_compress/train/normals'] = '/mnt/fs0/chengxuz/Data/scenenet_compress/normal_new'
DATA_PATH['scenenet_compress/train/depths'] = '/mnt/fs0/chengxuz/Data/scenenet_compress/depth_new'
DATA_PATH['scenenet_compress/val/images'] = '/mnt/fs0/chengxuz/Data/scenenet_compress_val/photo'
DATA_PATH['scenenet_compress/val/normals'] = '/mnt/fs0/chengxuz/Data/scenenet_compress_val/normal_new'
DATA_PATH['scenenet_compress/val/depths'] = '/mnt/fs0/chengxuz/Data/scenenet_compress_val/depth_new'

#scenenet_new_prefix = '/mnt/fs0/chengxuz/Data'
scenenet_new_prefix = '/mnt/fs1/Dataset/scenenet_all'

if host=='kanefsky':
    scenenet_new_prefix = '/mnt/data3/chengxuz/Dataset'

# Saved in png and use new normal computing method
DATA_PATH['scenenet_new/train/images'] = '%s/scenenet_new/photo' % scenenet_new_prefix
DATA_PATH['scenenet_new/train/normals'] = '%s/scenenet_new/normals' % scenenet_new_prefix
DATA_PATH['scenenet_new/train/depths'] = '%s/scenenet_new/depth' % scenenet_new_prefix
#DATA_PATH['scenenet_new/train/instances'] = '%s/scenenet_new/instance' % scenenet_new_prefix
DATA_PATH['scenenet_new/train/instances'] = '%s/scenenet_new/classes' % scenenet_new_prefix
DATA_PATH['scenenet_new/val/images'] = '%s/scenenet_new_val/photo' % scenenet_new_prefix
DATA_PATH['scenenet_new/val/normals'] = '%s/scenenet_new_val/normals' % scenenet_new_prefix
DATA_PATH['scenenet_new/val/depths'] = '%s/scenenet_new_val/depth' % scenenet_new_prefix
#DATA_PATH['scenenet_new/val/instances'] = '%s/scenenet_new_val/instance' % scenenet_new_prefix
DATA_PATH['scenenet_new/val/instances'] = '%s/scenenet_new_val/classes' % scenenet_new_prefix

# Scannet, not used now
DATA_PATH['scannet/train/images'] = '/mnt/fs0/chengxuz/Data/scannet/tfrecords/image'
DATA_PATH['scannet/train/depths'] = '/mnt/fs0/chengxuz/Data/scannet/tfrecords/depth'
DATA_PATH['scannet/val/images'] = '/mnt/fs0/chengxuz/Data/scannet/tfrecords_val/image'
DATA_PATH['scannet/val/depths'] = '/mnt/fs0/chengxuz/Data/scannet/tfrecords_val/depth'

# Smaller
DATA_PATH['scannet_re/train/images'] = '/mnt/fs0/chengxuz/Data/scannet/tfrecords_re/image'
DATA_PATH['scannet_re/train/depths'] = '/mnt/fs0/chengxuz/Data/scannet/tfrecords_re/depth'
DATA_PATH['scannet_re/val/images'] = '/mnt/fs0/chengxuz/Data/scannet/tfrecords_val_re/image'
DATA_PATH['scannet_re/val/depths'] = '/mnt/fs0/chengxuz/Data/scannet/tfrecords_val_re/depth'

#pbrnet_prefix = '/mnt/fs0/chengxuz/Data/pbrnet_folder'
pbrnet_prefix = '/mnt/fs1/Dataset/pbrnet'
if host=='kanefsky':
    pbrnet_prefix = '/mnt/data3/chengxuz/Dataset/pbrnet'

# Pbrnet
DATA_PATH['pbrnet/train/images'] = '%s/tfrecords/mlt' % pbrnet_prefix
DATA_PATH['pbrnet/train/normals'] = '%s/tfrecords/normal' % pbrnet_prefix
DATA_PATH['pbrnet/train/depths'] = '%s/tfrecords/depth' % pbrnet_prefix
DATA_PATH['pbrnet/train/instances'] = '%s/tfrecords/category' % pbrnet_prefix
DATA_PATH['pbrnet/val/images'] = '%s/tfrecords_val/mlt' % pbrnet_prefix
DATA_PATH['pbrnet/val/normals'] = '%s/tfrecords_val/normal' % pbrnet_prefix
DATA_PATH['pbrnet/val/depths'] = '%s/tfrecords_val/depth' % pbrnet_prefix
DATA_PATH['pbrnet/val/instances'] = '%s/tfrecords_val/category' % pbrnet_prefix

#imagenet_prefix = '/mnt/fs0/datasets'
imagenet_prefix = '/mnt/fs1/Dataset'
if host=='kanefsky':
    imagenet_prefix = '/mnt/data'

# ImageNet
DATA_PATH['imagenet/images'] = '%s/TFRecord_Imagenet_standard/images' % imagenet_prefix
DATA_PATH['imagenet/labels'] = '%s/TFRecord_Imagenet_standard/labels_0' % imagenet_prefix
DATA_PATH['imagenet/image_label'] = '%s/TFRecord_Imagenet_standard/image_label' % imagenet_prefix
DATA_PATH['imagenet/image_label_hdf5'] = '%s/TFRecord_Imagenet_standard/image_label_hdf5' % imagenet_prefix
DATA_PATH['imagenet/image_label_part'] = '%s/TFRecord_Imagenet_standard/image_label_part' % imagenet_prefix
DATA_PATH['imagenet/image_label_full'] = '%s/TFRecord_Imagenet_standard/image_label_full' % imagenet_prefix

imagenet_again_prefix = '/mnt/fs1/Dataset'
DATA_PATH['imagenet/train/images'] = '%s/imagenet_again/tfr_train/image' % imagenet_again_prefix
DATA_PATH['imagenet/train/labels'] = '%s/imagenet_again/tfr_train/label' % imagenet_again_prefix
DATA_PATH['imagenet/val/images'] = '%s/imagenet_again/tfr_val/image' % imagenet_again_prefix
DATA_PATH['imagenet/val/labels'] = '%s/imagenet_again/tfr_val/label' % imagenet_again_prefix

# Coco, use no 0 for new tensorflow
#coco_prefix = '/mnt/fs0/datasets/mscoco'
coco_prefix = '/mnt/fs1/Dataset/mscoco'
if host=='kanefsky':
    coco_prefix = '/mnt/data3/chengxuz/Dataset/coco_dataset'

FOLDERs = { 'train': '%s/train_tfrecords' % coco_prefix,
            'val':  '%s/val_tfrecords' % coco_prefix}
KEY_LIST = ['bboxes', 'height', 'images', 'labels', 'num_objects', \
        'segmentation_masks', 'width']
for key_group in FOLDERs:
    for key_feature in KEY_LIST:
        DATA_PATH[ 'coco/%s/%s' % (key_group, key_feature) ] = os.path.join(FOLDERs[key_group], key_feature)

FOLDERs_no0 = { 'train': '%s/train_tfrecords_no0' % coco_prefix,
            'val':  '%s/val_tfrecords_no0' % coco_prefix}
for key_group in FOLDERs_no0:
    for key_feature in KEY_LIST:
        DATA_PATH[ 'coco_no0/%s/%s' % (key_group, key_feature) ] = os.path.join(FOLDERs_no0[key_group], key_feature)

# Places
#place_prefix = '/mnt/fs0/chengxuz/Data'
place_prefix = '/mnt/fs1/Dataset'
DATA_PATH['place/train/images'] = '%s/places/tfrs_train/image' % place_prefix
DATA_PATH['place/train/labels'] = '%s/places/tfrs_train/label' % place_prefix
DATA_PATH['place/train/images_part'] = '%s/places/tfrs_train/image_part' % place_prefix
DATA_PATH['place/train/labels_part'] = '%s/places/tfrs_train/label_part' % place_prefix
DATA_PATH['place/val/images'] = '%s/places/tfrs_val/image' % place_prefix
DATA_PATH['place/val/labels'] = '%s/places/tfrs_val/label' % place_prefix
DATA_PATH['place/val/images_part'] = '%s/places/tfrs_val/image' % place_prefix
DATA_PATH['place/val/labels_part'] = '%s/places/tfrs_val/label' % place_prefix

# Nyuv2, only for validation
DATA_PATH['nyuv2/val/images'] = '/mnt/fs0/chengxuz/Data/nyuv2/labeled/image'
DATA_PATH['nyuv2/val/depths'] = '/mnt/fs0/chengxuz/Data/nyuv2/labeled/depth'

# Kinetics
kinetics_prefix = '/mnt/fs1/Dataset/kinetics/'
#ki_FOLDERs = { 'train': '%s/train_tfrs' % kinetics_prefix,
#            'val':  '%s/val_tfrs' % kinetics_prefix}
ki_FOLDERs = { 'train': '%s/train_tfrs_5fps' % kinetics_prefix,
            'val':  '%s/val_tfrs_5fps' % kinetics_prefix}
ki_KEY_LIST = ['path', 'label_p']
for key_group in ki_FOLDERs:
    for key_feature in ki_KEY_LIST:
        DATA_PATH[ 'kinetics/%s/%s' % (key_group, key_feature) ] = os.path.join(ki_FOLDERs[key_group], key_feature)


class Trainloop_class:
    def __init__(self, order = ['scene', 'pbr', 'imagenet', 'coco', 'place', 'kinetics']):
        self.order = order
        self.curr_pos = 0

    def add_pos(self):
        self.curr_pos = self.curr_pos + 1
        if self.curr_pos == len(self.order):
            self.curr_pos = 0

    def train_loop(self, sess, train_targets, num_minibatches=1):

        curr_train_targets = train_targets[0]
        while not self.order[self.curr_pos] in curr_train_targets['loss']:
            self.add_pos()

        curr_key_want = self.order[self.curr_pos]

        new_train_targets = {}
        for key, value in curr_train_targets.iteritems():
            if isinstance(value, dict):
                new_train_targets[key] = value[curr_key_want]
            else:
                new_train_targets[key] = value

        # Perform minibatching
        range_len = (int)(num_minibatches)
        for minibatch in range(range_len - 1):
            # Accumulate gradient for each minibatch
            sess.run(new_train_targets['__grads__'])

        # Compute final targets (includes zeroing gradient accumulator variable)

        ret_dict = sess.run(new_train_targets)
        ret_dict['dataset'] = curr_key_want

        self.add_pos()
        return [ret_dict]


def get_val_target(cfg_dataset, nfromd = 0):
    val_target = []
    if cfg_dataset.get('scenenet', 0)==1:
        need_normal = cfg_dataset.get('scene_normal', 1)==1 and nfromd==0
        need_depth = cfg_dataset.get('scene_depth', 1)==1 or (cfg_dataset.get('scene_normal', 1)==1 and nfromd!=0)
        need_instance = cfg_dataset.get('scene_instance', 0)==1
        #val_target.extend(['normal_scenenet', 'depth_scenenet'])
        if need_normal:
            val_target.append('normal_scenenet')
        if need_depth:
            val_target.append('depth_scenenet')
        if need_instance:
            val_target.append('instance_scenenet')

    if cfg_dataset.get('scannet', 0)==1:
        val_target.extend(['depth_scannet'])

    if cfg_dataset.get('pbrnet', 0)==1:
        need_normal = cfg_dataset.get('pbr_normal', 1)==1 and nfromd==0
        need_depth = cfg_dataset.get('pbr_depth', 1)==1 or (cfg_dataset.get('pbr_normal', 1)==1 and nfromd!=0)
        need_instance = cfg_dataset.get('pbr_instance', 0)==1

        #val_target.extend(['normal_pbrnet', 'depth_pbrnet'])
        if need_normal:
            val_target.append('normal_pbrnet')
        if need_depth:
            val_target.append('depth_pbrnet')
        if need_instance:
            val_target.append('instance_pbrnet')

    if cfg_dataset.get('imagenet', 0)==1:
        val_target.extend(['label_imagenet'])

    if cfg_dataset.get('coco', 0)==1:
        val_target.append('mask_coco')

    if cfg_dataset.get('place', 0)==1:
        val_target.extend(['label_place'])

    if cfg_dataset.get('kinetics', 0)==1:
        val_target.extend(['label_kinetics'])

    if cfg_dataset.get('nyuv2', 0)==1:
        val_target.extend(['depth_nyuv2'])

    return val_target

# TODO: use this instead of l2 loss
def loss_ave_invdot(output, label_0, label_1):
    def _process(label):
        label = tf.cast(label, tf.float32)
        label = tf.div(label, tf.constant(255, dtype=tf.float32))
        return label
    output_0 = tf.nn.l2_normalize(output[0], 3)
    labels_0 = tf.nn.l2_normalize(_process(label_0), 3)
    loss_0 = -tf.reduce_sum(tf.multiply(output_0, labels_0)) / np.prod(label_0.get_shape().as_list()) * 3

    output_1 = tf.nn.l2_normalize(output[1], 3)
    labels_1 = tf.nn.l2_normalize(_process(label_1), 3)
    loss_1 = -tf.reduce_sum(tf.multiply(output_1, labels_1)) / np.prod(label_1.get_shape().as_list()) * 3

    loss = tf.add(loss_0, loss_1)
    return loss

def normal_loss(output, label, normalloss = 0):
    def _process(label):
        label = tf.cast(label, tf.float32)
        label = tf.div(label, tf.constant(255, dtype=tf.float32))
        return label

    curr_label = _process(label)
    if normalloss==0:
        curr_loss = tf.nn.l2_loss(output - curr_label) / np.prod(curr_label.get_shape().as_list())
    elif normalloss==1:
        curr_label = tf.nn.l2_normalize(curr_label, 3)
        curr_output = tf.nn.l2_normalize(output, 3)
        curr_loss = -tf.reduce_sum(tf.multiply(curr_label, curr_output)) / np.prod(curr_label.get_shape().as_list()) * 3
    else:
        curr_label = tf.nn.l2_normalize(curr_label*255/128 - 1, 3)
        curr_output = tf.nn.l2_normalize(output, 3)
        curr_loss = -tf.reduce_sum(tf.multiply(curr_label, curr_output)) / np.prod(curr_label.get_shape().as_list()) * 3

    return curr_loss

def l2_loss_withmask(output, label, mask):
    mask = tf.cast(mask, tf.float32)
    #mask = tf.Print(mask, [tf.reduce_sum(mask)], message = 'Resuce sum of mask')
    return tf.nn.l2_loss(tf.multiply(output - label, mask)) / tf.reduce_sum(mask)

def dep_l2_loss_eigen(output, label):
    diff = output - label
    diff_shape = diff.get_shape().as_list()
    loss_0 = tf.nn.l2_loss(diff) / np.prod(diff_shape)
    loss_1 = tf.square(tf.reduce_sum(diff))/(2*np.prod(diff_shape)*np.prod(diff_shape))

    weight_np_x = np.zeros([3, 3, 1, 1])
    weight_np_x[1, 0, 0, 0] = 0.5
    weight_np_x[1, 2, 0, 0] = -0.5

    weight_conv2d_x = tf.constant(weight_np_x, dtype = tf.float32)

    weight_np_y = np.zeros([3, 3, 1, 1])
    weight_np_y[0, 1, 0, 0] = 0.5
    weight_np_y[2, 1, 0, 0] = -0.5

    weight_conv2d_y = tf.constant(weight_np_y, dtype = tf.float32)

    tmp_dx = tf.nn.conv2d(diff, weight_conv2d_x,
                strides=[1, 1, 1, 1],
                padding='SAME')
    tmp_dy = tf.nn.conv2d(diff, weight_conv2d_y,
                strides=[1, 1, 1, 1],
                padding='SAME')

    loss_2 = tf.reduce_sum(tf.add(tf.square(tmp_dx), tf.square(tmp_dy)))/np.prod(tmp_dx.get_shape().as_list())
    final_loss = loss_0 - loss_1 + loss_2

    return final_loss

def dep_loss_berHu(output, label):
    diff = output - label
    diff_shape = diff.get_shape().as_list()
    diff = tf.abs(diff)

    diff_c = 1.0/5*tf.reduce_max(diff)
    curr_mask = tf.less_equal(diff, diff_c)
    tmp_mask_0 = tf.cast(curr_mask, tf.float32)
    tmp_mask_1 = tf.cast(tf.logical_not(curr_mask), tf.float32)

    tmp_l2_loss = (tf.square(tf.multiply(diff, tmp_mask_1)) + tf.square(diff_c))/(2*diff_c)
    tmp_l1_loss = tf.multiply(diff, tmp_mask_0)
    loss = ( tf.reduce_sum(tmp_l1_loss) + tf.reduce_sum(tmp_l2_loss)) / np.prod(diff_shape)

    return loss

def depth_loss(output, label, depthloss = 0, depth_norm = 8000):

    def _process_dep(label):
        label = tf.cast(label, tf.float32)
        label = tf.div(label, tf.constant(depth_norm, dtype=tf.float32))
        return label

    curr_label = _process_dep(label)
    if depthloss==0:
        curr_loss = tf.nn.l2_loss(output - curr_label) / np.prod(curr_label.get_shape().as_list())
    elif depthloss==1: # loss from Eigen, Fergus 2015
        curr_loss = dep_l2_loss_eigen(output, curr_label)
    elif depthloss==2:
        curr_loss = l2_loss_withmask(output, curr_label, tf.not_equal(curr_label, tf.constant(0, tf.float32)))
    elif depthloss==3:
        curr_loss = dep_loss_berHu(output, curr_label)

    return curr_loss

def online_agg(agg_res, res, step):
    if agg_res is None:
        agg_res = {k: [] for k in res}
    for k, v in res.items():
        agg_res[k].append(np.mean(v))
    return agg_res

def get_n_from_d(depths):
    depths = tf.cast(depths, tf.float32)
    weight_np       = np.zeros([3, 3, 1, 3])
    weight_np[1, 0, 0, 0] = 0.5
    weight_np[1, 2, 0, 0] = -0.5
    weight_np[0, 1, 0, 1] = 0.5
    weight_np[2, 1, 0, 1] = -0.5
    weight_conv2d = tf.constant(weight_np, dtype = tf.float32)

    bias_np         = np.zeros([3])
    bias_np[2]      = 1
    bias_tf         = tf.constant(bias_np, dtype = tf.float32)

    tmp_nor         = tf.nn.conv2d(depths, weight_conv2d,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
    normals         = tf.nn.bias_add(tmp_nor, bias_tf)
    normals_n       = tf.nn.l2_normalize(normals, 3)
    #normals_p       = tf.add(tf.multiply(normals_n, tf.constant(0.5)), tf.constant(0.5))
    normals_p       = normals_n*0.5 + 0.5
    normals_u       = tf.cast(normals_p*255, tf.uint8)

    return normals_u

def get_semantic_loss(curr_predict, curr_truth, need_mask = True, mask_range = 40, less_or_large = 0):
    curr_shape = curr_predict.get_shape().as_list()
    curr_predict = tf.reshape(curr_predict, [-1, curr_shape[-1]])

    curr_truth = tf.reshape(curr_truth, [-1])
    curr_truth = tf.cast(curr_truth, tf.int32)

    if need_mask:
        if less_or_large==0:
            truth_mask = tf.less(curr_truth, mask_range)
        else:
            truth_mask = tf.greater(curr_truth, mask_range)

        curr_truth = tf.boolean_mask(curr_truth, truth_mask)
        curr_predict = tf.boolean_mask(curr_predict, truth_mask)

    curr_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = curr_predict, labels = curr_truth)

    return curr_loss

def add_trainable_loss(curr_loss, name_now):
    sigma_now = tf.get_variable(name = name_now, shape = [1], initializer = tf.ones_initializer(), trainable=True)
    sigma_now_sq = sigma_now*sigma_now
    #curr_loss = tf.Print(curr_loss, [curr_loss], message = name_now)
    curr_loss = 1/(2*sigma_now_sq) * curr_loss + tf.log(sigma_now_sq)
    return curr_loss

def get_softmax_loss(curr_label, curr_output, label_norm, multtime):
    if multtime==1:
        curr_label = tf.reshape(curr_label, [-1])
        curr_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = curr_output, labels = curr_label)
        curr_loss = tf.div(curr_loss, tf.constant(label_norm, dtype = tf.float32))
        curr_loss = tf.reduce_mean(curr_loss)
    else:
        list_output = tf.split(curr_output, axis = 1, num_or_size_splits = multtime)
        curr_list_loss = [get_softmax_loss(curr_label, _curr_output, label_norm = label_norm, multtime = 1) for _curr_output in list_output]
        curr_loss = tf.reduce_mean(curr_list_loss)

    return curr_loss

def loss_withcfg(output, *args, **kwargs):
    cfg_dataset = kwargs.get('cfg_dataset', {})
    depth_norm = kwargs.get('depth_norm', 8000)
    label_norm = kwargs.get('label_norm', 20)
    depthloss = kwargs.get('depthloss', 0)
    normalloss = kwargs.get('normalloss', 0)
    ret_dict = kwargs.get('ret_dict', 0)
    nfromd = kwargs.get('nfromd', 0)
    trainable = kwargs.get('trainable', 0)
    multtime = kwargs.get('multtime', 1)
    combine_dict = kwargs.get('combine_dict', 0)
    print_loss = kwargs.get('print_loss', 0)
    extra_feat = kwargs.get('extra_feat', 0)
    
    now_indx = 0
    loss_list = []
    loss_keys = []
    arg_offset = 0
    if cfg_dataset.get('scenenet', 0)==1:
        if ret_dict==1:
            tmp_loss_list = []

        if cfg_dataset.get('scene_normal', 1)==1:
            if nfromd==0:
                curr_loss = normal_loss(output[now_indx], args[now_indx + arg_offset], normalloss = normalloss)
            else:
                curr_loss = normal_loss(output[now_indx], get_n_from_d(args[now_indx + arg_offset]), normalloss = normalloss)
                if cfg_dataset.get('scene_depth', 1)==1:
                    arg_offset = arg_offset - 1

            if trainable==1:
                if normalloss>=1:
                    curr_loss = curr_loss + 1.2
                curr_loss = add_trainable_loss(curr_loss, name_now = 'sigma_scene_normal')
            if ret_dict==1:
                tmp_loss_list.append(curr_loss)
            else:
                loss_list.append(curr_loss)
            #loss_keys.append('scene_normal')
            now_indx = now_indx + 1

        if cfg_dataset.get('scene_depth', 1)==1:
            curr_loss = depth_loss(output[now_indx], args[now_indx + arg_offset], depthloss = depthloss, depth_norm = depth_norm)

            if trainable==1:
                curr_loss = add_trainable_loss(curr_loss, name_now = 'sigma_scene_depth')

            if ret_dict==1:
                tmp_loss_list.append(curr_loss)
            else:
                loss_list.append(curr_loss)
            #loss_keys.append('scene_depth')
            now_indx = now_indx + 1

        if cfg_dataset.get('scene_instance', 0)==1:
            curr_loss = get_semantic_loss(curr_predict = output[now_indx], curr_truth = args[now_indx + arg_offset], need_mask = False)
            curr_loss = tf.reduce_mean(curr_loss)

            if trainable==1:
                curr_loss = add_trainable_loss(curr_loss, name_now = 'sigma_scene_instance')

            if ret_dict==1:
                tmp_loss_list.append(curr_loss)
            else:
                loss_list.append(curr_loss)
            #loss_keys.append('scene_instance')
            now_indx = now_indx + 1
        if ret_dict==1:
            loss_list.append(tf.add_n(tmp_loss_list))
            loss_keys.append('scene')

    if cfg_dataset.get('scannet', 0)==1:
        curr_loss = depth_loss(output[now_indx], args[now_indx + arg_offset], depthloss = 2, depth_norm = depth_norm)

        if trainable==1:
            curr_loss = add_trainable_loss(curr_loss, name_now = 'sigma_scannet')

        loss_list.append(curr_loss)
        loss_keys.append('scannet')
        now_indx = now_indx + 1

    if cfg_dataset.get('pbrnet', 0)==1:
        if ret_dict==1:
            tmp_loss_list = []

        if cfg_dataset.get('pbr_normal', 1)==1:
            if nfromd==0:
                curr_loss = normal_loss(output[now_indx], args[now_indx + arg_offset], normalloss = normalloss)
            else:
                curr_loss = normal_loss(output[now_indx], get_n_from_d(args[now_indx + arg_offset]), normalloss = normalloss)
                if cfg_dataset.get('pbr_depth', 1)==1:
                    arg_offset = arg_offset - 1

            if trainable==1:
                if normalloss>=1:
                    curr_loss = curr_loss + 1.2
                curr_loss = add_trainable_loss(curr_loss, name_now = 'sigma_pbr_normal')

            if ret_dict==1:
                tmp_loss_list.append(curr_loss)
            else:
                loss_list.append(curr_loss)
            #loss_keys.append('pbr_normal')
            now_indx = now_indx + 1

        if cfg_dataset.get('pbr_depth', 1)==1:
            curr_loss = depth_loss(output[now_indx], args[now_indx + arg_offset], depthloss = depthloss, depth_norm = depth_norm)

            if trainable==1:
                curr_loss = add_trainable_loss(curr_loss, name_now = 'sigma_pbr_depth')

            if ret_dict==1:
                tmp_loss_list.append(curr_loss)
            else:
                loss_list.append(curr_loss)
            #loss_keys.append('pbr_depth')
            now_indx = now_indx + 1

        if cfg_dataset.get('pbr_instance', 0)==1:
            curr_loss = get_semantic_loss(curr_predict = output[now_indx], curr_truth = args[now_indx + arg_offset], need_mask = True, mask_range = 40)
            curr_loss = tf.reduce_mean(curr_loss)

            if trainable==1:
                curr_loss = add_trainable_loss(curr_loss, name_now = 'sigma_pbr_instance')

            if ret_dict==1:
                tmp_loss_list.append(curr_loss)
            else:
                loss_list.append(curr_loss)
            #loss_keys.append('pbr_instance')
            now_indx = now_indx + 1

        if ret_dict==1:
            loss_list.append(tf.add_n(tmp_loss_list))
            loss_keys.append('pbr')

    if cfg_dataset.get('imagenet', 0)==1:
        curr_loss = get_softmax_loss(curr_label = args[now_indx + arg_offset], curr_output = output[now_indx], label_norm = label_norm, multtime = multtime)

        if trainable==1:
            curr_loss = add_trainable_loss(curr_loss, name_now = 'sigma_imagenet')

        loss_list.append(curr_loss)
        loss_keys.append('imagenet')
        now_indx = now_indx + 1

        if extra_feat==1:
            # As we don't need normal and depth here, skip
            now_indx = now_indx + 2
            arg_offset = arg_offset - 2

    if cfg_dataset.get('coco', 0)==1:
        curr_loss = get_semantic_loss(curr_predict = output[now_indx], curr_truth = args[now_indx + arg_offset], need_mask = True, mask_range = 0, less_or_large = 1)
        curr_loss = tf.reduce_mean(curr_loss)

        if trainable==1:
            curr_loss = add_trainable_loss(curr_loss, name_now = 'sigma_coco')

        loss_list.append(curr_loss)
        loss_keys.append('coco')
        now_indx = now_indx + 1

    if cfg_dataset.get('place', 0)==1:
        curr_loss = get_softmax_loss(curr_label = args[now_indx + arg_offset], curr_output = output[now_indx], label_norm = label_norm, multtime = multtime)

        if trainable==1:
            curr_loss = add_trainable_loss(curr_loss, name_now = 'sigma_place')

        loss_list.append(curr_loss)
        loss_keys.append('place')
        now_indx = now_indx + 1

        if extra_feat==1:
            # As we don't need normal and depth here, skip
            now_indx = now_indx + 2
            arg_offset = arg_offset - 2

    if cfg_dataset.get('kinetics', 0)==1:
        curr_loss = get_softmax_loss(curr_label = args[now_indx + arg_offset], curr_output = output[now_indx], label_norm = label_norm, multtime = multtime)

        if trainable==1:
            curr_loss = add_trainable_loss(curr_loss, name_now = 'sigma_kinetics')

        loss_list.append(curr_loss)
        loss_keys.append('kinetics')
        now_indx = now_indx + 1

    if cfg_dataset.get('nyuv2', 0)==1:
        curr_loss = depth_loss(output[now_indx], args[now_indx + arg_offset], depthloss = depthloss, depth_norm = depth_norm)

        if trainable==1:
            curr_loss = add_trainable_loss(curr_loss, name_now = 'sigma_nyuv2')

        loss_list.append(curr_loss)
        loss_keys.append('nyuv2')
        now_indx = now_indx + 1

    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    if len(reg_losses)!=0:
        reg_losses = tf.add_n(reg_losses)
        if print_loss==1:
            reg_losses = tf.Print(reg_losses, [tf.add_n(loss_list)], message = 'Real loss')
        loss_list.append(tf.cast(reg_losses, tf.float32))

    if ret_dict==0:
        return tf.add_n(loss_list)
    else:
        final_dict = {key: value for key, value in zip(loss_keys, loss_list)}
        if combine_dict==1:
            cat_list = []
            non_cat_list = []
            for loss_key in final_dict:
                if loss_key in ['place', 'imagenet', 'kinetics']:
                    cat_list.append(final_dict[loss_key])
                elif loss_key in ['scene', 'pbr', 'coco']:
                    non_cat_list.append(final_dict[loss_key])
            
            new_dict = {'category': tf.add_n(cat_list), 'noncategory': tf.add_n(non_cat_list)}
            final_dict = new_dict

        return final_dict

def add_topn_report(curr_label, curr_output, label_norm, top_or_loss, multtime, loss_dict, str_suffix = 'imagenet'):

    if multtime==1:
        curr_label = tf.reshape(curr_label, [-1])
        if top_or_loss==0:
            curr_top1 = tf.nn.in_top_k(curr_output, curr_label, 1)
            curr_top5 = tf.nn.in_top_k(curr_output, curr_label, 5)
            loss_dict['loss_top1_%s' % str_suffix] = curr_top1
            loss_dict['loss_top5_%s' % str_suffix] = curr_top5
        else:
            curr_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = curr_output, labels = curr_label)
            curr_loss = tf.div(curr_loss, tf.constant(label_norm, dtype = tf.float32))
            loss_dict['loss_%s' % str_suffix] = curr_loss
    else:
        list_output = tf.split(curr_output, axis = 1, num_or_size_splits = multtime)
        for _curr_indx, _curr_output in enumerate(list_output):
            loss_dict = add_topn_report(curr_label = curr_label, curr_output = _curr_output, 
                    label_norm = label_norm, top_or_loss = top_or_loss, multtime = 1, 
                    loss_dict = loss_dict, str_suffix = '%i_%s' % (_curr_indx, str_suffix))

    return loss_dict

def rep_loss_withcfg(
        inputs, 
        output, 
        target, 
        cfg_dataset={}, 
        depth_norm=8000, 
        depthloss=0, 
        normalloss=0,
        nfromd=0, 
        label_norm=20, 
        top_or_loss=0, 
        multtime=1,
        extra_feat=0,
    ):

    now_indx = 0
    loss_dict = {}
    arg_offset = 0

    print(target)
    print(len(output))

    if cfg_dataset.get('scenenet', 0)==1:
        if cfg_dataset.get('scene_normal', 1)==1:
            if nfromd==0:
                curr_loss = normal_loss(output[now_indx], inputs[target[now_indx + arg_offset]], normalloss = normalloss)
            else:
                curr_loss = normal_loss(output[now_indx], get_n_from_d(inputs[target[now_indx + arg_offset]]), normalloss = normalloss)
                if cfg_dataset.get('scene_depth', 1)==1:
                    arg_offset = arg_offset - 1

            loss_dict['loss_normal_scenenet'] = curr_loss
            now_indx = now_indx + 1

        if cfg_dataset.get('scene_depth', 1)==1:
            curr_loss = depth_loss(output[now_indx], inputs[target[now_indx + arg_offset]], depthloss = depthloss, depth_norm = depth_norm)
            loss_dict['loss_depth_scenenet'] = curr_loss
            now_indx = now_indx + 1

        if cfg_dataset.get('scene_instance', 0)==1:
            curr_loss = get_semantic_loss(curr_predict = output[now_indx], curr_truth = inputs[target[now_indx + arg_offset]], need_mask = False)
            loss_dict['loss_instance_scenenet'] = tf.reduce_mean(curr_loss)
            now_indx = now_indx + 1

    if cfg_dataset.get('scannet', 0)==1:
        curr_loss = depth_loss(output[now_indx], inputs[target[now_indx + arg_offset]], depthloss = 2, depth_norm = depth_norm)
        loss_dict['loss_depth_scannet'] = curr_loss
        now_indx = now_indx + 1

    if cfg_dataset.get('pbrnet', 0)==1:

        if cfg_dataset.get('pbr_normal', 1)==1:
            if nfromd==0:
                curr_loss = normal_loss(output[now_indx], inputs[target[now_indx + arg_offset]], normalloss = normalloss)
            else:
                curr_loss = normal_loss(output[now_indx], get_n_from_d(inputs[target[now_indx + arg_offset]]), normalloss = normalloss)
                if cfg_dataset.get('pbr_depth', 1)==1:
                    arg_offset = arg_offset - 1
            loss_dict['loss_normal_pbrnet'] = curr_loss
            now_indx = now_indx + 1

        if cfg_dataset.get('pbr_depth', 1)==1:
            curr_loss = depth_loss(output[now_indx], inputs[target[now_indx + arg_offset]], depthloss = depthloss, depth_norm = depth_norm)
            loss_dict['loss_depth_pbrnet'] = curr_loss
            now_indx = now_indx + 1

        if cfg_dataset.get('pbr_instance', 0)==1:
            curr_loss = get_semantic_loss(curr_predict = output[now_indx], curr_truth = inputs[target[now_indx + arg_offset]], need_mask = True, mask_range = 40)

            loss_dict['loss_instance_pbrnet'] = tf.reduce_mean(curr_loss)
            now_indx = now_indx + 1

    if cfg_dataset.get('imagenet', 0)==1:
        loss_dict = add_topn_report(inputs[target[now_indx + arg_offset]], output[now_indx], 
                label_norm, top_or_loss, multtime, loss_dict, str_suffix = 'imagenet')

        now_indx = now_indx + 1

        if extra_feat==1:
            # As we don't need normal and depth here, skip
            now_indx = now_indx + 2
            arg_offset = arg_offset - 2

    if cfg_dataset.get('coco', 0)==1:
        curr_loss = get_semantic_loss(curr_predict = output[now_indx], curr_truth = inputs[target[now_indx + arg_offset]], need_mask = True, mask_range = 0, less_or_large = 1)

        loss_dict['loss_instance_coco'] = tf.reduce_mean(curr_loss)
        now_indx = now_indx + 1

    if cfg_dataset.get('place', 0)==1:
        loss_dict = add_topn_report(inputs[target[now_indx + arg_offset]], output[now_indx], 
                label_norm, top_or_loss, multtime, loss_dict, str_suffix = 'place')

        now_indx = now_indx + 1

        if extra_feat==1:
            # As we don't need normal and depth here, skip
            now_indx = now_indx + 2
            arg_offset = arg_offset - 2

    if cfg_dataset.get('kinetics', 0)==1:
        loss_dict = add_topn_report(inputs[target[now_indx + arg_offset]], output[now_indx], 
                label_norm, top_or_loss, multtime, loss_dict, str_suffix = 'kinetics')

        now_indx = now_indx + 1

    if cfg_dataset.get('nyuv2', 0)==1:
        curr_loss = depth_loss(output[now_indx], inputs[target[now_indx + arg_offset]], depthloss = depthloss, depth_norm = depth_norm)
        loss_dict['loss_depth_nyuv2'] = curr_loss
        now_indx = now_indx + 1

    return loss_dict

def encode_var_filter(curr_tensor):
    curr_name = curr_tensor.name
    if curr_name.startswith('encode') and 'weights' in curr_name:
        return True
    else:
        return False

def report_grad(inputs, outputs, loss_func, loss_func_kwargs, var_filter):
    all_var = tf.trainable_variables()
    all_var = filter(var_filter, all_var)
    var_name_list = [v.name for v in all_var]
    print('Tensors in interest: ', var_name_list)
    loss_dict = loss_func(inputs, outputs, **loss_func_kwargs)
    #print(loss_dict)
    loss_list = loss_dict.values()
    all_gradients = [tf.gradients(curr_loss, all_var) for curr_loss in loss_list]
    ret_dict = {'key_list': [{v: tf.constant(0)} for v in loss_dict.keys()]}
    len_loss = len(loss_list)
    for indx, key_name in enumerate(var_name_list):
        curr_dict = {}
        for which_loss in xrange(len_loss):
            sub_dict = {}
            sub_dict['norm'] = tf.norm(all_gradients[which_loss][indx])
            curr_dict[str(which_loss)] = sub_dict

        for which_loss in xrange(len_loss):
            for which_loss2 in xrange(which_loss+1, len_loss):
                curr_dict[str(which_loss)][str(which_loss2)] = tf.reduce_sum(tf.multiply(all_gradients[which_loss][indx], all_gradients[which_loss2][indx]))/(curr_dict[str(which_loss)]['norm']*curr_dict[str(which_loss2)]['norm'])

        ret_dict[str(indx)] = curr_dict
    #print(len(all_gradients))
    #exit()
    return ret_dict

def concat_output(output, n_gpus):
    if n_gpus>1:
        assert len(output)==n_gpus, 'Output shape is not right'
        curr_output = [tf.concat(values = [output[u][v] for u in xrange(n_gpus)], axis = 0) for v in xrange(len(output[0]))]
    else:
        curr_output = output[0]

    return curr_output

def parallel_rep_loss_withcfg(inputs, output, target, n_gpus = 1, **kwargs):
    return rep_loss_withcfg(inputs, concat_output(output, n_gpus), target, **kwargs)

def save_features(
        inputs, 
        outputs, 
        num_to_save, 
        cfg_dataset={}, 
        depth_norm=8000, 
        target=[], 
        normalloss=0, 
        nfromd=0, 
        depthnormal=0,
        extra_feat=0,
        **loss_params
        ):
    save_dict = {}
    now_indx = 0

    if cfg_dataset.get('scenenet', 0)==1:
        image_scenenet = inputs['image_scenenet'][:num_to_save]
        save_dict['fea_image_scenenet'] = image_scenenet

        if cfg_dataset.get('scene_normal', 1)==1:
            if nfromd==0:
                normal_scenenet = inputs['normal_scenenet'][:num_to_save]
            else:
                normal_all = get_n_from_d(inputs['depth_scenenet'])
                normal_scenenet = normal_all[:num_to_save]

            normal_scenenet_out = outputs[now_indx][:num_to_save]
            if normalloss==0:
                normal_scenenet_out = tf.multiply(normal_scenenet_out, tf.constant(255, dtype=tf.float32))
                normal_scenenet_out = tf.cast(normal_scenenet_out, tf.uint8)
            now_indx = now_indx + 1
            save_dict['fea_normal_scenenet'] = normal_scenenet
            save_dict['out_normal_scenenet'] = normal_scenenet_out

        if cfg_dataset.get('scene_depth', 1)==1:
            depth_scenenet = inputs['depth_scenenet'][:num_to_save]
            depth_scenenet_out = outputs[now_indx][:num_to_save]
            if depthnormal==0:
                depth_scenenet_out = tf.multiply(depth_scenenet_out, tf.constant(depth_norm, dtype=tf.float32))
                depth_scenenet_out = tf.cast(depth_scenenet_out, tf.int32)
            now_indx = now_indx + 1
            save_dict['fea_depth_scenenet'] = depth_scenenet
            save_dict['out_depth_scenenet'] = depth_scenenet_out

        if cfg_dataset.get('scene_instance', 0)==1:
            instance_scenenet = inputs['instance_scenenet'][:num_to_save]
            instance_scenenet_out = outputs[now_indx][:num_to_save]
            instance_scenenet_out = tf.argmax(instance_scenenet_out, axis = 3)
            #print(instance_scenenet_out.get_shape().as_list())
            now_indx = now_indx + 1
            save_dict['fea_instance_scenenet'] = instance_scenenet
            save_dict['out_instance_scenenet'] = instance_scenenet_out

    if cfg_dataset.get('scannet', 0)==1:
        image_scannet = inputs['image_scannet'][:num_to_save]
        depth_scannet = inputs['depth_scannet'][:num_to_save]

        depth_scannet_out = outputs[now_indx][:num_to_save]
        depth_scannet_out = tf.multiply(depth_scannet_out, tf.constant(depth_norm, dtype=tf.float32))
        depth_scannet_out = tf.cast(depth_scannet_out, tf.int32)
        now_indx = now_indx + 1

        save_dict['fea_image_scannet'] = image_scannet
        save_dict['fea_depth_scannet'] = depth_scannet
        save_dict['out_depth_scannet'] = depth_scannet_out

    if cfg_dataset.get('pbrnet', 0)==1:
        image_pbrnet = inputs['image_pbrnet'][:num_to_save]
        save_dict['fea_image_pbrnet'] = image_pbrnet

        if cfg_dataset.get('pbr_normal', 1)==1:
            if nfromd==0:
                normal_pbrnet = inputs['normal_pbrnet'][:num_to_save]
            else:
                normal_all = get_n_from_d(inputs['depth_pbrnet'])
                normal_pbrnet = normal_all[:num_to_save]

            normal_pbrnet_out = outputs[now_indx][:num_to_save]
            if normalloss==0:
                normal_pbrnet_out = tf.multiply(normal_pbrnet_out, tf.constant(255, dtype=tf.float32))
                normal_pbrnet_out = tf.cast(normal_pbrnet_out, tf.uint8)
            now_indx = now_indx + 1
            save_dict['fea_normal_pbrnet'] = normal_pbrnet
            save_dict['out_normal_pbrnet'] = normal_pbrnet_out

        if cfg_dataset.get('pbr_depth', 1)==1:
            depth_pbrnet = inputs['depth_pbrnet'][:num_to_save]
            depth_pbrnet_out = outputs[now_indx][:num_to_save]
            if depthnormal==0:
                depth_pbrnet_out = tf.multiply(depth_pbrnet_out, tf.constant(depth_norm, dtype=tf.float32))
                depth_pbrnet_out = tf.cast(depth_pbrnet_out, tf.int32)
            now_indx = now_indx + 1
            save_dict['fea_depth_pbrnet'] = depth_pbrnet
            save_dict['out_depth_pbrnet'] = depth_pbrnet_out

        if cfg_dataset.get('pbr_instance', 0)==1:
            instance_pbrnet = inputs['instance_pbrnet'][:num_to_save]
            instance_pbrnet_out = outputs[now_indx][:num_to_save]
            instance_pbrnet_out = tf.argmax(instance_pbrnet_out, axis = 3)
            #print(instance_pbrnet_out.get_shape().as_list())
            now_indx = now_indx + 1
            save_dict['fea_instance_pbrnet'] = instance_pbrnet
            save_dict['out_instance_pbrnet'] = instance_pbrnet_out

    if cfg_dataset.get('imagenet', 0)==1:
        now_indx = now_indx + 1

        # If extra_feat, save imagenet images, normals, and depths
        if extra_feat==1:
            image_imagenet = tf.cast(inputs['image_imagenet'][:num_to_save], tf.uint8)

            depth_imagenet_out = outputs[now_indx][:num_to_save]
            now_indx = now_indx + 1

            normal_imagenet_out = outputs[now_indx][:num_to_save]
            now_indx = now_indx + 1

            save_dict['fea_image_imagenet'] = image_imagenet
            save_dict['out_depth_imagenet'] = depth_imagenet_out
            save_dict['out_normal_imagenet'] = normal_imagenet_out


    if cfg_dataset.get('coco', 0)==1:
        image_coco = tf.cast(inputs['image_coco'][:num_to_save], tf.uint8)
        instance_coco = inputs['mask_coco'][:num_to_save]
        instance_coco_out = outputs[now_indx][:num_to_save]
        instance_coco_out = tf.argmax(instance_coco_out, axis = 3)
        #print(instance_pbrnet_out.get_shape().as_list())
        now_indx = now_indx + 1
        save_dict['fea_image_coco'] = image_coco
        save_dict['fea_instance_coco'] = instance_coco
        save_dict['out_instance_coco'] = instance_coco_out

    if cfg_dataset.get('place', 0)==1:
        now_indx = now_indx + 1

        # If extra_feat, save place images, normals, and depths
        if extra_feat==1:
            image_place = tf.cast(inputs['image_place'][:num_to_save], tf.uint8)

            depth_place_out = outputs[now_indx][:num_to_save]
            now_indx = now_indx + 1

            normal_place_out = outputs[now_indx][:num_to_save]
            now_indx = now_indx + 1

            save_dict['fea_image_place'] = image_place
            save_dict['out_depth_place'] = depth_place_out
            save_dict['out_normal_place'] = normal_place_out

    if cfg_dataset.get('nyuv2', 0)==1:
        image_nyuv2 = inputs['image_nyuv2'][:num_to_save]
        depth_nyuv2 = inputs['depth_nyuv2'][:num_to_save]

        depth_nyuv2_out = outputs[now_indx][:num_to_save]
        if depthnormal==0:
            depth_nyuv2_out = tf.multiply(depth_nyuv2_out, tf.constant(depth_norm, dtype=tf.float32))
            depth_nyuv2_out = tf.cast(depth_nyuv2_out, tf.int32)
        now_indx = now_indx + 1

        save_dict['fea_image_nyuv2'] = image_nyuv2
        save_dict['fea_depth_nyuv2'] = depth_nyuv2
        save_dict['out_depth_nyuv2'] = depth_nyuv2_out

    return save_dict

def parallel_save_features(inputs, output, n_gpus = 1, **kwargs):
    return save_features(inputs, concat_output(output, n_gpus), **kwargs)

def mean_losses_keep_rest(step_results):
    retval = {}
    keys = step_results[0].keys()
    for k in keys:
        plucked = [d[k] for d in step_results]
        if isinstance(k, str) and 'loss' in k:
            retval[k] = np.mean(plucked)
        else:
            retval[k] = plucked
    return retval

def postprocess_config(cfg):
    cfg = copy.deepcopy(cfg)
    for k in cfg.get("network_list", ["decode", "encode"]):
        if k in cfg:
            ks = cfg[k].keys()
            for _k in ks:
                if _k.isdigit():
                    cfg[k][int(_k)] = cfg[k].pop(_k)
    return cfg

def preprocess_config(cfg):
    cfg = copy.deepcopy(cfg)
    for k in cfg.get("network_list", ["decode", "encode"]):
        if k in cfg:
            ks = cfg[k].keys()
            for _k in ks:
                #assert isinstance(_k, int), _k
                cfg[k][str(_k)] = cfg[k].pop(_k)
    return cfg

class ParallelClipOptimizer(object):

    def __init__(self, optimizer_class, n_gpus = 1, gpu_offset = 0, clip=True, *optimizer_args, **optimizer_kwargs):
        self._optimizer = optimizer_class(*optimizer_args, **optimizer_kwargs)
        self.clip = clip
        self.gpu_offset = gpu_offset
        self.n_gpus = n_gpus

    def compute_gradients(self, *args, **kwargs):
        gvs = self._optimizer.compute_gradients(*args, **kwargs)
        if self.clip:
            # gradient clipping. Some gradients returned are 'None' because
            # no relation between the variable and loss; so we skip those.
            gvs = [(tf.clip_by_value(grad, -1., 1.), var)
                   for grad, var in gvs if grad is not None]
        return gvs

    def minimize(self, losses, global_step):
        with tf.variable_scope(tf.get_variable_scope()) as vscope:
            grads_and_vars = []
            if not isinstance(losses, list):
                losses = [losses]

            #print(losses)
            assert len(losses)==self.n_gpus, 'Wrong loss number %i, %i!' % (len(losses), self.n_gpus)

            for i, loss in enumerate(losses):
                with tf.device('/gpu:%d' % (i + self.gpu_offset)):
                    with tf.name_scope('gpu_' + str(i)) as gpu_scope:
                        tmp_grads_and_vars = self.compute_gradients(loss)
                        #print(tmp_grads_and_vars)
                        #for tmp_grads, _ in tmp_grads_and_vars:
                        #    print(tmp_grads.get_shape().as_list())
                        #print(len(tmp_grads_and_vars))
                        grads_and_vars.append(tmp_grads_and_vars)

            if len(losses)==1:
                grads_and_vars = self.average_gradients(grads_and_vars)
            else:
                #with tf.device('/cpu:0'):
                grads_and_vars = self.average_gradients(grads_and_vars)

            return self._optimizer.apply_gradients(grads_and_vars,
                                               global_step=global_step)

    def average_gradients(self, all_grads_and_vars):
        average_grads_and_vars = []
        for grads_and_vars in zip(*all_grads_and_vars):
            #print(grads_and_vars)
            grads = []
            for g, _ in grads_and_vars:
                #print(g.get_shape().as_list(), g)
                grads.append(tf.expand_dims(g, axis=0))
            grad = tf.concat(grads, axis=0)
            grad = tf.reduce_mean(grad, axis=0)
            # all variables are the same so we just use the first gpu variables
            var = grads_and_vars[0][1]
            grad_and_var = (grad, var)
            average_grads_and_vars.append(grad_and_var)
        return average_grads_and_vars

def parallel_reduce_mean(losses, **kwargs):
    #print(losses)
    with tf.variable_scope(tf.get_variable_scope()) as vscope:
        for i, loss in enumerate(losses):
            losses[i] = tf.reduce_mean(loss)
        return losses

def parallel_loss_withcfg(outputs, *args, **kwargs):
    #print(kwargs)
    if 'n_gpus' in kwargs:
        n_gpus = kwargs.pop('n_gpus')
    else:
        n_gpus = 1

    if 'gpu_offset' in kwargs:
        gpu_offset = kwargs.pop('gpu_offset')
    else:
        gpu_offset = 0

    #assert n_gpus>1, 'Only one gpu included!'

    with tf.variable_scope(tf.get_variable_scope()) as vscope:
        if n_gpus>1:
            temp_args = [tf.split(v, axis = 0, num_or_size_splits=n_gpus) for v in args]
            list_of_args = [[now_arg[ind] for now_arg in temp_args] for ind in xrange(n_gpus)]
        else:
            list_of_args = [args]
        losses = []
        for i, (curr_args, curr_outputs) in enumerate(zip(list_of_args, outputs)):
            with tf.device('/gpu:%d' % (i + gpu_offset)):
                with tf.name_scope('gpu_' + str(i)) as gpu_scope:
                    losses.append(loss_withcfg(curr_outputs, *curr_args, **kwargs))
                    tf.get_variable_scope().reuse_variables()

        #print('%i number of losses!' % len(losses))
        return losses
