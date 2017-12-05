from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import sys
import copy

sys.path.append('../normal_pred/')
from normal_encoder_asymmetric_with_bypass import *

def getWhetherResBlock(i, cfg, key_want = "encode"):
    tmp_dict = cfg[key_want][i]
    return 'ResBlock' in tmp_dict

def getResBlockSettings(i, cfg, key_want = "encode"):
    tmp_dict = cfg[key_want][i]
    return tmp_dict['ResBlock']

def getWhetherBn(i, cfg, key_want = "encode"):
    tmp_dict = cfg[key_want][i]
    return 'bn' in tmp_dict

def getWhetherSoftmax(i, cfg, key_want = "encode"):
    tmp_dict = cfg[key_want][i]
    return 'softmax' in tmp_dict

def getWhetherKin(cfg, key_want = "encode"):
    tmp_dict = cfg[key_want]
    return 'kin_act' in tmp_dict

def getKinFrom(cfg, key_want = "encode"):
    tmp_dict = cfg[key_want]
    return tmp_dict['kin_act']

def getKinSplitFrom(cfg, key_want = "encode"):
    tmp_dict = cfg[key_want]
    return tmp_dict['split_para']

def getWhetherFdb(i, cfg, key_want = "encode"):
    tmp_dict = cfg[key_want][i]
    return 'fdb' in tmp_dict

def getFdbFrom(i, cfg, key_want = "encode"):
    tmp_dict = cfg[key_want][i]['fdb']
    return tmp_dict['from']

def getFdbType(i, cfg, key_want = "encode"):
    tmp_dict = cfg[key_want][i]['fdb']
    return tmp_dict['type']

def getDepConvWhetherBn(i, cfg, key_want = "encode"):
    val = False
    tmp_dict = cfg[key_want][i]
    if 'conv' in tmp_dict:
        val = 'bn' in tmp_dict['conv']
    return val 

def getConvOutput(i, cfg, key_want = "encode"):
    tmp_dict = cfg[key_want][i]["conv"]
    return tmp_dict.get("output",0)==1

def getWhetherInitFile(i, cfg, key_want = "encode", layer_type = "conv"):
    tmp_dict = cfg[key_want][i][layer_type]
    return "init_file" in tmp_dict

def getInitFileName(i, cfg, key_want = "encode", layer_type = "conv"):
    tmp_dict = cfg[key_want][i][layer_type]
    init_path = tmp_dict["init_file"]
    if init_path[0]=='$':
        init_path = cfg[init_path[1:]]
    return init_path

def getInitFileArgs(i, cfg, key_want = "encode", layer_type = "conv"):
    tmp_dict = cfg[key_want][i][layer_type]
    init_args = tmp_dict["init_layer_keys"]
    return init_args

def getVarName(cfg, key_want = "encode"):
    tmp_dict = cfg[key_want]
    return tmp_dict.get('var_name', key_want)

def getVarOffset(cfg, key_want = "encode"):
    tmp_dict = cfg[key_want]
    return tmp_dict.get('var_offset', 0)

def getFdbVarName(cfg, key_want = "encode"):
    tmp_dict = cfg[key_want]
    return tmp_dict.get('fdb_var_name', key_want)

def getFdbVarOffset(cfg, key_want = "encode"):
    tmp_dict = cfg[key_want]
    return tmp_dict.get('fdb_var_offset', 0)

def getEncodeConvBn(i, cfg, which_one = 'encode'):
    val = False

    if which_one in cfg and (i in cfg[which_one]):
        if 'conv' in cfg[which_one][i]:
            if 'bn' in cfg[which_one][i]['conv']:
                val = True

    return val

def getPoolPadding(i, cfg, which_one = 'encode'):
    val = 'SAME'

    if which_one in cfg and (i in cfg[which_one]):
        if 'pool' in cfg[which_one][i]:
            if 'padding' in cfg[which_one][i]['pool']:
                val = cfg[which_one][i]['pool']['padding']

    return val

def getConvPadding(i, cfg, which_one = 'encode'):
    val = None

    if which_one in cfg and (i in cfg[which_one]):
        if 'conv' in cfg[which_one][i]:
            if 'padding' in cfg[which_one][i]['conv']:
                val = cfg[which_one][i]['conv']['padding']

    return val

def getConvUpsample(i, cfg, which_one = 'encode'):
    val = None

    if which_one in cfg and (i in cfg[which_one]):
        if 'conv' in cfg[which_one][i]:
            if 'upsample' in cfg[which_one][i]['conv']:
                val = cfg[which_one][i]['conv']['upsample']

    return val

def normal_vgg16_forcombine(inputs, cfg_initial, train=True, seed = None, reuse_flag = None, reuse_batch = None, batch_name = '', **kwargs):
    """The Model definition for normals"""

    cfg = cfg_initial
    if seed==None:
        fseed = getFilterSeed(cfg)
    else:
        fseed = seed

    m = NoramlNetfromConv(seed = fseed, **kwargs)

    encode_nodes = []
    encode_nodes.append(inputs)

    with tf.contrib.framework.arg_scope([m.conv], init='xavier',
                                        stddev=.01, bias=0, activation='relu'):
        encode_depth = getEncodeDepth(cfg)
        print('Encode depth: %d' % encode_depth)

        #with tf.variable_scope('encode_bn0%s' % batch_name, reuse=reuse_batch):
        #    inputs = m.batchnorm_corr(train, inputs = inputs)

        for i in range(1, encode_depth + 1):
            with tf.variable_scope('encode%i' % i, reuse=reuse_flag):
                cfs = getEncodeConvFilterSize(i, cfg)
                nf = getEncodeConvNumFilters(i, cfg)
                cs = getEncodeConvStride(i, encode_depth, cfg)

                if i==1:
                    new_encode_node = m.conv(nf, cfs, cs, padding='VALID', in_layer=inputs)
                else:
                    new_encode_node = m.conv(nf, cfs, cs)

                print('Encode conv %d with size %d stride %d numfilters %d' % (i, cfs, cs, nf))        
                do_pool = getEncodeDoPool(i, cfg)
                if do_pool:
                    pfs = getEncodePoolFilterSize(i, cfg)
                    ps = getEncodePoolStride(i, cfg)
                    pool_type = getEncodePoolType(i, cfg)

                    if pool_type == 'max':
                        pfunc = 'maxpool'
                    elif pool_type == 'avg':
                        pfunc = 'avgpool' 

                    new_encode_node = m.pool(pfs, ps, pfunc=pfunc)
                    print('Encode %s pool %d with size %d stride %d' % (pfunc, i, pfs, ps))
            if getWhetherBn(i, cfg):
                with tf.variable_scope('encode_bn%i%s' % (i, batch_name), reuse=reuse_batch):
                    new_encode_node = m.batchnorm_corr(train)

            encode_nodes.append(new_encode_node)   

        decode_depth = getDecodeDepth(cfg)
        print('Decode depth: %d' % decode_depth)

        for i in range(1, decode_depth + 1):
            with tf.variable_scope('decode%i' % (encode_depth + i), reuse=reuse_flag):

                add_bypass = getDecodeBypass(i, encode_nodes, None, 0, cfg)

                if add_bypass != None:
                    bypass_layer = encode_nodes[add_bypass]

                    decode = m.add_bypass(bypass_layer)

                    print('Decode bypass from %d at %d for shape' % (add_bypass, i), decode.get_shape().as_list())

                do_unpool = getDecodeDoUnPool(i, cfg)
                if do_unpool:
                    unpool_scale = getDecodeUnPoolScale(i, cfg)
                    new_encode_node = m.resize_images_scale(unpool_scale)

                    print('Decode unpool %d with scale %d' % (i, unpool_scale))

                cfs = getEncodeConvFilterSize(i, cfg, which_one = 'decode')
                nf = getEncodeConvNumFilters(i, cfg, which_one = 'decode')
                cs = getEncodeConvStride(i, encode_depth, cfg, which_one = 'decode')

                new_encode_node = m.conv(nf, cfs, cs)

                print('Decode conv %d with size %d stride %d numfilters %d' % (i, cfs, cs, nf))        

    return m

# Function for building subnetwork based on configurations
def build_partnet(
        inputs, 
        cfg_initial, 
        key_want='encode', 
        train=True, 
        seed=None, 
        reuse_flag=None, 
        reuse_batch=None, 
        fdb_reuse_flag=None, 
        batch_name='', 
        all_out_dict={}, 
        init_stddev=.01, 
        ignorebname=0, 
        weight_decay=None, 
        init_type='xavier', 
        cache_filter=0, 
        dict_cache_filter={},
        fix_pretrain = 0,
        **kwargs):

    cfg = cfg_initial
    if seed==None:
        fseed = getFilterSeed(cfg)
    else:
        fseed = seed

    if ignorebname==1:
        batch_name = ''
        reuse_batch = reuse_flag

    #print(cfg[key_want])

    m = NoramlNetfromConv(seed = fseed, **kwargs)

    assert key_want in cfg, "Wrong key %s for network" % key_want

    valid_flag = True
    if inputs==None:
        assert 'input' in cfg[key_want], "No inputs specified for network %s!" % key_want
        input_node = cfg[key_want]['input']
        assert input_node in all_out_dict, "Input nodes not built yet for network %s!" % key_want
        inputs = all_out_dict[input_node]
        valid_flag = False

    if getWhetherKin(cfg_initial, key_want = key_want):

        # Action related for kinetics

        kin_act = getKinFrom(cfg, key_want = key_want)

        # Reshape: put the time dimension to channel directly, assume time dimension is second dimension
        if kin_act=='reshape':
            inputs = tf.transpose(inputs, perm = [0,2,3,4,1])
            curr_shape = inputs.get_shape().as_list()
            inputs = tf.reshape(inputs, [curr_shape[0], curr_shape[1], curr_shape[2], -1])

        # Split: split the time dimension, build shared networks for all splits
        if kin_act=='split':
            split_para = getKinSplitFrom(cfg, key_want = key_want)
            split_inputs = tf.split(inputs, num_or_size_splits = split_para, axis = 1)

            new_cfg = copy.deepcopy(cfg)
            new_cfg[key_want]['kin_act'] = 'reshape'
            add_out_dict = {}
            all_outs = []

            for split_indx, curr_split in enumerate(split_inputs):
                curr_all_out_dict = copy.copy(all_out_dict)
                curr_m, curr_all_out_dict = build_partnet(
                        curr_split, new_cfg, key_want=key_want, train=train, 
                        seed=seed, reuse_flag=reuse_flag or (split_indx > 0), 
                        reuse_batch=reuse_batch or (split_indx > 0), 
                        fdb_reuse_flag=fdb_reuse_flag or (split_indx > 0), 
                        batch_name=batch_name, all_out_dict=curr_all_out_dict, 
                        init_stddev=init_stddev, ignorebname=ignorebname, 
                        weight_decay=weight_decay, cache_filter=cache_filter,
                        dict_cache_filter=dict_cache_filter,
                        **kwargs)
                all_outs.append(curr_m.output)
                for layer_name in curr_all_out_dict:
                    if layer_name in all_out_dict:
                        continue
                    if not layer_name in add_out_dict:
                        add_out_dict[layer_name] = []
                    add_out_dict[layer_name].append(curr_all_out_dict[layer_name])

            for layer_name in add_out_dict:
                all_out_dict[layer_name] = tf.stack(add_out_dict[layer_name], axis = 1)

            curr_m.output = tf.stack(all_outs, axis = 1)

            return curr_m, all_out_dict

    # Set the input
    m.output = inputs

    # General network building
    with tf.contrib.framework.arg_scope([m.conv], init = init_type,
                                        stddev=init_stddev, bias=0, activation='relu'):
        encode_depth = getPartnetDepth(cfg, key_want = key_want)

        # Sometimes we want this network share parameters with different network
        # we can achieve that by setting var_name (variable name) and var_offset (layer offset for sharing)
        var_name = getVarName(cfg, key_want = key_want)
        var_offset = getVarOffset(cfg, key_want = key_want)

        # fdb connections may have different var_name and var_offset
        fdb_var_name = getFdbVarName(cfg, key_want = key_want)
        fdb_var_offset = getFdbVarOffset(cfg, key_want = key_want)

        # Build each layer, as cfg file starts from 1, we also start from 1
        for i in range(1, encode_depth + 1):
            layer_name = "%s_%i" % (key_want, i)

            with tf.variable_scope('%s%i' % (var_name, i + var_offset), reuse=reuse_flag):

                # Build resnet block
                if getWhetherResBlock(i, cfg, key_want = key_want):
                    new_encode_node = m.resblock(
                            conv_settings = getResBlockSettings(i, cfg, key_want = key_want),
                            weight_decay = weight_decay, bias = 0, init = init_type, 
                            stddev = init_stddev, train = True, padding = 'SAME',
                            )

                # add bypass
                add_bypass = getDecodeBypass_light(i, cfg, key_want = key_want)

                if add_bypass != None:
                    for bypass_layer_name in add_bypass:
                        if bypass_layer_name=='_coord':
                            new_encode_node = m.add_coord()
                            #print('Add Coord here!')
                            continue
                             
                        assert bypass_layer_name in all_out_dict, "Node %s not built yet for network %s!" % (bypass_layer_name, key_want)
                        bypass_layer = all_out_dict[bypass_layer_name]
                        new_encode_node = m.add_bypass(bypass_layer)
                        #print('Network %s bypass from %s at %s' % (key_want, bypass_layer_name, layer_name))


                # do convolution
                if getDoConv(i, cfg, which_one = key_want):
                    cfs = getEncodeConvFilterSize(i, cfg, which_one = key_want)
                    nf = getEncodeConvNumFilters(i, cfg, which_one = key_want)
                    cs = getEncodeConvStride(i, encode_depth, cfg, which_one = key_want)
                    cvBn = getEncodeConvBn(i, cfg, which_one = key_want)
                    conv_padding = getConvPadding(i, cfg, which_one = key_want)

                    trans_out_shape = None
                    conv_upsample = getConvUpsample(i, cfg, which_one = key_want)
                    if not conv_upsample is None:
                        trans_out_shape = m.output.get_shape().as_list()
                        trans_out_shape[1] = conv_upsample*trans_out_shape[1]
                        trans_out_shape[2] = conv_upsample*trans_out_shape[2]
                        trans_out_shape[3] = nf

                    padding = 'SAME'
                    activation = 'relu'
                    bias = 0

                    if valid_flag:
                        padding = 'VALID'
                        valid_flag = False
                    else:
                        if getConvOutput(i, cfg, key_want = key_want):
                            activation = None
                            bias = 0

                    if conv_padding!=None:
                        padding = conv_padding

                    init = init_type
                    init_file = None
                    init_layer_keys = None

                    trainable = None

                    #if getWhetherInitFile(i, cfg, key_want = key_want) and (reuse_flag!=True):
                    if getWhetherInitFile(i, cfg, key_want = key_want):
                        init = 'from_file'
                        init_file = getInitFileName(i, cfg, key_want = key_want)
                        init_layer_keys = getInitFileArgs(i, cfg, key_want = key_want)

                        # if cache_filter is 1, will load into a tensor, save it for later reuse
                        if cache_filter==1:
                            init = 'from_cached'
                            filter_cache_str_prefix = '%s_%i' % (var_name, i + var_offset)
                            weight_key = '%s/weight' % filter_cache_str_prefix
                            bias_key = '%s/bias' % filter_cache_str_prefix
                            if not weight_key in dict_cache_filter:
                                params = np.load(init_file)
                                dict_cache_filter[weight_key] = tf.constant(params[init_layer_keys['weight']], dtype = tf.float32)
                                dict_cache_filter[bias_key] = tf.constant(params[init_layer_keys['bias']], dtype = tf.float32)
                                
                            init_layer_keys = {'weight': dict_cache_filter[weight_key], 'bias': dict_cache_filter[bias_key]}
                        else:
                            print('Layer conv %s init from file' % layer_name)

                        if fix_pretrain==1:
                            trainable = False

                    if not getConvDepsep(i, cfg, which_one = key_want):
                        new_encode_node = m.conv(nf, cfs, cs, activation=activation, bias=bias, padding=padding, 
                                weight_decay=weight_decay, init=init, init_file=init_file, whetherBn=cvBn,
                                train=train, init_layer_keys=init_layer_keys, trans_out_shape=trans_out_shape,
                                trainable=trainable,
                                )
                    else:
                        with_bn = getDepConvWhetherBn(i, cfg, key_want = key_want)
                        new_encode_node = m.depthsep_conv(nf, getConvDepmul(i, cfg, which_one = key_want), cfs, cs, 
                                dep_padding=padding, sep_padding=padding, activation = activation, bias = bias,
                                with_bn = with_bn, bn_name = batch_name, reuse_batch = reuse_batch, train = train, 
                                weight_decay = weight_decay,
                                )

                    #print('Network %s conv %s with size %d stride %d numfilters %d' % (key_want, layer_name, cfs, cs, nf))        

                # do unpool
                do_unpool = getDecodeDoUnPool(i, cfg, key_want = key_want)
                if do_unpool:
                    unpool_scale = getDecodeUnPoolScale(i, cfg, key_want = key_want)
                    new_encode_node = m.resize_images_scale(unpool_scale)

                    #print('Network %s unpool %s with scale %d' % (key_want, layer_name, unpool_scale))

                if getDoFc(i, cfg, which_one = key_want):

                    init = 'trunc_norm'
                    init_file = None
                    init_layer_keys = None

                    if getWhetherInitFile(i, cfg, key_want = key_want, layer_type = 'fc'):
                        print('Layer fc %s init from file' % layer_name)
                        init = 'from_file'
                        init_file = getInitFileName(i, cfg, key_want = key_want, layer_type = 'fc')
                        init_layer_keys = getInitFileArgs(i, cfg, key_want = key_want, layer_type = 'fc')

                        if cache_filter==1:
                            init = 'from_cached'
                            filter_cache_str_prefix = '%s_%i' % (var_name, i + var_offset)
                            weight_key = '%s/weight' % filter_cache_str_prefix
                            bias_key = '%s/bias' % filter_cache_str_prefix
                            if not weight_key in dict_cache_filter:
                                params = np.load(init_file)
                                dict_cache_filter[weight_key] = tf.constant(params[init_layer_keys['weight']], dtype = tf.float32)
                                dict_cache_filter[bias_key] = tf.constant(params[init_layer_keys['bias']], dtype = tf.float32)
                                
                            init_layer_keys = {'weight': dict_cache_filter[weight_key], 'bias': dict_cache_filter[bias_key]}

                    if getFcOutput(i, cfg, key_want = key_want):
                        if init == 'trunc_norm':
                            init = init_type
                        new_encode_node = m.fc(getFcNumFilters(i, cfg, key_want = key_want), 
                                               activation=None, dropout=None, bias=0, weight_decay = weight_decay,
                                               init = init, init_file = init_file, init_layer_keys = init_layer_keys)
                    else:
                        new_encode_node = m.fc(getFcNumFilters(i, cfg, key_want = key_want), 
                                               dropout=getFcDropout(i, cfg, train, key_want = key_want), bias=.1, 
                                               weight_decay = weight_decay,
                                               init = init, init_file = init_file, init_layer_keys = init_layer_keys)

                # do pool
                do_pool = getEncodeDoPool(i, cfg, key_want = key_want)
                if do_pool:
                    pfs = getEncodePoolFilterSize(i, cfg, key_want = key_want)
                    ps = getEncodePoolStride(i, cfg, key_want = key_want)
                    pool_type = getEncodePoolType(i, cfg, key_want = key_want)
                    pool_padding = getPoolPadding(i, cfg, which_one = key_want)

                    if pool_type == 'max':
                        pfunc = 'maxpool'
                    elif pool_type == 'avg':
                        pfunc = 'avgpool' 

                    new_encode_node = m.pool(pfs, ps, pfunc=pfunc, padding=pool_padding)
                    #print('Network %s %s pool %s with size %d stride %d' % (key_want, pfunc, layer_name, pfs, ps))

                if getWhetherSoftmax(i, cfg, key_want = key_want):
                    new_encode_node = m.softmax()

            if getWhetherBn(i, cfg, key_want = key_want):
                #with tf.variable_scope('%s_bn%i%s' % (key_want, i, batch_name), reuse=reuse_batch):
                with tf.variable_scope('%s_bn%i%s' % (var_name, i + var_offset, batch_name), reuse=reuse_batch):
                    new_encode_node = m.batchnorm_corr(train)

            if getWhetherFdb(i, cfg, key_want = key_want):
                from_layer = getFdbFrom(i, cfg, key_want = key_want)
                assert from_layer in all_out_dict, "Fdb nodes not built yet for network %s, layer %i!" % (key_want, i)
                with tf.variable_scope('%s_fdb%i' % (fdb_var_name, i + fdb_var_offset), reuse=fdb_reuse_flag):
                    new_encode_node = m.modulate(all_out_dict[from_layer], bias=0, init = 'trunc_norm', stddev=init_stddev,
                                               weight_decay = weight_decay)

            all_out_dict[layer_name] = new_encode_node

    return m, all_out_dict

def combine_normal_tfutils(inputs, center_im = False, **kwargs):
    image_scenenet = tf.cast(inputs['image_scenenet'], tf.float32)
    image_scenenet = tf.div(image_scenenet, tf.constant(255, dtype=tf.float32))
    if center_im:
        image_scenenet  = tf.subtract(image_scenenet, tf.constant(0.5, dtype=tf.float32))

    m_scenenet = normal_vgg16_forcombine(image_scenenet, reuse_flag = None, reuse_batch = None, batch_name = '_scenenet', **kwargs)

    image_pbrnet = tf.cast(inputs['image_pbrnet'], tf.float32)
    image_pbrnet = tf.div(image_pbrnet, tf.constant(255, dtype=tf.float32))
    if center_im:
        image_pbrnet  = tf.subtract(image_pbrnet, tf.constant(0.5, dtype=tf.float32))

    m_pbrnet = normal_vgg16_forcombine(image_pbrnet, reuse_flag = True, reuse_batch = None, batch_name = '_pbrnet', **kwargs)

    return [m_scenenet.output, m_pbrnet.output], m_pbrnet.params

def input_reshape_mult(inputs, categorymulti = 1):
    if categorymulti>1:
        if 'image_place' in inputs:
            old_shape = inputs['image_place'].get_shape().as_list()
            new_shape = [old_shape[0] * old_shape[1]] + old_shape[2:]
            inputs['image_place'] = tf.reshape(inputs['image_place'], new_shape)
            inputs['label_place'] = tf.reshape(inputs['label_place'], [-1])

        if 'image_imagenet' in inputs:
            old_shape = inputs['image_imagenet'].get_shape().as_list()
            new_shape = [old_shape[0] * old_shape[1]] + old_shape[2:]
            inputs['image_imagenet'] = tf.reshape(inputs['image_imagenet'], new_shape)
            inputs['label_imagenet'] = tf.reshape(inputs['label_imagenet'], [-1])

    return inputs

def combine_normal_tfutils_new_half(inputs, center_im = False, categorymulti = 1, cfg_dataset = {}, twonormals = 0, **kwargs):
    all_outputs = []
    encode_reuse = None
    decode_half_reuse = None
    decode_next_reuse = None
    normal_reuse = None
    depth_reuse = None
    ins_decode_reuse = None
    ret_params = None

    inputs = input_reshape_mult(inputs, categorymulti = categorymulti)

    if cfg_dataset.get('scenenet', 0)==1 and 'image_scenenet' in inputs:
        image_scenenet = tf.cast(inputs['image_scenenet'], tf.float32)
        image_scenenet = tf.div(image_scenenet, tf.constant(255, dtype=tf.float32))
        if center_im:
            image_scenenet  = tf.subtract(image_scenenet, tf.constant(0.5, dtype=tf.float32))

        output_nodes = []

        all_out_dict_scenenet = {}
        m_scenenet_encode, all_out_dict_scenenet = build_partnet(image_scenenet, key_want = 'encode', reuse_flag = encode_reuse, reuse_batch = None, batch_name = '_scenenet', all_out_dict = all_out_dict_scenenet, **kwargs)
        encode_reuse = True

        m_scenenet_decode, all_out_dict_scenenet = build_partnet(None, key_want = 'decode_half', reuse_flag = decode_half_reuse, reuse_batch = None, batch_name = '_scenenet', all_out_dict = all_out_dict_scenenet, **kwargs)
        decode_half_reuse = True

        m_scenenet_decode, all_out_dict_scenenet = build_partnet(None, key_want = 'decode_next', reuse_flag = decode_next_reuse, reuse_batch = None, batch_name = '_scenenet', all_out_dict = all_out_dict_scenenet, **kwargs)
        decode_next_reuse = True
        
        if cfg_dataset.get('scene_normal', 1)==1:

            if twonormals==0:
                m_scenenet_normal, all_out_dict_scenenet = build_partnet(None, key_want = 'normal', reuse_flag = normal_reuse, reuse_batch = None, batch_name = '_scenenet', all_out_dict = all_out_dict_scenenet, **kwargs)
                normal_reuse = True
            else:
                m_scenenet_normal, all_out_dict_scenenet = build_partnet(None, key_want = 'normal_s', reuse_flag = None, reuse_batch = None, batch_name = '_scenenet', all_out_dict = all_out_dict_scenenet, **kwargs)

            output_nodes.append(m_scenenet_normal.output)
            ret_params = m_scenenet_normal.params

        if cfg_dataset.get('scene_depth', 1)==1:
            m_scenenet_depth, all_out_dict_scenenet = build_partnet(None, key_want = 'depth', reuse_flag = depth_reuse, reuse_batch = None, batch_name = '_scenenet', all_out_dict = all_out_dict_scenenet, **kwargs)
            depth_reuse = True
            output_nodes.append(m_scenenet_depth.output)
            ret_params = m_scenenet_depth.params

        if cfg_dataset.get('scene_instance', 0)==1:
            m_scenenet_ins_decode, all_out_dict_scenenet = build_partnet(None, key_want = 'ins_decode', reuse_flag = ins_decode_reuse, reuse_batch = None, batch_name = '_scenenet', all_out_dict = all_out_dict_scenenet, **kwargs)
            ins_decode_reuse = True
            m_scenenet_instance, all_out_dict_scenenet = build_partnet(None, key_want = 'scene_instance', reuse_flag = None, reuse_batch = None, batch_name = '_scenenet', all_out_dict = all_out_dict_scenenet, **kwargs)
            output_nodes.append(m_scenenet_instance.output)
            ret_params = m_scenenet_instance.params

        all_outputs.extend(output_nodes)

    if cfg_dataset.get('scannet', 0)==1 and 'image_scannet' in inputs:
        image_scannet = tf.cast(inputs['image_scannet'], tf.float32)
        image_scannet = tf.div(image_scannet, tf.constant(255, dtype=tf.float32))
        if center_im:
            image_scannet  = tf.subtract(image_scannet, tf.constant(0.5, dtype=tf.float32))

        all_out_dict_scannet = {}
        m_scannet_encode, all_out_dict_scannet = build_partnet(image_scannet, key_want = 'encode', reuse_flag = encode_reuse, reuse_batch = None, batch_name = '_scannet', all_out_dict = all_out_dict_scannet, **kwargs)
        encode_reuse = True

        m_scannet_decode, all_out_dict_scannet = build_partnet(None, key_want = 'decode_half', reuse_flag = decode_half_reuse, reuse_batch = None, batch_name = '_scannet', all_out_dict = all_out_dict_scannet, **kwargs)
        decode_half_reuse = True

        m_scannet_decode, all_out_dict_scannet = build_partnet(None, key_want = 'decode_next', reuse_flag = decode_next_reuse, reuse_batch = None, batch_name = '_scannet', all_out_dict = all_out_dict_scannet, **kwargs)
        decode_next_reuse = True
        
        m_scannet_depth, all_out_dict_scannet = build_partnet(None, key_want = 'depth', reuse_flag = depth_reuse, reuse_batch = None, batch_name = '_scannet', all_out_dict = all_out_dict_scannet, **kwargs)
        depth_reuse = True

        all_outputs.extend([m_scannet_depth.output])
        ret_params = m_scannet_depth.params

    if cfg_dataset.get('pbrnet', 0)==1 and 'image_pbrnet' in inputs:
        image_pbrnet = tf.cast(inputs['image_pbrnet'], tf.float32)
        image_pbrnet = tf.div(image_pbrnet, tf.constant(255, dtype=tf.float32))
        if center_im:
            image_pbrnet  = tf.subtract(image_pbrnet, tf.constant(0.5, dtype=tf.float32))

        output_nodes = []

        all_out_dict_pbrnet = {}
        m_pbrnet_encode, all_out_dict_pbrnet = build_partnet(image_pbrnet, key_want = 'encode', reuse_flag = encode_reuse, reuse_batch = None, batch_name = '_pbrnet', all_out_dict = all_out_dict_pbrnet, **kwargs)
        encode_reuse = True

        m_pbrnet_decode, all_out_dict_pbrnet = build_partnet(None, key_want = 'decode_half', reuse_flag = decode_half_reuse, reuse_batch = None, batch_name = '_pbrnet', all_out_dict = all_out_dict_pbrnet, **kwargs)
        decode_half_reuse = True

        m_pbrnet_decode, all_out_dict_pbrnet = build_partnet(None, key_want = 'decode_next', reuse_flag = decode_next_reuse, reuse_batch = None, batch_name = '_pbrnet', all_out_dict = all_out_dict_pbrnet, **kwargs)
        decode_next_reuse = True

        if cfg_dataset.get('pbr_normal', 1)==1:
            if twonormals==0:
                m_pbrnet_normal, all_out_dict_pbrnet = build_partnet(None, key_want = 'normal', reuse_flag = normal_reuse, reuse_batch = None, batch_name = '_pbrnet', all_out_dict = all_out_dict_pbrnet, **kwargs)
                normal_reuse = True
            else:
                m_pbrnet_normal, all_out_dict_pbrnet = build_partnet(None, key_want = 'normal_p', reuse_flag = None, reuse_batch = None, batch_name = '_pbrnet', all_out_dict = all_out_dict_pbrnet, **kwargs)

            output_nodes.append(m_pbrnet_normal.output)
            ret_params = m_pbrnet_normal.params
        
        if cfg_dataset.get('pbr_depth', 1)==1:
            m_pbrnet_depth, all_out_dict_pbrnet = build_partnet(None, key_want = 'depth', reuse_flag = depth_reuse, reuse_batch = None, batch_name = '_pbrnet', all_out_dict = all_out_dict_pbrnet, **kwargs)
            depth_reuse = True
            output_nodes.append(m_pbrnet_depth.output)
            ret_params = m_pbrnet_depth.params

        if cfg_dataset.get('pbr_instance', 0)==1:
            m_pbrnet_ins_decode, all_out_dict_pbrnet = build_partnet(None, key_want = 'ins_decode', reuse_flag = ins_decode_reuse, reuse_batch = None, batch_name = '_pbrnet', all_out_dict = all_out_dict_pbrnet, **kwargs)
            ins_decode_reuse = True
            m_pbrnet_instance, all_out_dict_pbrnet = build_partnet(None, key_want = 'pbr_instance', reuse_flag = None, reuse_batch = None, batch_name = '_pbrnet', all_out_dict = all_out_dict_pbrnet, **kwargs)
            output_nodes.append(m_pbrnet_instance.output)
            ret_params = m_pbrnet_instance.params

        all_outputs.extend(output_nodes)

    if cfg_dataset.get('imagenet', 0)==1 and 'image_imagenet' in inputs:
        image_imagenet = tf.cast(inputs['image_imagenet'], tf.float32)
        image_imagenet = tf.div(image_imagenet, tf.constant(255, dtype=tf.float32))
        if center_im:
            image_imagenet  = tf.subtract(image_imagenet, tf.constant(0.5, dtype=tf.float32))

        all_out_dict_imagenet = {}
        m_imagenet_encode, all_out_dict_imagenet = build_partnet(image_imagenet, key_want = 'encode', reuse_flag = encode_reuse, reuse_batch = None, batch_name = '_imagenet', all_out_dict = all_out_dict_imagenet, **kwargs)
        encode_reuse = True
        
        m_imagenet_category, all_out_dict_imagenet = build_partnet(None, key_want = 'category', reuse_flag = None, reuse_batch = None, batch_name = '_imagenet', all_out_dict = all_out_dict_imagenet, **kwargs)

        all_outputs.extend([m_imagenet_category.output])
        ret_params = m_imagenet_category.params

    if cfg_dataset.get('coco', 0)==1 and 'image_coco' in inputs:

        image_coco = tf.cast(inputs['image_coco'], tf.float32)
        image_coco = tf.div(image_coco, tf.constant(255, dtype=tf.float32))
        if center_im:
            image_coco  = tf.subtract(image_coco, tf.constant(0.5, dtype=tf.float32))

        output_nodes = []

        all_out_dict_coco = {}
        m_coco_encode, all_out_dict_coco = build_partnet(image_coco, key_want = 'encode', reuse_flag = encode_reuse, reuse_batch = None, batch_name = '_coco', all_out_dict = all_out_dict_coco, **kwargs)
        encode_reuse = True

        m_coco_decode, all_out_dict_coco = build_partnet(None, key_want = 'decode_half', reuse_flag = decode_half_reuse, reuse_batch = None, batch_name = '_coco', all_out_dict = all_out_dict_coco, **kwargs)
        decode_half_reuse = True

        m_coco_ins_decode, all_out_dict_coco = build_partnet(None, key_want = 'ins_decode', reuse_flag = ins_decode_reuse, reuse_batch = None, batch_name = '_coco', all_out_dict = all_out_dict_coco, **kwargs)
        ins_decode_reuse = True
        m_coco_instance, all_out_dict_coco = build_partnet(None, key_want = 'coco_instance', reuse_flag = None, reuse_batch = None, batch_name = '_coco', all_out_dict = all_out_dict_coco, **kwargs)
        output_nodes.append(m_coco_instance.output)
        ret_params = m_coco_instance.params

        all_outputs.extend(output_nodes)

    if cfg_dataset.get('place', 0)==1 and 'image_place' in inputs:
        image_place = tf.cast(inputs['image_place'], tf.float32)
        image_place = tf.div(image_place, tf.constant(255, dtype=tf.float32))
        if center_im:
            image_place  = tf.subtract(image_place, tf.constant(0.5, dtype=tf.float32))

        all_out_dict_place = {}
        m_place_encode, all_out_dict_place = build_partnet(image_place, key_want = 'encode', reuse_flag = encode_reuse, reuse_batch = None, batch_name = '_place', all_out_dict = all_out_dict_place, **kwargs)
        encode_reuse = True
        
        m_place_category, all_out_dict_place = build_partnet(None, key_want = 'place_category', reuse_flag = None, reuse_batch = None, batch_name = '_place', all_out_dict = all_out_dict_place, **kwargs)

        all_outputs.extend([m_place_category.output])
        ret_params = m_place_category.params

    if cfg_dataset.get('nyuv2', 0)==1 and 'image_nyuv2' in inputs:
        image_nyuv2 = tf.cast(inputs['image_nyuv2'], tf.float32)
        image_nyuv2 = tf.div(image_nyuv2, tf.constant(255, dtype=tf.float32))
        if center_im:
            image_nyuv2  = tf.subtract(image_nyuv2, tf.constant(0.5, dtype=tf.float32))

        all_out_dict_nyuv2 = {}
        m_nyuv2_encode, all_out_dict_nyuv2 = build_partnet(image_nyuv2, key_want = 'encode', reuse_flag = encode_reuse, reuse_batch = None, batch_name = '_nyuv2', all_out_dict = all_out_dict_nyuv2, **kwargs)
        encode_reuse = True

        m_nyuv2_decode, all_out_dict_nyuv2 = build_partnet(None, key_want = 'decode_half', reuse_flag = decode_half_reuse, reuse_batch = None, batch_name = '_nyuv2', all_out_dict = all_out_dict_nyuv2, **kwargs)
        decode_half_reuse = True

        m_nyuv2_decode, all_out_dict_nyuv2 = build_partnet(None, key_want = 'decode_next', reuse_flag = decode_next_reuse, reuse_batch = None, batch_name = '_nyuv2', all_out_dict = all_out_dict_nyuv2, **kwargs)
        decode_next_reuse = True
        
        m_nyuv2_depth, all_out_dict_nyuv2 = build_partnet(None, key_want = 'depth', reuse_flag = depth_reuse, reuse_batch = None, batch_name = '_nyuv2', all_out_dict = all_out_dict_nyuv2, **kwargs)
        depth_reuse = True

        all_outputs.extend([m_nyuv2_depth.output])
        #ret_params = m_nyuv2_depth.params

    return all_outputs, ret_params

def combine_normal_tfutils_new_f2(inputs, center_im = False, categorymulti = 1, cfg_dataset = {}, twonormals = 0, **kwargs):
    all_outputs = []
    encode_reuse = None
    decode_reuse = None
    normal_reuse = None
    depth_reuse = None
    category_reuse = None
    ins_decode_reuse = None
    decode_encode_reuse = None
    ret_params = None

    inputs = input_reshape_mult(inputs, categorymulti = categorymulti)

    if cfg_dataset.get('scenenet', 0)==1 and 'image_scenenet' in inputs:
        image_scenenet = tf.cast(inputs['image_scenenet'], tf.float32)
        image_scenenet = tf.div(image_scenenet, tf.constant(255, dtype=tf.float32))
        if center_im:
            image_scenenet  = tf.subtract(image_scenenet, tf.constant(0.5, dtype=tf.float32))

        output_nodes = []

        all_out_dict_scenenet = {}
        m_scenenet_encode, all_out_dict_scenenet = build_partnet(image_scenenet, key_want = 'encode', reuse_flag = encode_reuse, reuse_batch = None, batch_name = '_scenenet', all_out_dict = all_out_dict_scenenet, **kwargs)
        encode_reuse = True

        m_scenenet_decode, all_out_dict_scenenet = build_partnet(None, key_want = 'decode', reuse_flag = decode_reuse, reuse_batch = None, batch_name = '_scenenet', all_out_dict = all_out_dict_scenenet, **kwargs)
        decode_reuse = True
        
        if cfg_dataset.get('scene_normal', 1)==1:

            if twonormals==0:
                m_scenenet_normal, all_out_dict_scenenet = build_partnet(None, key_want = 'normal', reuse_flag = normal_reuse, reuse_batch = None, batch_name = '_scenenet', all_out_dict = all_out_dict_scenenet, **kwargs)
                normal_reuse = True
            else:
                m_scenenet_normal, all_out_dict_scenenet = build_partnet(None, key_want = 'normal_s', reuse_flag = None, reuse_batch = None, batch_name = '_scenenet', all_out_dict = all_out_dict_scenenet, **kwargs)

            output_nodes.append(m_scenenet_normal.output)
            ret_params = m_scenenet_normal.params

        if cfg_dataset.get('scene_depth', 1)==1:
            m_scenenet_depth, all_out_dict_scenenet = build_partnet(None, key_want = 'depth', reuse_flag = depth_reuse, reuse_batch = None, batch_name = '_scenenet', all_out_dict = all_out_dict_scenenet, **kwargs)
            depth_reuse = True
            output_nodes.append(m_scenenet_depth.output)
            ret_params = m_scenenet_depth.params

        if cfg_dataset.get('scene_instance', 0)==1:
            m_scenenet_ins_decode, all_out_dict_scenenet = build_partnet(None, key_want = 'ins_decode', reuse_flag = ins_decode_reuse, reuse_batch = None, batch_name = '_scenenet', all_out_dict = all_out_dict_scenenet, **kwargs)
            ins_decode_reuse = True
            m_scenenet_instance, all_out_dict_scenenet = build_partnet(None, key_want = 'scene_instance', reuse_flag = None, reuse_batch = None, batch_name = '_scenenet', all_out_dict = all_out_dict_scenenet, **kwargs)
            output_nodes.append(m_scenenet_instance.output)
            ret_params = m_scenenet_instance.params

        all_outputs.extend(output_nodes)

    if cfg_dataset.get('scannet', 0)==1 and 'image_scannet' in inputs:
        image_scannet = tf.cast(inputs['image_scannet'], tf.float32)
        image_scannet = tf.div(image_scannet, tf.constant(255, dtype=tf.float32))
        if center_im:
            image_scannet  = tf.subtract(image_scannet, tf.constant(0.5, dtype=tf.float32))

        all_out_dict_scannet = {}
        m_scannet_encode, all_out_dict_scannet = build_partnet(image_scannet, key_want = 'encode', reuse_flag = encode_reuse, reuse_batch = None, batch_name = '_scannet', all_out_dict = all_out_dict_scannet, **kwargs)
        encode_reuse = True

        m_scannet_decode, all_out_dict_scannet = build_partnet(None, key_want = 'decode', reuse_flag = decode_reuse, reuse_batch = None, batch_name = '_scannet', all_out_dict = all_out_dict_scannet, **kwargs)
        decode_reuse = True
        
        m_scannet_depth, all_out_dict_scannet = build_partnet(None, key_want = 'depth', reuse_flag = depth_reuse, reuse_batch = None, batch_name = '_scannet', all_out_dict = all_out_dict_scannet, **kwargs)
        depth_reuse = True

        all_outputs.extend([m_scannet_depth.output])
        ret_params = m_scannet_depth.params

    if cfg_dataset.get('pbrnet', 0)==1 and 'image_pbrnet' in inputs:
        image_pbrnet = tf.cast(inputs['image_pbrnet'], tf.float32)
        image_pbrnet = tf.div(image_pbrnet, tf.constant(255, dtype=tf.float32))
        if center_im:
            image_pbrnet  = tf.subtract(image_pbrnet, tf.constant(0.5, dtype=tf.float32))

        output_nodes = []

        all_out_dict_pbrnet = {}
        m_pbrnet_encode, all_out_dict_pbrnet = build_partnet(image_pbrnet, key_want = 'encode', reuse_flag = encode_reuse, reuse_batch = None, batch_name = '_pbrnet', all_out_dict = all_out_dict_pbrnet, **kwargs)
        encode_reuse = True

        m_pbrnet_decode, all_out_dict_pbrnet = build_partnet(None, key_want = 'decode', reuse_flag = decode_reuse, reuse_batch = None, batch_name = '_pbrnet', all_out_dict = all_out_dict_pbrnet, **kwargs)
        decode_reuse = True

        if cfg_dataset.get('pbr_normal', 1)==1:
            if twonormals==0:
                m_pbrnet_normal, all_out_dict_pbrnet = build_partnet(None, key_want = 'normal', reuse_flag = normal_reuse, reuse_batch = None, batch_name = '_pbrnet', all_out_dict = all_out_dict_pbrnet, **kwargs)
                normal_reuse = True
            else:
                m_pbrnet_normal, all_out_dict_pbrnet = build_partnet(None, key_want = 'normal_p', reuse_flag = None, reuse_batch = None, batch_name = '_pbrnet', all_out_dict = all_out_dict_pbrnet, **kwargs)

            output_nodes.append(m_pbrnet_normal.output)
            ret_params = m_pbrnet_normal.params
        
        if cfg_dataset.get('pbr_depth', 1)==1:
            m_pbrnet_depth, all_out_dict_pbrnet = build_partnet(None, key_want = 'depth', reuse_flag = depth_reuse, reuse_batch = None, batch_name = '_pbrnet', all_out_dict = all_out_dict_pbrnet, **kwargs)
            depth_reuse = True
            output_nodes.append(m_pbrnet_depth.output)
            ret_params = m_pbrnet_depth.params

        if cfg_dataset.get('pbr_instance', 0)==1:
            m_pbrnet_ins_decode, all_out_dict_pbrnet = build_partnet(None, key_want = 'ins_decode', reuse_flag = ins_decode_reuse, reuse_batch = None, batch_name = '_pbrnet', all_out_dict = all_out_dict_pbrnet, **kwargs)
            ins_decode_reuse = True
            m_pbrnet_instance, all_out_dict_pbrnet = build_partnet(None, key_want = 'pbr_instance', reuse_flag = None, reuse_batch = None, batch_name = '_pbrnet', all_out_dict = all_out_dict_pbrnet, **kwargs)
            output_nodes.append(m_pbrnet_instance.output)
            ret_params = m_pbrnet_instance.params

        all_outputs.extend(output_nodes)

    if cfg_dataset.get('imagenet', 0)==1 and 'image_imagenet' in inputs:
        image_imagenet = tf.cast(inputs['image_imagenet'], tf.float32)
        image_imagenet = tf.div(image_imagenet, tf.constant(255, dtype=tf.float32))
        if center_im:
            image_imagenet  = tf.subtract(image_imagenet, tf.constant(0.5, dtype=tf.float32))

        all_out_dict_imagenet = {}
        m_imagenet_encode, all_out_dict_imagenet = build_partnet(image_imagenet, key_want = 'encode', reuse_flag = encode_reuse, reuse_batch = None, batch_name = '_imagenet', all_out_dict = all_out_dict_imagenet, **kwargs)
        encode_reuse = True

        m_imagenet_decode, all_out_dict_imagenet = build_partnet(None, key_want = 'decode', reuse_flag = decode_reuse, reuse_batch = None, batch_name = '_imagenet', all_out_dict = all_out_dict_imagenet, **kwargs)
        decode_reuse = True

        m_imagenet_ins_decode, all_out_dict_imagenet = build_partnet(None, key_want = 'ins_decode', reuse_flag = ins_decode_reuse, reuse_batch = None, batch_name = '_imagenet', all_out_dict = all_out_dict_imagenet, **kwargs)
        ins_decode_reuse = True

        m_imagenet_decode, all_out_dict_imagenet = build_partnet(None, key_want = 'decode_encode', reuse_flag = decode_encode_reuse, reuse_batch = None, batch_name = '_imagenet', all_out_dict = all_out_dict_imagenet, **kwargs)
        decode_encode_reuse = True
        
        m_imagenet_category, all_out_dict_imagenet = build_partnet(None, key_want = 'category', reuse_flag = category_reuse, reuse_batch = None, batch_name = '_imagenet', all_out_dict = all_out_dict_imagenet, **kwargs)
        category_reuse = True

        all_outputs.extend([m_imagenet_category.output])
        ret_params = m_imagenet_category.params

    if cfg_dataset.get('coco', 0)==1 and 'image_coco' in inputs:

        image_coco = tf.cast(inputs['image_coco'], tf.float32)
        image_coco = tf.div(image_coco, tf.constant(255, dtype=tf.float32))
        if center_im:
            image_coco  = tf.subtract(image_coco, tf.constant(0.5, dtype=tf.float32))

        output_nodes = []

        all_out_dict_coco = {}
        m_coco_encode, all_out_dict_coco = build_partnet(image_coco, key_want = 'encode', reuse_flag = encode_reuse, reuse_batch = None, batch_name = '_coco', all_out_dict = all_out_dict_coco, **kwargs)
        encode_reuse = True

        m_coco_ins_decode, all_out_dict_coco = build_partnet(None, key_want = 'ins_decode', reuse_flag = ins_decode_reuse, reuse_batch = None, batch_name = '_coco', all_out_dict = all_out_dict_coco, **kwargs)
        ins_decode_reuse = True
        m_coco_instance, all_out_dict_coco = build_partnet(None, key_want = 'coco_instance', reuse_flag = None, reuse_batch = None, batch_name = '_coco', all_out_dict = all_out_dict_coco, **kwargs)
        output_nodes.append(m_coco_instance.output)
        ret_params = m_coco_instance.params

        all_outputs.extend(output_nodes)

    if cfg_dataset.get('place', 0)==1 and 'image_place' in inputs:
        image_place = tf.cast(inputs['image_place'], tf.float32)
        image_place = tf.div(image_place, tf.constant(255, dtype=tf.float32))
        if center_im:
            image_place  = tf.subtract(image_place, tf.constant(0.5, dtype=tf.float32))

        all_out_dict_place = {}
        m_place_encode, all_out_dict_place = build_partnet(image_place, key_want = 'encode', reuse_flag = encode_reuse, reuse_batch = None, batch_name = '_place', all_out_dict = all_out_dict_place, **kwargs)
        encode_reuse = True

        m_place_decode, all_out_dict_place = build_partnet(None, key_want = 'decode', reuse_flag = decode_reuse, reuse_batch = None, batch_name = '_place', all_out_dict = all_out_dict_place, **kwargs)
        decode_reuse = True

        m_place_ins_decode, all_out_dict_place = build_partnet(None, key_want = 'ins_decode', reuse_flag = ins_decode_reuse, reuse_batch = None, batch_name = '_place', all_out_dict = all_out_dict_place, **kwargs)
        ins_decode_reuse = True

        m_place_decode, all_out_dict_place = build_partnet(None, key_want = 'decode_encode', reuse_flag = decode_encode_reuse, reuse_batch = None, batch_name = '_place', all_out_dict = all_out_dict_place, **kwargs)
        decode_encode_reuse = True
        
        m_place_category, all_out_dict_place = build_partnet(None, key_want = 'place_category', reuse_flag = None, reuse_batch = None, batch_name = '_place', all_out_dict = all_out_dict_place, **kwargs)

        all_outputs.extend([m_place_category.output])
        ret_params = m_place_category.params

    if cfg_dataset.get('nyuv2', 0)==1 and 'image_nyuv2' in inputs:
        image_nyuv2 = tf.cast(inputs['image_nyuv2'], tf.float32)
        image_nyuv2 = tf.div(image_nyuv2, tf.constant(255, dtype=tf.float32))
        if center_im:
            image_nyuv2  = tf.subtract(image_nyuv2, tf.constant(0.5, dtype=tf.float32))

        all_out_dict_nyuv2 = {}
        m_nyuv2_encode, all_out_dict_nyuv2 = build_partnet(image_nyuv2, key_want = 'encode', reuse_flag = encode_reuse, reuse_batch = None, batch_name = '_nyuv2', all_out_dict = all_out_dict_nyuv2, **kwargs)
        encode_reuse = True

        m_nyuv2_decode, all_out_dict_nyuv2 = build_partnet(None, key_want = 'decode', reuse_flag = decode_reuse, reuse_batch = None, batch_name = '_nyuv2', all_out_dict = all_out_dict_nyuv2, **kwargs)
        decode_reuse = True
        
        m_nyuv2_depth, all_out_dict_nyuv2 = build_partnet(None, key_want = 'depth', reuse_flag = depth_reuse, reuse_batch = None, batch_name = '_nyuv2', all_out_dict = all_out_dict_nyuv2, **kwargs)
        depth_reuse = True

        all_outputs.extend([m_nyuv2_depth.output])
        #ret_params = m_nyuv2_depth.params

    return all_outputs, ret_params

def build_datasetnet(
        inputs, 
        cfg_initial, 
        dataset_prefix, 
        all_outputs=[], 
        reuse_dict={}, 
        center_im=False, 
        cfg_dataset={}, 
        no_prep=0, 
        cache_filter=0, 
        extra_feat=0, 
        **kwargs):
    ret_params = None

    now_input_name = 'image_%s' % dataset_prefix

    if cfg_dataset.get(dataset_prefix, 0)==1 and now_input_name in inputs:

        image_dataset = tf.cast(inputs[now_input_name], tf.float32)
        if no_prep==0:
            image_dataset = tf.div(image_dataset, tf.constant(255, dtype=tf.float32))
            if center_im:
                image_dataset  = tf.subtract(image_dataset, tf.constant(0.5, dtype=tf.float32))

        output_nodes = []
        all_out_dict_dataset = {}
        dict_cache_filter = {}

        curr_order = '%s_order' % dataset_prefix
        assert curr_order in cfg_initial
        network_order = cfg_initial.get(curr_order)

        if extra_feat==1 and dataset_prefix in ['imagenet', 'place']:
            # If extra_feat is 1, then depth and normal branch will be added to imagenet and place dataset as outputs
            # Remember to skip them during calculating loss and calculating rep_loss!

            add_branch_list = ['depth', 'normal']
            
            # Check whether needed information is there
            for curr_add_branch in add_branch_list:
                assert '%s_order' % curr_add_branch in cfg_initial, 'Model cfg should include %s branch info!' % curr_add_branch

            # Work on adding the branches into network order
            for curr_add_branch in add_branch_list:
                add_network_order = cfg_initial.get('%s_order' % curr_add_branch)
                for add_network in add_network_order:
                    if add_network not in network_order:
                        network_order.append(add_network)

        first_flag = True

        for network_name in network_order:
            if first_flag:
                input_now = image_dataset
                first_flag = False
            else:
                input_now = None

            var_name = getVarName(cfg_initial, key_want = network_name)
            reuse_name = '%s_reuse' % var_name
            reuse_curr = reuse_dict.get(reuse_name, None)

            fdb_var_name = getFdbVarName(cfg_initial, key_want = network_name)
            fdb_reuse_name = '_fdb_%s_reuse' % fdb_var_name
            fdb_reuse_curr = reuse_dict.get(fdb_reuse_name, None)

            m_curr, all_out_dict_dataset = build_partnet(
                    input_now, cfg_initial=cfg_initial, key_want=network_name, reuse_flag=reuse_curr, 
                    fdb_reuse_flag=fdb_reuse_curr, reuse_batch=None, batch_name='_%s' % network_name, 
                    all_out_dict=all_out_dict_dataset, cache_filter=cache_filter, 
                    dict_cache_filter=dict_cache_filter,
                    **kwargs)

            reuse_dict[reuse_name] = True
            reuse_dict[fdb_reuse_name] = True
            as_output = cfg_initial.get(network_name).get('as_output', 0)
            if as_output==1:
                output_nodes.append(m_curr.output)
                ret_params = m_curr.params

        all_outputs.extend(output_nodes)

    return all_outputs, reuse_dict, ret_params

def combine_tfutils_general(inputs, categorymulti = 1, **kwargs):

    inputs = input_reshape_mult(inputs, categorymulti = categorymulti)

    all_outputs = []
    reuse_dict = {}
    ret_params_final = None

    #dataset_prefix_list = ['scenenet', 'pbrnet', 'imagenet', 'coco', 'place']
    dataset_prefix_list = ['scenenet', 'pbrnet', 'imagenet', 'coco', 'place', 'kinetics']
    for dataset_prefix in dataset_prefix_list:
        all_outputs, reuse_dict, ret_params = build_datasetnet(inputs, all_outputs = all_outputs, reuse_dict = reuse_dict, dataset_prefix = dataset_prefix, **kwargs)
        if not ret_params is None:
            ret_params_final = ret_params

    all_outputs, reuse_dict, _ = build_datasetnet(inputs, all_outputs = all_outputs, reuse_dict = reuse_dict, dataset_prefix = 'nyuv2', **kwargs)

    return all_outputs, ret_params_final

def combine_normal_tfutils_new(inputs, center_im = False, cfg_dataset = {}, twonormals = 0, categorymulti = 1, no_prep = 0, **kwargs):
    all_outputs = []
    encode_reuse = None
    decode_reuse = None
    normal_reuse = None
    depth_reuse = None
    category_reuse = None
    ins_decode_reuse = None
    ret_params = None

    inputs = input_reshape_mult(inputs, categorymulti = categorymulti)

    if cfg_dataset.get('scenenet', 0)==1 and 'image_scenenet' in inputs:
        image_scenenet = tf.cast(inputs['image_scenenet'], tf.float32)
        image_scenenet = tf.div(image_scenenet, tf.constant(255, dtype=tf.float32))
        if center_im:
            image_scenenet  = tf.subtract(image_scenenet, tf.constant(0.5, dtype=tf.float32))

        output_nodes = []

        all_out_dict_scenenet = {}
        m_scenenet_encode, all_out_dict_scenenet = build_partnet(image_scenenet, key_want = 'encode', reuse_flag = encode_reuse, reuse_batch = None, batch_name = '_scenenet', all_out_dict = all_out_dict_scenenet, **kwargs)
        encode_reuse = True

        m_scenenet_decode, all_out_dict_scenenet = build_partnet(None, key_want = 'decode', reuse_flag = decode_reuse, reuse_batch = None, batch_name = '_scenenet', all_out_dict = all_out_dict_scenenet, **kwargs)
        decode_reuse = True
        
        if cfg_dataset.get('scene_normal', 1)==1:

            if twonormals==0:
                m_scenenet_normal, all_out_dict_scenenet = build_partnet(None, key_want = 'normal', reuse_flag = normal_reuse, reuse_batch = None, batch_name = '_scenenet', all_out_dict = all_out_dict_scenenet, **kwargs)
                normal_reuse = True
            else:
                m_scenenet_normal, all_out_dict_scenenet = build_partnet(None, key_want = 'normal_s', reuse_flag = None, reuse_batch = None, batch_name = '_scenenet', all_out_dict = all_out_dict_scenenet, **kwargs)

            output_nodes.append(m_scenenet_normal.output)
            ret_params = m_scenenet_normal.params

        if cfg_dataset.get('scene_depth', 1)==1:
            m_scenenet_depth, all_out_dict_scenenet = build_partnet(None, key_want = 'depth', reuse_flag = depth_reuse, reuse_batch = None, batch_name = '_scenenet', all_out_dict = all_out_dict_scenenet, **kwargs)
            depth_reuse = True
            output_nodes.append(m_scenenet_depth.output)
            ret_params = m_scenenet_depth.params

        if cfg_dataset.get('scene_instance', 0)==1:
            m_scenenet_ins_decode, all_out_dict_scenenet = build_partnet(None, key_want = 'ins_decode', reuse_flag = ins_decode_reuse, reuse_batch = None, batch_name = '_scenenet', all_out_dict = all_out_dict_scenenet, **kwargs)
            ins_decode_reuse = True
            m_scenenet_instance, all_out_dict_scenenet = build_partnet(None, key_want = 'scene_instance', reuse_flag = None, reuse_batch = None, batch_name = '_scenenet', all_out_dict = all_out_dict_scenenet, **kwargs)
            output_nodes.append(m_scenenet_instance.output)
            ret_params = m_scenenet_instance.params

        all_outputs.extend(output_nodes)

    if cfg_dataset.get('scannet', 0)==1 and 'image_scannet' in inputs:
        image_scannet = tf.cast(inputs['image_scannet'], tf.float32)
        image_scannet = tf.div(image_scannet, tf.constant(255, dtype=tf.float32))
        if center_im:
            image_scannet  = tf.subtract(image_scannet, tf.constant(0.5, dtype=tf.float32))

        all_out_dict_scannet = {}
        m_scannet_encode, all_out_dict_scannet = build_partnet(image_scannet, key_want = 'encode', reuse_flag = encode_reuse, reuse_batch = None, batch_name = '_scannet', all_out_dict = all_out_dict_scannet, **kwargs)
        encode_reuse = True

        m_scannet_decode, all_out_dict_scannet = build_partnet(None, key_want = 'decode', reuse_flag = decode_reuse, reuse_batch = None, batch_name = '_scannet', all_out_dict = all_out_dict_scannet, **kwargs)
        decode_reuse = True
        
        m_scannet_depth, all_out_dict_scannet = build_partnet(None, key_want = 'depth', reuse_flag = depth_reuse, reuse_batch = None, batch_name = '_scannet', all_out_dict = all_out_dict_scannet, **kwargs)
        depth_reuse = True

        all_outputs.extend([m_scannet_depth.output])
        ret_params = m_scannet_depth.params

    if cfg_dataset.get('pbrnet', 0)==1 and 'image_pbrnet' in inputs:
        image_pbrnet = tf.cast(inputs['image_pbrnet'], tf.float32)
        image_pbrnet = tf.div(image_pbrnet, tf.constant(255, dtype=tf.float32))
        if center_im:
            image_pbrnet  = tf.subtract(image_pbrnet, tf.constant(0.5, dtype=tf.float32))

        output_nodes = []

        all_out_dict_pbrnet = {}
        m_pbrnet_encode, all_out_dict_pbrnet = build_partnet(image_pbrnet, key_want = 'encode', reuse_flag = encode_reuse, reuse_batch = None, batch_name = '_pbrnet', all_out_dict = all_out_dict_pbrnet, **kwargs)
        encode_reuse = True

        m_pbrnet_decode, all_out_dict_pbrnet = build_partnet(None, key_want = 'decode', reuse_flag = decode_reuse, reuse_batch = None, batch_name = '_pbrnet', all_out_dict = all_out_dict_pbrnet, **kwargs)
        decode_reuse = True

        if cfg_dataset.get('pbr_normal', 1)==1:
            if twonormals==0:
                m_pbrnet_normal, all_out_dict_pbrnet = build_partnet(None, key_want = 'normal', reuse_flag = normal_reuse, reuse_batch = None, batch_name = '_pbrnet', all_out_dict = all_out_dict_pbrnet, **kwargs)
                normal_reuse = True
            else:
                m_pbrnet_normal, all_out_dict_pbrnet = build_partnet(None, key_want = 'normal_p', reuse_flag = None, reuse_batch = None, batch_name = '_pbrnet', all_out_dict = all_out_dict_pbrnet, **kwargs)

            output_nodes.append(m_pbrnet_normal.output)
            ret_params = m_pbrnet_normal.params
        
        if cfg_dataset.get('pbr_depth', 1)==1:
            m_pbrnet_depth, all_out_dict_pbrnet = build_partnet(None, key_want = 'depth', reuse_flag = depth_reuse, reuse_batch = None, batch_name = '_pbrnet', all_out_dict = all_out_dict_pbrnet, **kwargs)
            depth_reuse = True
            output_nodes.append(m_pbrnet_depth.output)
            ret_params = m_pbrnet_depth.params

        if cfg_dataset.get('pbr_instance', 0)==1:
            m_pbrnet_ins_decode, all_out_dict_pbrnet = build_partnet(None, key_want = 'ins_decode', reuse_flag = ins_decode_reuse, reuse_batch = None, batch_name = '_pbrnet', all_out_dict = all_out_dict_pbrnet, **kwargs)
            ins_decode_reuse = True
            m_pbrnet_instance, all_out_dict_pbrnet = build_partnet(None, key_want = 'pbr_instance', reuse_flag = None, reuse_batch = None, batch_name = '_pbrnet', all_out_dict = all_out_dict_pbrnet, **kwargs)
            output_nodes.append(m_pbrnet_instance.output)
            ret_params = m_pbrnet_instance.params

        all_outputs.extend(output_nodes)

    if cfg_dataset.get('imagenet', 0)==1 and 'image_imagenet' in inputs:
        image_imagenet = tf.cast(inputs['image_imagenet'], tf.float32)
        if no_prep==0:
            image_imagenet = tf.div(image_imagenet, tf.constant(255, dtype=tf.float32))
            if center_im:
                image_imagenet  = tf.subtract(image_imagenet, tf.constant(0.5, dtype=tf.float32))

        all_out_dict_imagenet = {}
        m_imagenet_encode, all_out_dict_imagenet = build_partnet(image_imagenet, key_want = 'encode', reuse_flag = encode_reuse, reuse_batch = None, batch_name = '_imagenet', all_out_dict = all_out_dict_imagenet, **kwargs)
        encode_reuse = True
        
        m_imagenet_category, all_out_dict_imagenet = build_partnet(None, key_want = 'category', reuse_flag = category_reuse, reuse_batch = None, batch_name = '_imagenet', all_out_dict = all_out_dict_imagenet, **kwargs)
        category_reuse = True

        all_outputs.extend([m_imagenet_category.output])
        ret_params = m_imagenet_category.params

    if cfg_dataset.get('coco', 0)==1 and 'image_coco' in inputs:

        image_coco = tf.cast(inputs['image_coco'], tf.float32)
        image_coco = tf.div(image_coco, tf.constant(255, dtype=tf.float32))
        if center_im:
            image_coco  = tf.subtract(image_coco, tf.constant(0.5, dtype=tf.float32))

        output_nodes = []

        all_out_dict_coco = {}
        m_coco_encode, all_out_dict_coco = build_partnet(image_coco, key_want = 'encode', reuse_flag = encode_reuse, reuse_batch = None, batch_name = '_coco', all_out_dict = all_out_dict_coco, **kwargs)
        encode_reuse = True

        m_coco_ins_decode, all_out_dict_coco = build_partnet(None, key_want = 'ins_decode', reuse_flag = ins_decode_reuse, reuse_batch = None, batch_name = '_coco', all_out_dict = all_out_dict_coco, **kwargs)
        ins_decode_reuse = True
        m_coco_instance, all_out_dict_coco = build_partnet(None, key_want = 'coco_instance', reuse_flag = None, reuse_batch = None, batch_name = '_coco', all_out_dict = all_out_dict_coco, **kwargs)
        output_nodes.append(m_coco_instance.output)
        ret_params = m_coco_instance.params

        all_outputs.extend(output_nodes)

    if cfg_dataset.get('place', 0)==1 and 'image_place' in inputs:
        image_place = tf.cast(inputs['image_place'], tf.float32)
        image_place = tf.div(image_place, tf.constant(255, dtype=tf.float32))
        if center_im:
            image_place  = tf.subtract(image_place, tf.constant(0.5, dtype=tf.float32))

        all_out_dict_place = {}
        m_place_encode, all_out_dict_place = build_partnet(image_place, key_want = 'encode', reuse_flag = encode_reuse, reuse_batch = None, batch_name = '_place', all_out_dict = all_out_dict_place, **kwargs)
        encode_reuse = True
        
        m_place_category, all_out_dict_place = build_partnet(None, key_want = 'place_category', reuse_flag = None, reuse_batch = None, batch_name = '_place', all_out_dict = all_out_dict_place, **kwargs)

        all_outputs.extend([m_place_category.output])
        ret_params = m_place_category.params

    if cfg_dataset.get('nyuv2', 0)==1 and 'image_nyuv2' in inputs:
        image_nyuv2 = tf.cast(inputs['image_nyuv2'], tf.float32)
        image_nyuv2 = tf.div(image_nyuv2, tf.constant(255, dtype=tf.float32))
        if center_im:
            image_nyuv2  = tf.subtract(image_nyuv2, tf.constant(0.5, dtype=tf.float32))

        all_out_dict_nyuv2 = {}
        m_nyuv2_encode, all_out_dict_nyuv2 = build_partnet(image_nyuv2, key_want = 'encode', reuse_flag = encode_reuse, reuse_batch = None, batch_name = '_nyuv2', all_out_dict = all_out_dict_nyuv2, **kwargs)
        encode_reuse = True

        m_nyuv2_decode, all_out_dict_nyuv2 = build_partnet(None, key_want = 'decode', reuse_flag = decode_reuse, reuse_batch = None, batch_name = '_nyuv2', all_out_dict = all_out_dict_nyuv2, **kwargs)
        decode_reuse = True
        
        m_nyuv2_depth, all_out_dict_nyuv2 = build_partnet(None, key_want = 'depth', reuse_flag = depth_reuse, reuse_batch = None, batch_name = '_nyuv2', all_out_dict = all_out_dict_nyuv2, **kwargs)
        depth_reuse = True

        all_outputs.extend([m_nyuv2_depth.output])
        #ret_params = m_nyuv2_depth.params

    return all_outputs, ret_params

def split_input(inputs, n_gpus = 1):
    if n_gpus==1:
        return [inputs]

    temp_args = {v: tf.split(inputs[v], axis = 0, num_or_size_splits=n_gpus) for v in inputs}
    list_of_args = [{now_arg: temp_args[now_arg][ind] for now_arg in temp_args} for ind in xrange(n_gpus)]

    return list_of_args

def parallel_network_tfutils(inputs, model_func, n_gpus = 1, gpu_offset = 0, **kwargs):
    with tf.variable_scope(tf.get_variable_scope()) as vscope:
        inputs = split_input(inputs, n_gpus)

        outputs = []
        params = []
        for i, curr_input in enumerate(inputs):
            with tf.device('/gpu:%d' % (i + gpu_offset)):
                with tf.name_scope('gpu_' + str(i)) as gpu_scope:
                    curr_output, curr_param = model_func(curr_input, **kwargs)
                    outputs.append(curr_output)
                    params.append(curr_param)
                    tf.get_variable_scope().reuse_variables()

        return outputs, params[0]
