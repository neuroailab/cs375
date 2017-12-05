from __future__ import division, print_function, absolute_import
import os, sys
import numpy as np

import tensorflow as tf
import cPickle
import argparse

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def get_parser():

    parser = argparse.ArgumentParser(description='The script to test the cond dataset input')
    parser.add_argument('--tfr1path', default = '/data/chengxuz/test_tfrs/tfrs_1', type = str, action = 'store', help = 'Tfr path for dataset 1')
    parser.add_argument('--tfr2path', default = '/data/chengxuz/test_tfrs/tfrs_2', type = str, action = 'store', help = 'Tfr path for dataset 2')
    parser.add_argument('--maketfr', default = 0, type = int, action = 'store', help = 'Whether to make tfr')
    parser.add_argument('--tfrlen', default = 10, type = int, action = 'store', help = 'Length of tfrs to make')
    parser.add_argument('--contlen', default = 100, type = int, action = 'store', help = 'Number of records in one tfrecord file')
    
    return parser

def make_tfr(args):
    # Build the tfrecords needed for test

    dir_list = [args.tfr1path, args.tfr2path]

    for curr_dir in dir_list:
        curr_num = 0
        os.system('mkdir -p %s' % curr_dir)
        for which_tfr in xrange(args.tfrlen):
            curr_path = os.path.join(curr_dir, 'data_%i.tfrecords' % which_tfr)
            writer = tf.python_io.TFRecordWriter(curr_path)
            for curr_indx in xrange(args.contlen):
                example = tf.train.Example(features=tf.train.Features(feature={
                    'feat': _int64_feature(curr_num)}))
                writer.write(example.SerializeToString())

                curr_num = curr_num + 1
            writer.close()

def get_data(file_list):
    filename_queue = tf.train.string_input_producer(
        file_list, num_epochs=None, shuffle=False)

    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)

    features_dict = {
            'feat': tf.FixedLenFeature([], tf.int64),
        }
    features = tf.parse_single_example(
        serialized_example,
        features=features_dict)
    feats = tf.cast(features['feat'], tf.int32)

    #feats_batch = tf.train.batch_join(
    #    [[feats]], batch_size=5)

    #return feats_batch, feats
    return None, feats

def main():
    parser = get_parser()

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = ''

    if args.maketfr==1:
        make_tfr(args)
        return
    
    file_list1 = tf.gfile.Glob(os.path.join(args.tfr1path, '*.tfrecords'))
    file_list1.sort()
    file_list2 = tf.gfile.Glob(os.path.join(args.tfr2path, '*.tfrecords'))
    file_list2.sort()

    feats_batch1, feats1 = get_data(file_list1)
    feats_batch2, feats2 = get_data(file_list2)

    feats_batch1_, feats_batch2_ = tf.train.batch_join(
        [[feats1, feats2]], batch_size=5)

    sess = tf.Session()

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    control_flag = tf.constant(0)

    #feat_cond = tf.cond(tf.equal(control_flag, tf.constant(0)), fn1 = lambda : feats_batch1, fn2 = lambda : feats_batch2)

    for _ in xrange(100):
        #feats1, feats2 = sess.run([feats_batch1, feats_batch2])
        #feat = sess.run(feat_cond)

        #feat = sess.run(feats_batch1)
        #feats1, feats2 = sess.run([feats_batch1, feats_batch2])

        feat = sess.run(feats_batch1_)
        feats1, feats2 = sess.run([feats_batch1_, feats_batch2_])
        print(feat, feats1, feats2)

    coord.request_stop()

    # Wait for threads to finish.
    coord.join(threads)
    sess.close()

if __name__ == '__main__':
    main()
