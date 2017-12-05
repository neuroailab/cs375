from __future__ import division, print_function, absolute_import
import os, sys
import numpy as np

import cPickle

import json
import copy
import argparse
import pymongo as pm
import gridfs

def get_parser():
    parser = argparse.ArgumentParser(description='The script to delete the models saved in mongodb')
    parser.add_argument('--nport', default = 27009, type = int, action = 'store', help = 'Port number of mongodb')
    parser.add_argument('--expId', default = "combinet_alexnet_ncp_new_2", type = str, action = 'store', help = 'Name of experiment id')
    parser.add_argument('--dbname', default = "combinet-test", type = str, action = 'store', help = 'Database name')
    parser.add_argument('--collname', default = "combinet", type = str, action = 'store', help = 'Collection name')

    return parser

def main():

    parser = get_parser()
    args = parser.parse_args()

    load_conn = pm.MongoClient(port=args.nport)

    collfs = gridfs.GridFS(load_conn[args.dbname], args.collname)
    coll = collfs._GridFS__files
    query = {'exp_id': args.expId, 'saved_filters': True}
    count = collfs.find(query).count()
    count_gfs = coll.find(query).count()

    print(count, count_gfs)
    find_res = coll.find(query)
    print(find_res[0].keys())
    print(find_res[0]['chunkSize'])
    print(find_res[0]['filename'])
    print(find_res[0]['_id'])

    '''
    loading_from = coll
    fsbucket = gridfs.GridFSBucket(loading_from._Collection__database, bucket_name=loading_from.name.split('.')[0])

    filename = os.path.basename(find_res[0]['filename'])
    cache_filename = os.path.join('/home/chengxuz/.tfutils/tmp', filename)

    load_dest = open(cache_filename, "w+")
    load_dest.close()
    load_dest = open(cache_filename, 'rwb+')
    fsbucket.download_to_stream(find_res[0]['_id'], load_dest)
    load_dest.close()
    '''

    #collfs.delete(find_res[0]['_id'])
    loading_from = coll
    fsbucket = gridfs.GridFSBucket(loading_from._Collection__database, bucket_name=loading_from.name.split('.')[0])
    #fsbucket.delete(find_res[0]['_id'])

if __name__ == '__main__':
    main()
