#!/usr/bin/env python
# coding: utf-8

# # motionLAB Annotations Data Format

import argparse
import datetime
import json

from tqdm import tqdm


def init_json(file='mlab.json'):
    output = {
        "info": None,
        "licenses": [],
        "categories": [],
        "videos": [],
        "images": [],
        "tracks": [],
        "segment_info": [],
        "annotations": [],
        "datasets": [{'name': 'COCO', 'id': 1}, {'name': 'TAO', 'id': 2}]
    }
    output['info'] = {
        "description": "Mixed Dataset",
        "url": "",
        "version": "1",
        "year": 2020,
        "date_created": datetime.datetime.utcnow().isoformat(' ')
    }

    with open(file, 'w') as f:
        json.dump(output, f)
    print("JSON INITIATED : {}".format(file))


def last_value(mlabjson, key="categories", subkey="id", initvalue=None):
    last_value = initvalue
    l = mlabjson[key]
    if l:
        last = mlabjson[key][-1]
        if last[subkey]: last_value = last[subkey]
    return last_value


def merge_keys(mlabjson, newjson, keys_struct_d, dataset_id=1, key="images", indexed_key='image_id', root_dir=None,
               dir_key=['file_name', 'video']):
    # NOTE: root_dir format str: 'root/path/'
    # NOTE: should update mlabjson and newjson

    # ###update newjson
    mlab_last_id = last_value(mlabjson, key=key, subkey="id", initvalue=0)  # get last key value
    original_id_a = []
    new_id_a = []
    # #update key
    for ik, k in enumerate(tqdm(newjson[key], desc='update key: {}'.format(key))):  # update keys and values
        if isinstance(k, dict):  # in case is "info": str or 'licenses' : ['unknown']
            original_id = k['id']
            new_id = mlab_last_id + (ik + 1)
            original_id_a.append(original_id)
            new_id_a.append(new_id)
            # ('original id : {} > new id: {}'.format(original_id, new_id))
            # update newjson id keys
            newjson[key][ik]['id'] = new_id  # update newjson id
            # additional specific updates
            newjson[key][ik]['dataset'] = dataset_id  # add dataset_id
            if key == 'images':  # update image filnemate root dir
                if root_dir:
                    for dk in dir_key:
                        try:
                            newjson[key][ik][dk] = root_dir + newjson[key][ik][dk]
                        except:
                            continue
    if indexed_key:  # update newjson indexed keys id (inside other newjson keys)
        for nj_k in newjson:
            subs = 0
            # print("Finding {} in {}".format(indexed_key, nj_k))
            if isinstance(newjson[nj_k], list) and indexed_key in keys_struct_d[nj_k]:
                # print("Found {} in {}".format(indexed_key, nj_k))
                for inj_v, nj_v in enumerate(
                        tqdm(newjson[nj_k], desc='update {} : {}'.format(nj_k, indexed_key))):  # list
                    try:
                        if nj_v[indexed_key] in original_id_a:
                            idx = original_id_a.index(nj_v[indexed_key])
                            newjson[nj_k][inj_v][indexed_key] = new_id_a[idx]
                            subs += 1
                    except Exception as e:
                        # print('#WARNING: Problem in indexing: {}'.format(repr(e)))
                        continue
            # print("{} id substitutions: {}".format(nj_k,subs))
    # ###update mlabjson
    mlabjson[key] = mlabjson[key] + newjson[key]  # merge newjson : + or .extend()

    return mlabjson, newjson


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mlabfile', type=str, default='mlab.json', help='mlab json path')
    parser.add_argument('--mergefile', type=str, help='annotation to merge json path')
    parser.add_argument('--dataset_id', type=int, help='view this script in init_json() the dataset id')
    parser.add_argument('--root_dir', type=str, default=None, help='root dir of the datasets location')
    parser.add_argument('--dir_key', nargs="+", default=['file_name', 'video'], help='key to change the root_dir')
    parser.add_argument('--initjson', action='store_true', help='create the mlab.json, or initiae it again')
    opt = parser.parse_args()

    mlabjsonfile = opt.mlabfile
    newjsonfile = opt.mergefile
    dataset_id = opt.dataset_id
    root_dir = opt.root_dir
    dir_key = opt.dir_key
    initjson = opt.initjson
    print('\n>>' + str(opt))

    # INIT JSON
    if initjson: init_json(file=mlabjsonfile)

    # LOAD JSONS
    mlabjson = json.load(open(mlabjsonfile))
    newjson = json.load(open(newjsonfile))

    # GEG JSON STRUCK
    keys_struct_d = {}
    for nj_k in newjson:
        keys_struct_d[nj_k] = []
        for ik, k in enumerate(tqdm(newjson[nj_k], desc='get keys {}'.format(nj_k))):
            if isinstance(k, dict):  # in case is "info": str or 'licenses' : ['unknown']
                child_keys = list(newjson[nj_k][ik].keys())  # child keys
                keys_struct_d[nj_k] = keys_struct_d[nj_k] + list(
                    set(child_keys) - set(keys_struct_d[nj_k]))  # append only new keys

    # ##### Merging licenses (#WARNING update before images, videos, etc)
    try:
        print('\n >> MERGING LICENSES...')
        key = 'licenses'
        indexed_key = 'license'
        mlabjson, newjson = merge_keys(mlabjson, newjson, keys_struct_d, dataset_id=dataset_id, key=key,
                                       indexed_key=indexed_key)
    except Exception as e:
        print('#WARNING: Problem in Merging: {}'.format(repr(e)))
    pass

    # ##### Merging Catagories
    try:
        print('\n >> MERGING CATEGORIES...')
        key = 'categories'
        indexed_key = 'category_id'
        mlabjson, newjson = merge_keys(mlabjson, newjson, keys_struct_d, dataset_id=dataset_id, key=key,
                                       indexed_key=indexed_key)
    except Exception as e:
        print('#WARNING: Problem in Merging: {}'.format(repr(e)))
        pass

    # ##### Merging videos (TAO; before merging images and tracks in TAO, because of video_id)
    try:
        print('\n >> MERGING VIDEOS...')
        key = 'videos'
        indexed_key = 'video_id'
        mlabjson, newjson = merge_keys(mlabjson, newjson, keys_struct_d, key=key, dataset_id=dataset_id,
                                       indexed_key=indexed_key,
                                       root_dir=root_dir,
                                       dir_key=dir_key)
    except Exception as e:
        print('#WARNING: Problem in Merging: {}'.format(repr(e)))
        pass

    # ##### Merging images (licenses needs to be first updated)
    try:
        print('\n >> MERGING IMAGES...')
        key = 'images'
        indexed_key = 'image_id'
        mlabjson, newjson = merge_keys(mlabjson, newjson, keys_struct_d, key=key, dataset_id=dataset_id,
                                       indexed_key=indexed_key,
                                       root_dir=root_dir,
                                       dir_key=dir_key)
    except Exception as e:
        print('#WARNING: Problem in Merging: {}'.format(repr(e)))
        pass

    # ##### Merge tracks (TAO;)
    try:
        print('\n >> MERGING TRACKS...')
        key = 'tracks'
        indexed_key = 'track_id'
        mlabjson, newjson = merge_keys(mlabjson, newjson, keys_struct_d, dataset_id=dataset_id, key=key, indexed_key=indexed_key,
                                       root_dir=root_dir, dir_key=dir_key)
    except Exception as e:
        print('#WARNING: Problem in Merging: {}'.format(repr(e)))
        pass

    # ##### Merge segment_info (COCO;)
    try:
        print('\n >> MERGING SEGMENT_INFO...')
        key = 'segment_info'
        indexed_key = None
        mlabjson, newjson = merge_keys(mlabjson, newjson, keys_struct_d, dataset_id=dataset_id, key=key, indexed_key=indexed_key,
                                       root_dir=root_dir,
                                       dir_key=dir_key)
    except Exception as e:
        print('#WARNING: Problem in Merging: {}'.format(repr(e)))
        pass

    # ##### Merging Annotations (#WARNING LAST to Merge, because it has the other ids)
    try:
        print('\n >> MERGING ANNOTATIONS...')
        key = 'annotations'
        indexed_key = None
        mlabjson, newjson = merge_keys(mlabjson, newjson, keys_struct_d, dataset_id=dataset_id, key=key, indexed_key=indexed_key)
    except Exception as e:
        print('Problem in Merging licences: {}'.format(repr(e)))
        pass

    # SAVE JSON
    print('\n >> SAVING...')
    with open(mlabjsonfile, 'w') as f:
        json.dump(mlabjson, f)
    print("JSON SAVED : {}".format(mlabjsonfile))
