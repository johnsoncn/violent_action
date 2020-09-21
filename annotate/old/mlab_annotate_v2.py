#!/usr/bin/env python
# coding: utf-8

# # motionLAB Annotations Data Format

import json
import datetime
import argparse

def init_json(file='mlab.json'):
    output = {
        "info": None,
        "licenses": [],
        "categories": [],
        "images": [],
        "annotations": [],
        "datasets": [ {'name': 'COCO', 'id' : 1 }, {'name': 'TAO', 'id' : 2} ]
    }

    output['info']= {
        "description": "Mixed Dataset",
        "url": "",
        "version": "1",
        "year": 2020,
        "date_created": datetime.datetime.utcnow().isoformat(' ')
    }



    with open(file, 'w') as f:
        json.dump(output,f)
    print("JSON INITIATED : {}".format(file))

init_json()



def last_value(mlabjson, key="categories", subkey="id", initvalue=None):
    last_value=initvalue
    l=mlabjson[key]
    if l:
        last = mlabjson[key][-1]
        if last[subkey]: last_value=last[subkey]
    return last_value



def merge_keys(mlabjson, newjson, dataset_id=1, key="images", indexed_key='image_id', root_dir=None, dir_key=['file_name', 'video']):
    #NOTE: root_dir format str: 'root/path/'
    #NOTE: should update mlabjson and newjson

    #update newjson
    mlab_last_id = last_value(mlabjson, key=key, subkey="id", initvalue=0) #get last key value
    for ik,k in enumerate(newjson[key]): #update keys and values
        if isinstance(k, dict): #in case is "info": str or 'licenses' : ['unknown']
            original_id=k['id']
            new_id=mlab_last_id+(ik+1)
            print('original id : {} > new id: {}'.format(original_id, new_id))
            #update newjson id keys
            newjson[key][ik]['id']=new_id  #update newjson id
            if indexed_key:  #update newjson indexed keys id (inside other newjson keys)
                for nj_k in newjson:
                    subs=0
                    if isinstance(newjson[nj_k], list) and isinstance(newjson[nj_k][0], dict) and newjson[nj_k][0].get(indexed_key, False):
                        for inj_v, nj_v in enumerate(newjson[nj_k]):
                                if nj_v[indexed_key]==original_id:
                                    newjson[nj_k][inj_v][indexed_key] = new_id
                                    subs+=1
                    print("{} id substitutions: {}".format(nj_k,subs))

            #additional specific updates
            newjson[key][ik]['dataset']=dataset_id  #add dataset_id
            if key=='images':  #update image filnemate root dir
                if root_dir:
                    for dk in dir_key:
                        try:
                            newjson[key][ik][dk]=root_dir+newjson[key][ik][dk]
                        except:
                            pass

    #update mlabjson
    mlabjson[key]=mlabjson[key]+newjson[key] #merge newjson : + or .extend()

    return mlabjson, newjson




if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mlabfile', type=str, default='mlab.json', help='mlab json path')
    parser.add_argument('--mergefile', type=str, help='annotation to merge json path')
    parser.add_argument('--dataset_id', type=int, help='view this script in init_json() the dataset id')
    parser.add_argument('--root_dir', type=str, default=None, help='root dir of the datasets location')
    parser.add_argument('--dir_key', type=str, default=['file_name', 'video'], help='key to change the root_dir')
    parser.add_argument('--init_json', type=str, default=False, help='create the mlab.json, or initiae it again')
    opt = parser.parse_args()

    mlabjsonfile=opt.mlabfile
    newjsonfile=opt.mergefile
    dataset_id=opt.dataset_id
    root_dir=opt.root_dir
    dir_key=opt.dir_key
    init_json=opt.init_json

    #INIT JSON
    if init_json: init_json(file=mlabjsonfile)

    #LOAD JSONS
    mlabjson=json.load(open('mlab.json'))
    newjson=json.load(open(jsonfile))

    # ##### Merging licenses (#WARNING update before images, videos, etc)
    key='licenses'
    indexed_key='license'
    mlabjson, newjson=merge_keys(mlabjson, newjson, dataset_id=dataset_id, key=key, indexed_key=indexed_key)

    # ##### Merging Catagories
    key='categories'
    indexed_key='category_id'
    mlabjson, newjson=merge_keys(mlabjson, newjson, dataset_id=dataset_id, key=key, indexed_key=indexed_key)

    # ##### Merging images (licenses needs to be first updated)
    key='images'
    indexed_key='image_id'
    mlabjson, newjson=merge_keys(mlabjson, newjson, key=key, indexed_key=indexed_key, root_dir=root_dir, dir_key=dir_key)

    # ##### Merging Annotations (#WARNING LAST to Merge, because it has the other ids)
    key='annotations'
    indexed_key=None
    mlabjson, newjson=merge_keys(mlabjson, newjson, key=key, indexed_key=indexed_key)

    #SAVE JSON
    with open(mlabjsonfile, 'w') as f:
        json.dump(output,f)
    print("JSON SAVED : {}".format(mlabjsonfile))
