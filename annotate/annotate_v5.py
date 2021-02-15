#!/usr/bin/env python
# coding: utf-8

# # motionLAB Annotations Data Format

import argparse
import datetime
import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Image, display
from matplotlib.image import imread
from matplotlib.patches import Rectangle
from tqdm import tqdm


# ## Merge Functions

def init_json(file='mola.json'):
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


def last_value(molajson, key="categories", subkey="id", initvalue=None):
    last_value = initvalue
    l = molajson[key]
    if l:
        last = molajson[key][-1]
        if last[subkey]: last_value = last[subkey]
    return last_value


def merge_keys(molajson, newjson, keys_struct_d, dataset_id=1, key="images", indexed_key='image_id', root_dir=None,
               dir_key=['file_name', 'video']):
    # NOTE: root_dir format str: 'root/path/'
    # NOTE: should update molajson and newjson

    # ###update newjson
    mola_last_id = last_value(molajson, key=key, subkey="id", initvalue=0)  # get last key value
    original_id_a = []
    new_id_a = []
    # #update key
    for ik, k in enumerate(tqdm(newjson[key], desc='update key: {}'.format(key))):  # update keys and values
        if isinstance(k, dict):  # in case is "info": str or 'licenses' : ['unknown']
            original_id = k['id']
            new_id = mola_last_id + (ik + 1)
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
    # ###update molajson
    molajson[key] = molajson[key] + newjson[key]  # merge newjson : + or .extend()

    return molajson, newjson


# ## Fix categories Functions

def display_imgs(rdir, i, dset_l, classes_l_catid, classes_l_dset, classes_l_imgid, img_l, img_l_id, imgidx=0):
    for ii, did in enumerate(
            classes_l_catid[i]):  # WARNING: only in classes each id should correspond to a different dataset
        print(dset_l[classes_l_dset[i][ii][imgidx] - 1])  # WARNING: it works because molajson[datasets] is ordered
        imgid = classes_l_imgid[i][ii][imgidx]
        img_l_idx = img_l_id.index(imgid)  # np.where(img_l_id_np==imgid)[0].tolist()
        imgpath = rdir + img_l[img_l_idx]
        print(imgpath)
        display(Image(imgpath))


def save_classtofix(i, classtofix_l, classtofix_l_catid, classes_l, classes_l_catid, classes_l_dset,
                    classes_l_imgid, img_l, img_l_id, imgidx=0):
    # INCEPTION Functions ===>dream within a dream =>>> e fez-se chockapic!!!
    x = input('\n>> Choose: Rename Class and fix; 1=Fix only; 2=Do not fix; 3=show other pair; 4=Stop loop ')
    if x == '1':
        classtofix_l.append(classes_l[i])
        classtofix_l_catid.append(classes_l_catid[i])
    elif x == '2':
        pass
    elif x == '3':
        print('\n>> #WARNING showing another pair')
        imgidx += 1
        display_imgs(i, classes_l_catid, classes_l_dset, classes_l_imgid, img_l, img_l_id, imgidx=imgidx)
        classtofix_l, classtofix_l_catid = save_classtofix(i, classtofix_l, classtofix_l_catid, classes_l,
                                                           classes_l_catid, classes_l_dset, classes_l_imgid,
                                                           img_l, img_l_id, imgidx=imgidx)
    elif x == '4':
        raise RuntimeError('Stop')
    elif isinstance(x, str) and len(x) > 2:  # Rename only if string with more than 2 characters
        classtofix_l.append(x)
        classtofix_l_catid.append(classes_l_catid[i])
    else:
        print('\n>> #ERROR Try again! If Renaming use more than 2 characters! ')
        classtofix_l, classtofix_l_catid = save_classtofix(i, classtofix_l, classtofix_l_catid, classes_l,
                                                           classes_l_catid, classes_l_dset, classes_l_imgid,
                                                           img_l, img_l_id, imgidx=imgidx)

    return classtofix_l, classtofix_l_catid


def save_imgs(dataframe, rdir, path, i, dset_l, classes_l, classes_l_catid, classes_l_bbox, classes_l_dset,
              classes_l_imgid, img_l, img_l_id, startidx=0, imgnr=1, imgstep=None, showimage=False):
    annotations_missing = np.zeros(len(classes_l_catid[i]),
                                   dtype=int).tolist()  # missing annotation for each classe
    images_missing = np.zeros(len(classes_l_catid[i]), dtype=int).tolist()  # missing images for each classe
    for ii, did in enumerate(
            classes_l_catid[i]):  # WARNING: only in classes each id should correspond to a different dataset
        # category
        category = classes_l[i]
        if isinstance(category, list):
            category = category[ii]  # if similar names [cow, cowgirl]
            print(">>> finding similar category: ", category)
        # ANNOTATION exist?
        if not classes_l_imgid[i][ii] or len(classes_l_imgid[i][ii]) == 0:
            # print("\n>> {} is missing from annotations dataset {}".format(category,dset_l[ii] ))
            print(">>>> #WARNING {} is missing from json annotations".format(category))
            print('annotations datasets: ', classes_l_dset[i][ii])
            print('annotations imgids: ', classes_l_imgid[i][ii])
            print('annotations bbox: ', classes_l_bbox[i][ii])
            annotations_missing[ii] = 1
            continue

        dataset = dset_l[
            classes_l_dset[i][ii][startidx] - 1]  # #WARNING: it works because molajson[datasets] is ordered
        print(dataset)
        # SAVE IMAGE
        dpi = 80
        imgidx_l = [startidx]  # default
        classes_imgidx = range(len(classes_l_imgid[i][ii]))
        if not imgstep: imgstep = 1  # default
        if imgnr > len(classes_imgidx): imgnr = len(classes_imgidx)
        if isinstance(imgstep, int) and startidx + imgstep * imgnr > len(classes_imgidx): imgstep = int(
            (len(classes_imgidx) - startidx) / imgnr)
        if isinstance(imgstep, int): imgidx_l = [startidx + inr * imgstep for inr in range(imgnr)]
        if isinstance(imgstep, str) and imgstep == 'random': imgidx_l = random.sample(classes_imgidx, imgnr)
        for il, igidx in enumerate(imgidx_l):
            imgid = classes_l_imgid[i][ii][igidx]
            bbox = classes_l_bbox[i][ii][igidx]
            img_l_idx = img_l_id.index(imgid)  # np.where(img_l_id_np==imgid)[0].tolist()
            imgpath = rdir + img_l[img_l_idx]
            src = imgpath
            # IMAGE exist?
            found = False
            try:
                img = imread(src)
                found = True
            except FileNotFoundError as e:
                print(e)
                for igidx_new, imgid in enumerate(tqdm(classes_l_imgid[i][ii],
                                                       desc='>>> Finding {} {}...'.format(dataset,
                                                                                          category))):  # find other not in the list
                    if igidx_new in imgidx_l: continue  # don't search the index in the imgidx_l
                    imgid = classes_l_imgid[i][ii][igidx_new]
                    bbox = classes_l_bbox[i][ii][igidx_new]
                    img_l_idx = img_l_id.index(imgid)  # np.where(img_l_id_np==imgid)[0].tolist()
                    imgpath = rdir + img_l[img_l_idx]
                    src = imgpath
                    try:
                        img = imread(src)
                        found = True
                        imgidx_l[il] = igidx_new
                        break
                    except:
                        continue
            if found:
                extension = os.path.splitext(src)[1]
                dst = path + dataset + '_' + category + '_' + str(imgid) + extension
                try:
                    height, width, nbands = img.shape
                except:
                    height, width = img.shape
                figsize = width / float(dpi), height / float(
                    dpi)  # What size does the figure need to be in inches to fit the image?
                # bbox
                x, y, w, h = bbox
                rect = Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
                # save fig
                fig = plt.figure(
                    figsize=figsize)  # Create a figure of the right size with one axes that takes up the full figure
                ax = fig.add_axes([0, 0, 1, 1])
                ax.axis('off')  # Hide spines, ticks, etc.
                # fig,ax = plt.subplots(1)
                ax.imshow(img)
                ax.add_patch(rect)
                # Ensure we're displaying with square pixels and the right extent.
                # This is optional if you haven't called `plot` or anything else that might
                # change the limits/aspect.  We don't need this step in this case.
                ax.set(xlim=[-0.5, width - 0.5], ylim=[height - 0.5, -0.5], aspect=1)
                fig.savefig(dst, dpi=dpi, transparent=True)
                if showimage: plt.show()
                plt.ioff()
                plt.close()
            else:
                images_missing[ii] = 1
                print('>>>> #WARNING: No file found for: {} {}...'.format(dataset, category))
    dataframe.at[i, 'annotations_missing'] = annotations_missing
    dataframe.at[i, 'images_missing'] = images_missing
    return dataframe


def view_classe(rdir, i, ii, dataset, category, classes_l_imgid, classes_l_bbox, img_l, img_l_id, step=10,
                   dpi=80):
    counter = 0
    for imgidx, imgid in enumerate(
            tqdm(classes_l_imgid[i][ii], desc='>> Finding {} {}...'.format(dataset, category))):
        if imgidx == counter * step:
            imgid = classes_l_imgid[i][ii][imgidx]
            bbox = classes_l_bbox[i][ii][imgidx]
            img_l_idx = img_l_id.index(imgid)  # np.where(img_l_id_np==imgid)[0].tolist()
            imgpath = rdir + img_l[img_l_idx]
            src = imgpath
            try:
                # imgread
                img = imread(src)
            except:
                continue
            height, width, nbands = img.shape
            figsize = width / float(dpi), height / float(
                dpi)  # What size does the figure need to be in inches to fit the image?
            # bbox
            x, y, w, h = bbox
            rect = Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
            # fig
            fig = plt.figure(
                figsize=figsize)  # Create a figure of the right size with one axes that takes up the full figure
            ax = fig.add_axes([0, 0, 1, 1])
            ax.axis('off')  # Hide spines, ticks, etc.
            # fig,ax = plt.subplots(1)
            ax.imshow(img)
            ax.add_patch(rect)
            # Ensure we're displaying with square pixels and the right extent.
            # This is optional if you haven't called `plot` or anything else that might
            # change the limits/aspect.  We don't need this step in this case.
            ax.set(xlim=[-0.5, width - 0.5], ylim=[height - 0.5, -0.5], aspect=1)
            plt.show()
            counter += 1


def convert_unicode(uni, method='liststr'):
    result = None
    if method == 'liststr':
        result = []
        try:
            result = [value.replace("'", "") for value in
                      uni.strip('[]').split(', ')]  # WARNING: ", " - space to split is necessary
        except:
            pass
    if method == 'listnum':
        result = []
        try:
            result = [int(value) for value in uni.strip('[]').split(
                ', ')]  # uni.strip('[]').split(',') #[value for value in uni.strip('[]').split(',')]
        except:
            pass
    if method == 'listoflistnum':
        result = []
        try:
            result = [[float(value.replace('[', '')) for value in epoch.split(',')] for epoch in
                      uni.strip('[]').split('],')]
        except:
            pass
    if method == 'itemnum':
        result = []
        try:
            result = int(uni.strip('[]'))  # uni.strip('[]').split(',') #[value for value in uni.strip('[]').split(',')]
        except:
            pass

    return result


def parse_path(path):
    """
    Python only works with '/', not '\\'or '\'

    WARNING:
        in windows use r'path' because of escape literals , e.g: "."
        os.path.realpath(path).replace('\\', '/') #BUG os.path.realpath removes the last '\\' and if your sending a folder it is a problem

    """
    parsed_path = path.replace('\\', '/')
    parsed_path = parsed_path.replace("\ ", '/')
    return parsed_path


def assure_path_exists(path):
    """
    Create directory folders from path

    Note: windows path strings should be r'path'
    """

    # Only for dirs - for complete you have to change dir for path
    dir_p = os.path.normpath(path)  # dirname is obrigatory - make sure it is a dir
    if not os.path.exists(dir_p):
        os.makedirs(dir_p)


if __name__ == '__main__':
    # ## Run merge functions
    parser = argparse.ArgumentParser()
    parser.add_argument('--molafile', type=str, default='mola.json', help='mola json path')
    parser.add_argument('--mergefile', type=str, help='annotation to merge json path')
    parser.add_argument('--dataset_id', type=int, help='view this script in init_json() the dataset id')
    parser.add_argument('--root_dir', type=str, default=None, help='root dir of the datasets location')
    parser.add_argument('--dir_key', nargs="+", default=['file_name', 'video'], help='key to change the root_dir')
    parser.add_argument('--initjson', action='store_true', help='create the mola.json, or initiae it again')
    opt = parser.parse_args()

    molajsonfile = opt.molafile
    newjsonfile = opt.mergefile
    dataset_id = opt.dataset_id
    root_dir = opt.root_dir
    dir_key = opt.dir_key
    initjson = opt.initjson
    print('\n>>' + str(opt))

    # INIT JSON
    if initjson: init_json(file=molajsonfile)

    # LOAD JSONS
    molajson = json.load(open(molajsonfile))
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
        molajson, newjson = merge_keys(molajson, newjson, keys_struct_d, dataset_id=dataset_id, key=key,
                                       indexed_key=indexed_key)
    except Exception as e:
        print('#WARNING: Problem in Merging: {}'.format(repr(e)))
    pass

    # ##### Merging Catagories
    try:
        print('\n >> MERGING CATEGORIES...')
        key = 'categories'
        indexed_key = 'category_id'
        molajson, newjson = merge_keys(molajson, newjson, keys_struct_d, dataset_id=dataset_id, key=key,
                                       indexed_key=indexed_key)
    except Exception as e:
        print('#WARNING: Problem in Merging: {}'.format(repr(e)))
        pass

    # ##### Merging videos (TAO; before merging images and tracks in TAO, because of video_id)
    try:
        print('\n >> MERGING VIDEOS...')
        key = 'videos'
        indexed_key = 'video_id'
        molajson, newjson = merge_keys(molajson, newjson, keys_struct_d, key=key, dataset_id=dataset_id,
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
        molajson, newjson = merge_keys(molajson, newjson, keys_struct_d, key=key, dataset_id=dataset_id,
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
        molajson, newjson = merge_keys(molajson, newjson, keys_struct_d, dataset_id=dataset_id, key=key,
                                       indexed_key=indexed_key,
                                       root_dir=root_dir, dir_key=dir_key)
    except Exception as e:
        print('#WARNING: Problem in Merging: {}'.format(repr(e)))
        pass

    # ##### Merge segment_info (COCO;)
    try:
        print('\n >> MERGING SEGMENT_INFO...')
        key = 'segment_info'
        indexed_key = None
        molajson, newjson = merge_keys(molajson, newjson, keys_struct_d, dataset_id=dataset_id, key=key,
                                       indexed_key=indexed_key,
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
        molajson, newjson = merge_keys(molajson, newjson, keys_struct_d, dataset_id=dataset_id, key=key,
                                       indexed_key=indexed_key)
    except Exception as e:
        print('Problem in Merging licences: {}'.format(repr(e)))
        pass

    # SAVE JSON
    print('\n >> SAVING...')
    with open(molajsonfile, 'w') as f:
        json.dump(molajson, f)
    print("JSON SAVED : {}".format(molajsonfile))
