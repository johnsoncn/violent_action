#!/usr/bin/env python
# coding: utf-8

# # Create INCAR and INVICON MOLA JSON
# version: 1
# 
# info:
# - Create standard MOLA JSON
# 
# author: nuno costa


"""
# ## MOLA Annotations Data Format

# If you wish to combine multiple datasets, it is often useful to convert them into a unified data format. 
# 
# Objective: 

# In[2]:

 #ANNOTATIONS FORMAT (BASED ON COCO)

 #Annotations format keys:

 { "info": None, 
   "licenses": [], TODO
   "categories": [], 
   "images": [],
   "annotations": [],
   "videos": [], TODO
   "tracks": [], TODO
   "segment_info": [], TODO
   "datasets": [{'name': 'INCAR', 'id': 1}, {'name': 'INVICON', 'id': 2}] 
 }

 #1 object definition:

 info{
     "year": int, 
     "version": str, 
     "description": str, 
     "contributor": str, 
     "url": str, 
     "date_created": datetime,
 }
 
 license{
     "id": int, 
     "name": str, 
     "url": str,
 }
 
 category{
     "id": int, 
     "name": str, 
     "supercategory": str,
 }
 
 image: {
     "id" : int,
     "video_id": int, #TODO
     "file_name" : str,
     "license" : int,
     # Redundant fields for COCO-compatibility
     "width": int,
     "height": int,
     "frame_index": int,
     "date_captured": datetime,
 
 annotation: {
     "category_id": int
     "image_id": int,
     "track_id": int,
     "bbox": [x,y,width,height],
     "area": float,
     # Redundant field for compatibility with COCO scripts
     "id": int,
     "iscrowd": 0 or 1,  (iscrowd=1) are used to label large groups of objects (e.g. a crowd of people)
     "segmentation": RLE(iscrowd=1) or [polygon](iscrowd=0), 
     
 }
 
 video: { #TODO
     "id": int,
     "name": str,
     "width" : int,
     "height" : int,
     "metadata": dict,  # Metadata about the video
 }
 
 segment{ #TODO
     "id": int, 
     "category_id": int, 
     "area": int, 
     "bbox": [x,y,width,height], 
     # Redundant field for compatibility with COCO scripts
     "iscrowd": 0 or 1,
 }
 
    
 track: { #TODO
     "id": int,
     "category_id": int,
     "video_id": int
 }

"""
# ## SETUP

import platform 
import json
import os
from tqdm import tqdm

# ## Functions
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
        "datasets": [] #[{'name': 'COCO', 'id': 1}, {'name': 'TAO', 'id': 2}]
    }
    output['info'] = {
        "description": "MOLA Dataset",
        "url": "",
        "version": "1",
        "year": 2021,
        "date_created": datetime.datetime.utcnow().isoformat(' ')
    }

    with open(file, 'w') as f:
        json.dump(output, f)
    print("JSON INITIATED : {}".format(file))


def fix_pahts(gt):
    #fix gt datasource
    paths=gt['gTruth']['DataSource']
    originalpath=paths[0]
    for p in paths:
        if p.find("gt") >-1 : 
            originalpath=p
            break
    paths = [os.path.join(*originalpath.split('\\')[:-1], p.split('\\')[-1]) if p.find("MATLAB") > -1 else p for p in paths]  #remove MATLAB BUG: 'C:\\Tools\\MATLAB\\R2020a\\examples\\symbolic\\data\\196.png'
    paths = [os.path.join(*p.split('\\')[-7:]) for p in paths] #remove original 
    gt['gTruth']['DataSource']=paths
    return gt

def import_categories(molajson, gt):
    # IMPORT categories name and id
    cat_l=[]
    cat_l_id=[]
    cat_l_dset=[]
    cat=gt['gTruth']['LabelDefinitions']
    for i,c in enumerate(tqdm(cat)):
        cat_l.append(c['Name'])
        cat_l_id.append(i+1) # id start from 1
        cat_l_dset.append(1) # dataset index
        molajson['categories'].append({'name':cat_l[i],'id':cat_l_id[i],'dataset':cat_l_dset[i]})
    # ADDITIONAL CATEGORIES: MANUAL
    name='NONVIOLENT'
    cid=len(cat_l)+1
    dset=1
    molajson['categories'].append({'name':name,'id':cid,'dataset':dset})
    cat_l.append(name)
    cat_l_id.append(cid)
    cat_l_dset.append(dset)
    print("\n>> categories:\n", molajson['categories'][:2])
    return molajson, cat_l, cat_l_id, cat_l_dset

def import_images(molajson, gt):
    # images filepath and id
    img_l=[]
    img_l_id=[]
    img=gt['gTruth']['DataSource']
    for i,im in enumerate(tqdm(img)):
        img_l.append(im)
        img_l_id.append(i+1) # id start from 1
        molajson['images'].append({'file_name':img_l[i],
                                   'id':img_l_id[i],
                                   'caption':img_l[i].split('\\')[-4], # scenario
                                   'dataset':1})
    print("\n>> images:\n", molajson['images'][:2])
    return molajson, img_l, img_l_id

def create_annotations(molajson, gt, res, cat_l, cat_l_id, cat_l_dset, img_l_id):
    # annotations category_id, image_id, bbox, and dataset
    ann_id=[]
    ann_catid=[]
    ann_imgid=[]
    ann_bbox=[]
    ann_dset=[]
    labels=gt['gTruth']['LabelData']
    for i,l in enumerate(tqdm(labels)):
        annid=i+1
        catidx=cat_l.index("VIOLENT")
        if not l["VIOLENT"]: catidx=cat_l.index("NONVIOLENT")
        catid=cat_l_id[catidx]
        dataset=cat_l_dset[catidx]
        imgidx=i
        imgid=img_l_id[imgidx]
        bbox=[0, 0, res['rgb'][0], res['rgb'][1]] # [x,y,width,height], #default RGB
        area=res['rgb'][0]*res['rgb'][1] #default RGB
        ann_id.append(annid)
        ann_catid.append(catid)
        ann_imgid.append(imgid)
        ann_bbox.append(bbox)
        ann_dset.append(dataset)
        molajson['annotations'].append({'id':annid,
                                        'category_id':catid,
                                        'image_id':imgid,
                                        'bbox': bbox,
                                        'area': area,
                                        'iscrowd': 0,
                                        'dataset':dataset})
    print("\n>> annotations:\n", molajson['annotations'][:2])
    return molajson, ann_id, ann_catid, ann_imgid, ann_bbox, ann_dset
    
if __name__ == '__main__':
    #Define root dir dependent on OS
    rdir='D:/external_datasets/MOLA/' #WARNING needs to be root datasets 
    print('OS: {}'.format(platform.platform()))
    if str(platform.platform()).upper().find('linux'.upper())>-1: rdir='/home/administrator/Z/Datasets/External Datasets/MOLA/' #'/mnt/d/external_datasets/'
    print('root dir: {}'.format(rdir))
    # define resolutions
    res={
        'rgb': [2048, 1536], #w,h
        'thermal': [640,512],
        'pointcloud': [640,576]
    }

    #INIT JSON
    molafile=rdir+'INCAR/'+'mola.json'
    init_json(file=molafile)
    molajson =  json.load(open(molafile))
    molajson['datasets']=[{'name': 'INCAR', 'id': 1}]
    with open(molafile, 'w') as f:
        json.dump(molajson, f)
    rootdir="D:/external_datasets/MOLA/INCAR/"
    #FOR LOOP
    days=os.listdir(rootdir)
    imported_cats=False
    for day in days:
        sessiondir=os.path.join(rootdir, day)
        if not os.path.isdir(sessiondir): continue #test if is a folder
        sessions=os.listdir(sessiondir)
        for session in sessions:
            scenariosdir=os.path.join(sessiondir, session)
            if not os.path.isdir(scenariosdir): continue #test if is a folder
            scenarios=os.listdir(scenariosdir)
            for scenario in scenarios:
                imgdir=os.path.join(scenariosdir, scenario)
                if not os.path.isdir(imgdir): continue #test if is a folder
                labeldir=os.path.join(imgdir,'gt') 
                #if not os.path.isdir(labeldir): continue #should exist
                filename=os.path.join(labeldir, "gt.json")
                gt=json.load(open(filename))
                #fix gt paths
                gt=fix_pahts(gt)
                #update molajson
                if not imported_cats: #only imports one time
                    molajson, cat_l, cat_l_id, cat_l_dset=import_categories(molajson, gt)
                    imported_cats=True
                molajson, img_l, img_l_id=import_images(molajson, gt)
                molajson, ann_id, ann_catid, ann_imgid, ann_bbox, ann_dset=create_annotations(molajson, gt, res, cat_l, cat_l_id, cat_l_dset, img_l_id)

    # results
    for k in molajson:
        print(k, len(molajson[k]))

    # # Save
    print('\n >> SAVING...')
    jsonfile=molafile
    with open(jsonfile, 'w') as f:
        json.dump(molajson, f)
    print("JSON SAVED : {} \n".format(jsonfile))

    #retest results
    molajson =  json.load(open(molafile))
    for k in molajson:
        print(k, len(molajson[k]))







