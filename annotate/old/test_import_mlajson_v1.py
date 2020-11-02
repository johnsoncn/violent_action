import json
import numpy as np
import tqdm

rdir=''
mlabjson =  json.load(open(rdir+'mlab.json'))

for k in mlabjson:
    print(k, len(mlabjson[k]))

### mlabjson data format
'''
mlabjson = {
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
'''

### import ids
#### #NOTE: work with ids and index so you can use numpy for faster operations

# "datasets" name and id
dset_l=[]
dset_l_id=[]
for d in mlabjson['datasets']:
    dset_l.append(d['name'])
    dset_l_id.append(d['id'])
print(dset_l, dset_l_id)

# "categories" name and id
cat_l=[]
cat_l_id=[]
cat_l_dset=[]
for c in mlabjson['categories']:
    cat_l.append(c['name'])
    cat_l_id.append(c['id'])
    cat_l_dset.append(dset_l[c['dataset']-1]) # dset_l index is same as id-1
#print(cat_l_id)

# "images" filepath and id
img_l=[]
img_l_id=[]
for c in mlabjson['images']:
    img_l.append(c['file_name'])
    img_l_id.append(c['id'])

# "annotations" category_id, image_id, bbox, and dataset
ann_catid=[]
ann_imgid=[]
ann_bbox=[]
ann_dset=[]
for an in tqdm(mlabjson['annotations']):
    try:
        ann_catid.append(an['category_id'])
        ann_imgid.append(an['image_id'])
        ann_bbox.append(an['bbox'])
        ann_dset.append(an['dataset'])
    except:
        pass

if __name__=='__main__':
    #Example: get duplicate categories
    duplicates_l = []
    duplicates_l_catid = []
    duplicates_l_catdset = []
    duplicates_l=list(set([x for x in cat_l if cat_l.count(x) > 1])) # duplicates l
    for duplicate in tqdm(duplicates_l):
        idx_mask = [name == duplicate for name in cat_l] #mask of index of duplicate
        catids = np.array(cat_l_id)[idx_mask] #duplicate catids
        catdsets = np.array(cat_l_dset)[idx_mask] #duplicate catdset
        duplicates_l_catid.append(catids.tolist())
        duplicates_l_catdset.append(catdsets.tolist())
    print(duplicates_l)
    print(duplicates_l_catid)
    print(duplicates_l_catdset)

    # get annotations of duplicates
    ann_catid_np = np.array(ann_catid)
    ann_imgid_np = np.array(ann_imgid)
    ann_bbox_np = np.array(ann_bbox)
    ann_dset_np = np.array(ann_dset)
    duplicates_l_imgid = []
    duplicates_l_bbox = []
    duplicates_l_dset = []
    for catids in tqdm(duplicates_l_catid):
        l_imgid = []
        l_bbox = []
        l_dset = []
        for catid in catids:
            ann_idx = np.where(ann_catid_np == catid)[0].tolist()  # annotation index of ids
            l_imgid.append(ann_imgid_np[ann_idx].tolist())
            l_bbox.append(ann_bbox_np[ann_idx].tolist())
            l_dset.append(ann_dset_np[ann_idx].tolist())
        duplicates_l_imgid.append(l_imgid)
        duplicates_l_bbox.append(l_bbox)
        duplicates_l_dset.append(l_dset)

    print(duplicates_l_imgid)
    print(duplicates_l_bbox)
    print(duplicates_l_dset)