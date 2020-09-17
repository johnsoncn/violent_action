import json
import os
import argparse
import datetime
from joblib import Parallel, delayed
import multiprocessing

INFO = {
    "description": "Mixed Dataset",
    "url": "",
    "version": "OpenImages+COCO",
    "year": 2019,
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}

LICENSES = [
    {
        "id": 1,
        "name": "Creative Commons Attribution 4.0 International license by the Allen Institute for Cell Science",
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
    }
]

parser = argparse.ArgumentParser()
parser.add_argument('--filenames', type=str, required=True, default='',
                    help='path to sampled filenames file')
args = parser.parse_args()

SELECTED_IMAGEPATHS = None

filenames = json.load(open(args.filenames))
if 'filenames' in filenames:
    SELECTED_IMAGEPATHS = filenames['filenames'] if filenames is not None else None

if 'imageids' in filenames:
    SELECTED_IMAGEPATHS = filenames['imageids']

print("SELECTED {} images".format(len(SELECTED_IMAGEPATHS)))


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def main():
    coco_json = json.load(open('instances_train2017.json')))
    print("Loading OpenImages bbox.json")
    openimages_json = json.load(open(os.path.join('train-annotations-bbox.json')))

    updated_coco_categories = coco_json['categories']
    for c in updated_coco_categories:
        c['id'] += 700

    CATEGORIES = openimages_json['categories'] + updated_coco_categories
    IMAGES = []
    IMG_IDS = {}
    ANNOTATIONS = []

    num_cores = multiprocessing.cpu_count()
    N = len(coco_json['images']) // num_cores
    print("Parallel num_cores=", num_cores, "each chunk is size {}".format(N))

    list_chunks = list(chunks(coco_json['images'], N))
    list_images = Parallel(n_jobs=num_cores)(delayed(process_coco_image_chunk)(chunk) for chunk in list_chunks)

    for img_lst in list_images:
        IMAGES.extend(img_lst)

    IMG_IDS = {img['id'] for img in IMAGES}
    N = len(coco_json['annotations']) // num_cores
    list_chunks = list(chunks(coco_json['annotations'], N))
    list_anns = Parallel(n_jobs=num_cores)(delayed(process_coco_ann_chunk)(chunk, IMG_IDS) for chunk in list_chunks)

    for ann_lst in list_anns:
        ANNOTATIONS.extend(ann_lst)

    N = len(openimages_json['images']) // num_cores
    list_chunks = list(chunks(openimages_json['images'], N))
    list_images = Parallel(n_jobs=num_cores)(delayed(process_openimages_image_chunk)(chunk) for chunk in list_chunks)

    for img_lst in list_images:
        IMAGES.extend(img_lst)

    IMG_IDS = {img['id'] for img in IMAGES}
    N = len(openimages_json['annotations']) // num_cores
    list_chunks = list(chunks(openimages_json['annotations'], N))
    list_anns = Parallel(n_jobs=num_cores)(
        delayed(process_openimages_ann_chunk)(chunk, IMG_IDS) for chunk in list_chunks)

    for ann_lst in list_anns:
        ANNOTATIONS.extend(ann_lst)

    output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": IMAGES,
        "annotations": ANNOTATIONS
    }

    # print some stats
    print("Number of Images: {}".format(len(IMAGES)))
    print("Number of Categories: {}".format(len(CATEGORIES)))
    print("Number of Annotations: {}".format(len(ANNOTATIONS)))

    with open('{}-{}-bbox.json'.format(os.path.basename(args.filenames), len(IMAGES)), 'w') as f:
        json.dump(output, f)

    print("Done!")


def process_coco_image_chunk(image_chunk):
    images = []
    for i, img in enumerate(image_chunk):
        if 'coco/{}'.format(img['file_name']) in SELECTED_IMAGEPATHS:
            img['file_name'] = 'coco/{}'.format(img['file_name'])
            if type(img['id']) is int:
                img['id'] = str(img['id'])
            images.append(img)

    return images


def process_openimages_image_chunk(image_chunk):
    images = []
    for i, img in enumerate(image_chunk):
        if 'openimages/{}'.format(img['file_name']) in SELECTED_IMAGEPATHS:
            img['file_name'] = 'openimages/{}'.format(img['file_name'])
            images.append(img)

    return images


def process_coco_ann_chunk(ann_chunk, image_ids):
    annotations = []
    for i, ann in enumerate(ann_chunk):
        if type(ann['image_id']) is int:
            ann['image_id'] = str(ann['image_id'])
        if ann['image_id'] in image_ids:
            ann['category_id'] += 700
            ann['id'] += 100000000
            annotations.append(ann)

    return annotations


def process_openimages_ann_chunk(ann_chunk, image_ids):
    annotations = []
    for i, ann in enumerate(ann_chunk):
        if ann['image_id'] in image_ids:
            annotations.append(ann)

    return annotations

if __name__ == "__main__":
    main()