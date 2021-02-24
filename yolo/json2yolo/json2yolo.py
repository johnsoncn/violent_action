import json
from shutil import copyfile

import os
import tqdm

import cv2
import pandas as pd
#from PIL import Image

import argparse

from utils import *


# Convert Labelbox JSON file into YOLO-format labels ---------------------------
def convert_labelbox_json(name, file):
    # Create folders
    path = make_folders()

    # Import json
    with open(file) as f:
        data = json.load(f)

    # Write images and shapes
    name = 'out' + os.sep + name
    file_id, file_name, width, height = [], [], [], []
    for i, x in enumerate(tqdm(data['images'], desc='Files and Shapes')):
        file_id.append(x['id'])
        file_name.append('IMG_' + x['file_name'].split('IMG_')[-1])
        width.append(x['width'])
        height.append(x['height'])

        # filename
        with open(name + '.txt', 'a') as file:
            file.write('%s\n' % file_name[i])

        # shapes
        with open(name + '.shapes', 'a') as file:
            file.write('%g, %g\n' % (x['width'], x['height']))

    # Write *.names file
    for x in tqdm(data['categories'], desc='Names'):
        with open(name + '.names', 'a') as file:
            file.write('%s\n' % x['name'])

    # Write labels file
    for x in tqdm(data['annotations'], desc='Annotations'):
        i = file_id.index(x['image_id'])  # image index
        label_name = Path(file_name[i]).stem + '.txt'

        # The Labelbox bounding box format is [top left x, top left y, width, height]
        box = np.array(x['bbox'], dtype=np.float64)
        box[:2] += box[2:] / 2  # xy top-left corner to center
        box[[0, 2]] /= width[i]  # normalize x
        box[[1, 3]] /= height[i]  # normalize y

        if (box[2] > 0.) and (box[3] > 0.):  # if w > 0 and h > 0
            with open('out/labels/' + label_name, 'a') as file:
                file.write('%g %.6f %.6f %.6f %.6f\n' % (x['category_id'] - 1, *box))

    # Split data into train, test, and validate files
    split_files(name, file_name)
    print('Done. Output saved to %s' % (os.getcwd() + os.sep + path))


# Convert INFOLKS JSON file into YOLO-format labels ----------------------------
def convert_infolks_json(name, files, img_path):
    # Create folders
    path = make_folders()

    # Import json
    data = []
    for file in glob.glob(files):
        with open(file) as f:
            jdata = json.load(f)
            jdata['json_file'] = file
            data.append(jdata)

    # Write images and shapes
    name = path + os.sep + name
    file_id, file_name, wh, cat = [], [], [], []
    for x in tqdm(data, desc='Files and Shapes'):
        f = glob.glob(img_path + Path(x['json_file']).stem + '.*')[0]
        file_name.append(f)
        wh.append(exif_size(Image.open(f)))  # (width, height)
        cat.extend(a['classTitle'].lower() for a in x['output']['objects'])  # categories

        # filename
        with open(name + '.txt', 'a') as file:
            file.write('%s\n' % f)

    # Write *.names file
    names = sorted(np.unique(cat))
    # names.pop(names.index('Missing product'))  # remove
    with open(name + '.names', 'a') as file:
        [file.write('%s\n' % a) for a in names]

    # Write labels file
    for i, x in enumerate(tqdm(data, desc='Annotations')):
        label_name = Path(file_name[i]).stem + '.txt'

        with open(path + '/labels/' + label_name, 'a') as file:
            for a in x['output']['objects']:
                # if a['classTitle'] == 'Missing product':
                #    continue  # skip

                category_id = names.index(a['classTitle'].lower())

                # The INFOLKS bounding box format is [x-min, y-min, x-max, y-max]
                box = np.array(a['points']['exterior'], dtype=np.float32).ravel()
                box[[0, 2]] /= wh[i][0]  # normalize x by width
                box[[1, 3]] /= wh[i][1]  # normalize y by height
                box = [box[[0, 2]].mean(), box[[1, 3]].mean(), box[2] - box[0], box[3] - box[1]]  # xywh
                if (box[2] > 0.) and (box[3] > 0.):  # if w > 0 and h > 0
                    file.write('%g %.6f %.6f %.6f %.6f\n' % (category_id, *box))

    # Split data into train, test, and validate files
    split_files(name, file_name)
    write_data_data(name + '.data', nc=len(names))
    print('Done. Output saved to %s' % (os.getcwd() + os.sep + path))


# Convert vott JSON file into YOLO-format labels -------------------------------
def convert_vott_json(name, files, img_path):
    # Create folders
    path = make_folders()
    name = path + os.sep + name

    # Import json
    data = []
    for file in glob.glob(files):
        with open(file) as f:
            jdata = json.load(f)
            jdata['json_file'] = file
            data.append(jdata)

    # Get all categories
    file_name, wh, cat = [], [], []
    for i, x in enumerate(tqdm(data, desc='Files and Shapes')):
        try:
            cat.extend(a['tags'][0] for a in x['regions'])  # categories
        except:
            pass

    # Write *.names file
    names = sorted(pd.unique(cat))
    with open(name + '.names', 'a') as file:
        [file.write('%s\n' % a) for a in names]

    # Write labels file
    n1, n2 = 0, 0
    missing_images = []
    for i, x in enumerate(tqdm(data, desc='Annotations')):

        f = glob.glob(img_path + x['asset']['name'] + '.jpg')
        if len(f):
            f = f[0]
            file_name.append(f)
            wh = exif_size(Image.open(f))  # (width, height)

            n1 += 1
            if (len(f) > 0) and (wh[0] > 0) and (wh[1] > 0):
                n2 += 1

                # append filename to list
                with open(name + '.txt', 'a') as file:
                    file.write('%s\n' % f)

                # write labelsfile
                label_name = Path(f).stem + '.txt'
                with open(path + '/labels/' + label_name, 'a') as file:
                    for a in x['regions']:
                        category_id = names.index(a['tags'][0])

                        # The INFOLKS bounding box format is [x-min, y-min, x-max, y-max]
                        box = a['boundingBox']
                        box = np.array([box['left'], box['top'], box['width'], box['height']]).ravel()
                        box[[0, 2]] /= wh[0]  # normalize x by width
                        box[[1, 3]] /= wh[1]  # normalize y by height
                        box = [box[0] + box[2] / 2, box[1] + box[3] / 2, box[2], box[3]]  # xywh

                        if (box[2] > 0.) and (box[3] > 0.):  # if w > 0 and h > 0
                            file.write('%g %.6f %.6f %.6f %.6f\n' % (category_id, *box))
        else:
            missing_images.append(x['asset']['name'])

    print('Attempted %g json imports, found %g images, imported %g annotations successfully' % (i, n1, n2))
    if len(missing_images):
        print('WARNING, missing images:', missing_images)

    # Split data into train, test, and validate files
    split_files(name, file_name)
    print('Done. Output saved to %s' % (os.getcwd() + os.sep + path))


# Convert ath JSON file into YOLO-format labels --------------------------------
def convert_ath_json(json_dir):  # dir contains json annotations and images
    # Create folders
    dir = make_folders()  # output directory

    jsons = []
    for dirpath, dirnames, filenames in os.walk(json_dir):
        for filename in [f for f in filenames if f.lower().endswith('.json')]:
            jsons.append(os.path.join(dirpath, filename))

    # Import json
    n1, n2, n3 = 0, 0, 0
    missing_images, file_name = [], []
    for json_file in sorted(jsons):
        with open(json_file) as f:
            data = json.load(f)

        # # Get classes
        # try:
        #     classes = list(data['_via_attributes']['region']['class']['options'].values())  # classes
        # except:
        #     classes = list(data['_via_attributes']['region']['Class']['options'].values())  # classes

        # # Write *.names file
        # names = pd.unique(classes)  # preserves sort order
        # with open(dir + 'data.names', 'w') as f:
        #     [f.write('%s\n' % a) for a in names]

        # Write labels file
        for i, x in enumerate(tqdm(data['_via_img_metadata'].values(), desc='Processing %s' % json_file)):

            image_file = str(Path(json_file).parent / x['filename'])
            f = glob.glob(image_file)  # image file
            if len(f):
                f = f[0]
                file_name.append(f)
                wh = exif_size(Image.open(f))  # (width, height)

                n1 += 1  # all images
                if len(f) > 0 and wh[0] > 0 and wh[1] > 0:
                    label_file = dir + 'labels/' + Path(f).stem + '.txt'

                    nlabels = 0
                    try:
                        with open(label_file, 'a') as file:  # write labelsfile
                            for a in x['regions']:
                                # try:
                                #     category_id = int(a['region_attributes']['class'])
                                # except:
                                #     category_id = int(a['region_attributes']['Class'])
                                category_id = 0  # single-class

                                # bounding box format is [x-min, y-min, x-max, y-max]
                                box = a['shape_attributes']
                                box = np.array([box['x'], box['y'], box['width'], box['height']],
                                               dtype=np.float32).ravel()
                                box[[0, 2]] /= wh[0]  # normalize x by width
                                box[[1, 3]] /= wh[1]  # normalize y by height
                                box = [box[0] + box[2] / 2, box[1] + box[3] / 2, box[2],
                                       box[3]]  # xywh (left-top to center x-y)

                                if box[2] > 0. and box[3] > 0.:  # if w > 0 and h > 0
                                    file.write('%g %.6f %.6f %.6f %.6f\n' % (category_id, *box))
                                    n3 += 1
                                    nlabels += 1

                        if nlabels == 0:  # remove non-labelled images from dataset
                            os.system('rm %s' % label_file)
                            # print('no labels for %s' % f)
                            continue  # next file

                        # write image
                        img_size = 4096  # resize to maximum
                        img = cv2.imread(f)  # BGR
                        assert img is not None, 'Image Not Found ' + f
                        r = img_size / max(img.shape)  # size ratio
                        if r < 1:  # downsize if necessary
                            h, w, _ = img.shape
                            img = cv2.resize(img, (int(w * r), int(h * r)), interpolation=cv2.INTER_AREA)

                        ifile = dir + 'images/' + Path(f).name
                        if cv2.imwrite(ifile, img):  # if success append image to list
                            with open(dir + 'data.txt', 'a') as file:
                                file.write('%s\n' % ifile)
                            n2 += 1  # correct images

                    except:
                        os.system('rm %s' % label_file)
                        print('problem with %s' % f)

            else:
                missing_images.append(image_file)

    nm = len(missing_images)  # number missing
    print('\nFound %g JSONs with %g labels over %g images. Found %g images, labelled %g images successfully' %
          (len(jsons), n3, n1, n1 - nm, n2))
    if len(missing_images):
        print('WARNING, missing images:', missing_images)

    # Write *.names file
    names = ['knife']  # preserves sort order
    with open(dir + 'data.names', 'w') as f:
        [f.write('%s\n' % a) for a in names]

    # Split data into train, test, and validate files
    split_rows_simple(dir + 'data.txt')
    write_data_data(dir + 'data.data', nc=1)
    print('Done. Output saved to %s' % Path(dir).absolute())


# Convert coco JSON file into YOLO-format labels --------------------------------
def convert_coco_json(json_dir='../coco/annotations/'):
    dir = make_folders(path='out/')  # output directory
    jsons = glob.glob(json_dir + '*.json')
    coco80 = coco91_to_coco80_class()  # converts 80-index (val2014) to 91-index (paper)

    # Import json
    for json_file in sorted(jsons):
        fn = 'out/labels/%s/' % Path(json_file).stem.replace('instances_', '')  # folder name
        os.mkdir(fn)
        with open(json_file) as f:
            data = json.load(f)

        # Create image dict
        images = {'%g' % x['id']: x for x in data['images']}

        # Write labels file
        for x in tqdm(data['annotations'], desc='Annotations %s' % json_file):
            if x['iscrowd']:
                continue

            img = images['%g' % x['image_id']]
            h, w, f = img['height'], img['width'], img['file_name']

            # The Labelbox bounding box format is [top left x, top left y, width, height]
            box = np.array(x['bbox'], dtype=np.float64)
            box[:2] += box[2:] / 2  # xy top-left corner to center
            box[[0, 2]] /= w  # normalize x
            box[[1, 3]] /= h  # normalize y

            if (box[2] > 0.) and (box[3] > 0.):  # if w > 0 and h > 0
                with open(fn + Path(f).stem + '.txt', 'a') as file:
                    file.write('%g %.6f %.6f %.6f %.6f\n' % (coco80[x['category_id'] - 1], *box))


# Convert tao JSON file into YOLO-format labels --------------------------------
def convert_tao_json(json_dir='../tao/annotations/', outdir='out/', copy_images=False, img_number=None):
    dir = make_folders(path=outdir)  # output directory
    jsons = glob.glob(json_dir + '*.json')
    # tao_cat_idx = [x for x in range(833)]  # object catagories index - 833 cat = 488 LVIS + 345 free-form

    # Import json
    for json_file in sorted(jsons):
        labeldir_path = 'labels/%s/' % Path(json_file).stem  # folder name (train, val, test) remove other info
        imgdir_path = 'images/%s/' % Path(json_file).stem  # folder name (train, val, test) remove other info
        dir_label = outdir + labeldir_path
        os.mkdir(dir_label)
        dir_img = outdir + imgdir_path
        os.mkdir(dir_img)
        with open(json_file) as f:
            data = json.load(f)

        # image array
        img_a = []
        # image counter
        img_counter = 0

        # Create image dict
        images = {'%g' % x['id']: x for x in data['images']}

        # Write labels file
        for x in tqdm(data['annotations'], desc='Annotations %s' % json_file):
            if x['iscrowd']:
                continue

            img = images['%g' % x['image_id']]
            h, w, f = img['height'], img['width'], img['file_name']
            _, img_ext = os.path.splitext(f)
            img_fn = Path(f).stem
            img_new_fn = '%g' % (x['image_id'])  # WARNING - it needs to be the an id

            # The Labelbox bounding box format is [top left x, top left y, width, height]
            box = np.array(x['bbox'], dtype=np.float64)
            box[:2] += box[2:] / 2  # xy top-left corner to center
            box[[0, 2]] /= w  # normalize x
            box[[1, 3]] /= h  # normalize y

            if (box[2] > 0.) and (box[3] > 0.):  # if w > 0 and h > 0
                # write labels
                with open(dir_label + img_new_fn + '.txt', 'a') as file:
                    file.write('%g %.6f %.6f %.6f %.6f\n' % (x['category_id'] - 1, *box))
                # write images
                if not (any([True for im in img_a if im.find(str(x['image_id'])) > -1])):  # don't repeat
                    img_a.append(str(x['image_id']))  #
                    try:
                        # copy imgs
                        if copy_images:
                            tao_root_dir = os.path.dirname(os.path.dirname(json_dir))
                            src = os.path.join(tao_root_dir, 'frames', f)
                            dst = os.path.join(dir_img, img_new_fn + img_ext)
                            copyfile(src, dst)  # missing files
                        # save img path (if image copied)
                        with open(outdir + Path(json_file).stem + '.txt', 'a+') as file:
                            img_new_path = '%s\n' % (imgdir_path + img_new_fn + img_ext)
                            file.write(img_new_path)
                        img_counter += 1
                    except:
                        img_a[-1] = img_a[-1] + '_MISSING'
                        print("missing : {}".format(os.path.join(tao_root_dir, 'frames', f)))
                        print(img_a[-1])

            # STOP
            if img_number and img_counter >= img_number: break


# Convert motionLab JSON file into YOLO-format labels --------------------------------
def convert_mola_json(datasets_root_dir=None, json_dir='../mola/annotations/', outdir='out/', copy_images=True, only_labels=False, img_number=None):
    '''
    #WARNING copy_images="1" #1=copy images from original datasets, easiest way to create yolo format dataset
    # "0"= don't copy images use the ones from original #WARNING labels need to be organized
    » copy labels to the same root dir as "images" in original dataset and also have the same path as the images
    (yolov5 only works with "images" folder, so in original dataset one needs to change to "images":
    yolov5.utils.datasets.LoadImagesAndLabels
    # Define label paths as a function of image paths
    sa, sb = os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep  # /images/, /labels/ substrings
    '''
    dir = make_folders(path=outdir)  # output directory
    jsons = glob.glob(json_dir + '*.json')
    # tao_cat_idx = [x for x in range(833)]  # object catagories index - 833 cat = 488 LVIS + 345 free-form

    # Import json
    for json_file in sorted(jsons):
        labeldir_path = 'labels/%s/' % Path(json_file).stem  # folder name (train, val, test) remove other info
        imgdir_path = 'images/%s/' % Path(json_file).stem  # folder name (train, val, test) remove other info
        dir_label = outdir + labeldir_path
        os.mkdir(dir_label)
        dir_img = outdir + imgdir_path
        os.mkdir(dir_img)
        with open(json_file) as f:
            data = json.load(f)

        # image array
        img_a = []
        save_img_a= []
        # label dict
        lb_d = {}
        # image counter
        img_counter = 0

        # Create image dict {id: image}
        images = {'%g' % x['id']: x for x in data['images']}

        # WRITE
        method="for"
        #cd yolo/json2yolo/batchs
        import time
        start=time.time()
        if method=="for":
            for x in tqdm(data['annotations'], desc='Annotations %s' % json_file):
                try:
                    if x['iscrowd']: continue
                except:
                    print('missing "iscrowd" key')

                img = images['%g' % x['image_id']]
                h, w, f = img['height'], img['width'], img['file_name']
                _, img_ext = os.path.splitext(f)
                img_fn = Path(f).stem
                img_new_fn = img_fn #original filename - if the images are not copied mantain original path
                if copy_images: img_new_fn = '%g' % (x['image_id'])  # WARNING - it needs to be the an id
                if only_labels: img_new_fn = '%g' % (x['image_id'])  # WARNING - it needs to be the an id

                # The Labelbox bounding box format is [top left x, top left y, width, height]
                box = np.array(x['bbox'], dtype=np.float64)
                box[:2] += box[2:] / 2  # xy top-left corner to center
                box[[0, 2]] /= w  # normalize x
                box[[1, 3]] /= h  # normalize y

                if (box[2] > 0.) and (box[3] > 0.):  # if w > 0 and h > 0
                    # write images - 1st because if copy fails the
                    if not str(x['image_id']) in img_a: # don't repeat write images #if not (any([True for im in img_a if im.find(str(x['image_id'])) > -1])):
                        img_a.append(str(x['image_id']))  # all images
                        # copy imgs
                        if copy_images or only_labels:
                            src = os.path.join(datasets_root_dir, f)
                            dst = os.path.join(dir_img, img_new_fn + img_ext)
                            try:
                                if copy_images: copyfile(src, dst)  # missing files
                                if only_labels and not os.path.exists(src): raise
                            except:
                                img_a[-1] = img_a[-1] + '_MISSING'
                                print("\n>> missing : {}".format(Path(os.path.join(datasets_root_dir, f))))
                                print(img_a[-1])
                                continue #if image missing from dataset when copy dont write nothing more
                        # save img path
                        with open(outdir + Path(json_file).stem + '.txt', 'a+') as file:
                            img_new_path = '%s\n' % Path(os.path.join(datasets_root_dir, f)) #original path
                            if copy_images or only_labels: img_new_path = '%s\n' % Path(outdir+imgdir_path + img_new_fn + img_ext) #new path
                            file.write(img_new_path)
                        img_counter += 1
                        lb_d[str(x['image_id'])]=[]
                        save_img_a.append(str(x['image_id']))  # only images that are saved
                        print(str(x['image_id']))
                    # write labels
                    if  (str(x['image_id']) in save_img_a) and (not str(box) in lb_d[str(x['image_id'])]):  # don't write same labels (even if annotations are duplicated)
                        lb_d[str(x['image_id'])].append(str(box))
                        if copy_images or only_labels:
                            with open(dir_label + img_new_fn + '.txt', 'a') as file:
                                file.write('%g %.6f %.6f %.6f %.6f\n' % (x['category_id'] - 1, *box))
                        else:
                            with open(dir_label + os.path.splitext(f)[0]+ '.txt', 'a+') as file:
                                file.write('%g %.6f %.6f %.6f %.6f\n' % (x['category_id'] - 1, *box))

                # STOP
                if img_number and img_counter >= img_number: break
            stop = time.time()

        if method == "parfor":
            #for x in tqdm(data['annotations'], desc='Annotations %s' % json_file):
            from parfor import parfor
            an_data=data['annotations']
            if img_number: an_data=an_data[:img_number]
            @parfor(an_data,(img_number, img_a, save_img_a, lb_d,img_counter, images), nP=4)
            def run_ann(x, img_number, img_a, save_img_a, lb_d, img_counter, images):
                try:
                    if x['iscrowd']: return img_number, img_a, save_img_a, lb_d, img_counter, images #continue
                except:
                    #print('missing "iscrowd" key')
                    pass

                img = images['%g' % x['image_id']]
                h, w, f = img['height'], img['width'], img['file_name']
                _, img_ext = os.path.splitext(f)
                img_fn = Path(f).stem
                img_new_fn = img_fn #original filename - if the images are not copied mantain original path
                if copy_images: img_new_fn = '%g' % (x['image_id'])  # WARNING - it needs to be the an id

                # The Labelbox bounding box format is [top left x, top left y, width, height]
                box = np.array(x['bbox'], dtype=np.float64)
                box[:2] += box[2:] / 2  # xy top-left corner to center
                box[[0, 2]] /= w  # normalize x
                box[[1, 3]] /= h  # normalize y

                if (box[2] > 0.) and (box[3] > 0.):  # if w > 0 and h > 0
                    # write images - 1st because if copy fails the
                    if not str(x['image_id']) in img_a: # don't repeat write images #if not (any([True for im in img_a if im.find(str(x['image_id'])) > -1])):
                        img_a.append(str(x['image_id']))  # all images
                        # copy imgs
                        if copy_images:
                            src = os.path.join(datasets_root_dir, f)
                            dst = os.path.join(dir_img, img_new_fn + img_ext)
                            try:
                                copyfile(src, dst)  # missing files
                            except:
                                img_a[-1] = img_a[-1] + '_MISSING'
                                print("\n>> missing : {}".format(Path(os.path.join(datasets_root_dir, f))))
                                print(img_a[-1])
                                return img_number, img_a, save_img_a, lb_d, img_counter, images#continue #if image missing from dataset when copy dont write nothing more
                        # save img path (if image copied)
                        with open(outdir + Path(json_file).stem + '.txt', 'a+') as file:
                            img_new_path = '%s\n' % Path(os.path.join(datasets_root_dir, f)) #original path
                            if copy_images: img_new_path = '%s\n' % Path(outdir+imgdir_path + img_new_fn + img_ext) #new path
                            file.write(img_new_path)
                        img_counter += 1
                        lb_d[str(x['image_id'])]=[]
                        save_img_a.append(str(x['image_id']))  # only images that are saved
                        #print(str(x['image_id']))
                    # write labels
                    if  (str(x['image_id']) in save_img_a) and (not str(box) in lb_d[str(x['image_id'])]):  # don't write same labels (even if annotations are duplicated)
                        lb_d[str(x['image_id'])].append(str(box))
                        if copy_images:
                            with open(dir_label + img_new_fn + '.txt', 'a') as file:
                                file.write('%g %.6f %.6f %.6f %.6f\n' % (x['category_id'] - 1, *box))
                        else:
                            with open(dir_label + os.path.splitext(f)[0]+ '.txt', 'a') as file:
                                file.write('%g %.6f %.6f %.6f %.6f\n' % (x['category_id'] - 1, *box))

                # STOP
                if img_number and img_counter >= img_number: raise #break
            stop = time.time()

        if method == "multifor": #not working
            from multiprocessing import Pool
            def run_ann(x, img_number=img_number, img_a=img_a, save_img_a=save_img_a, lb_d=lb_d, img_counter=img_counter, images=images):
                try:
                    if x['iscrowd']: return img_number, img_a, save_img_a, lb_d, img_counter, images  # continue
                except:
                    print('missing "iscrowd" key')

                img = images['%g' % x['image_id']]
                h, w, f = img['height'], img['width'], img['file_name']
                _, img_ext = os.path.splitext(f)
                img_fn = Path(f).stem
                img_new_fn = img_fn  # original filename - if the images are not copied mantain original path
                if copy_images: img_new_fn = '%g' % (x['image_id'])  # WARNING - it needs to be the an id

                # The Labelbox bounding box format is [top left x, top left y, width, height]
                box = np.array(x['bbox'], dtype=np.float64)
                box[:2] += box[2:] / 2  # xy top-left corner to center
                box[[0, 2]] /= w  # normalize x
                box[[1, 3]] /= h  # normalize y

                if (box[2] > 0.) and (box[3] > 0.):  # if w > 0 and h > 0
                    # write images - 1st because if copy fails the
                    if not str(x[
                                   'image_id']) in img_a:  # don't repeat write images #if not (any([True for im in img_a if im.find(str(x['image_id'])) > -1])):
                        img_a.append(str(x['image_id']))  # all images
                        # copy imgs
                        if copy_images:
                            src = os.path.join(datasets_root_dir, f)
                            dst = os.path.join(dir_img, img_new_fn + img_ext)
                            try:
                                copyfile(src, dst)  # missing files
                            except:
                                img_a[-1] = img_a[-1] + '_MISSING'
                                print("\n>> missing : {}".format(Path(os.path.join(datasets_root_dir, f))))
                                print(img_a[-1])
                                return img_number, img_a, save_img_a, lb_d, img_counter, images  # continue #if image missing from dataset when copy dont write nothing more
                        # save img path (if image copied)
                        with open(outdir + Path(json_file).stem + '.txt', 'a+') as file:
                            img_new_path = '%s\n' % Path(os.path.join(datasets_root_dir, f))  # original path
                            if copy_images: img_new_path = '%s\n' % Path(
                                outdir + imgdir_path + img_new_fn + img_ext)  # new path
                            file.write(img_new_path)
                        img_counter += 1
                        lb_d[str(x['image_id'])] = []
                        save_img_a.append(str(x['image_id']))  # only images that are saved
                        # print(str(x['image_id']))
                    # write labels
                    if (str(x['image_id']) in save_img_a) and (not str(box) in lb_d[
                        str(x['image_id'])]):  # don't write same labels (even if annotations are duplicated)
                        lb_d[str(x['image_id'])].append(str(box))
                        if copy_images:
                            with open(dir_label + img_new_fn + '.txt', 'a') as file:
                                file.write('%g %.6f %.6f %.6f %.6f\n' % (x['category_id'] - 1, *box))
                        else:
                            with open(dir_label + os.path.splitext(f)[0] + '.txt', 'a') as file:
                                file.write('%g %.6f %.6f %.6f %.6f\n' % (x['category_id'] - 1, *box))

                # STOP
                if img_number and img_counter >= img_number: raise  # break
            pool = Pool()  # Create a multiprocessing Pool
            pool.map(run_ann, data['annotations'])  # process data_inputs iterable with pool
            stop = time.time()

        elapsed=stop-start
        print(elapsed)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='mola', help='source type to convert')
    parser.add_argument('--datasets_root_dir', type=str,
                        default=None,
                        help='root dir for all the datasets')
    parser.add_argument('--json_dir', type=str,
                        default='D:/external_datasets/MOLA/annotations/train_mlab_mix_aggressive/',
                        help='json annotations input path')
    parser.add_argument('--outdir', type=str,
                        default="D:/external_datasets/MOLA/yoloformat/mola_train/",
                        help='yoloformat dataset output path')
    parser.add_argument('--img_number', type=int, default=None, help='number of images to convert. None=convert all')
    parser.add_argument('--copy_images', type=int, default=0, help='copy images to folder /images and add new path to .txt . If 0 no image is copied and .txt has the original paths')
    parser.add_argument('--only_labels', type=int, default=0,
                        help='write labels only as if you were copying images to folder /images and add new path to .txt .')

    opt = parser.parse_args()

    source = opt.source
    datasets_root_dir = opt.datasets_root_dir
    json_dir = opt.json_dir
    outdir = opt.outdir
    img_number = opt.img_number
    copy_images = False
    if opt.copy_images == 1: copy_images = True
    only_labels = False
    if opt.only_labels == 1: only_labels = True
    print('\n>>' + str(opt))

    if not datasets_root_dir: raise RuntimeError('Select datasets_root_dir')

    if source == 'mola':
        # CREATE LABELS and IMAGES FOLDER
        convert_mola_json(datasets_root_dir=datasets_root_dir, json_dir=json_dir,
                          outdir=outdir, img_number=img_number, copy_images=copy_images, only_labels=only_labels)

    elif source == 'labelbox':  # Labelbox https://labelbox.com/
        convert_labelbox_json(name='supermarket2',
                              file='../supermarket2/export-coco.json')

    elif source == 'infolks':  # Infolks https://infolks.info/
        convert_infolks_json(name='out',
                             files='../data/sm4/json/*.json',
                             img_path='../data/sm4/images/')

    elif source == 'vott':  # VoTT https://github.com/microsoft/VoTT
        convert_vott_json(name='data',
                          files='../../Downloads/athena_day/20190715/*.json',
                          img_path='../../Downloads/athena_day/20190715/')  # images folder

    elif source == 'ath':  # ath format
        convert_ath_json(json_dir='../../Downloads/athena/')  # images folder

    elif source == 'coco':
        convert_coco_json(json_dir=json_dir)

    elif source == 'tao':
        # CREATE LABELS and IMAGES FOLDER
        convert = True
        if convert:
            convert_tao_json(json_dir=json_dir,
                             outdir=outdir, img_number=img_number)

        # EXTRACT INFORMATION
        taofile = 'D:/external_datasets/TAO_old/TAO_DIR/annotations/train.json'  # 'D:/external_datasets/TAO/TAO_DIR/annotations'
        toolkit = False
        if toolkit:  # using tao toolkit
            import tao
            from tao.toolkit.tao import tao

            TAO = tao.Tao(taofile)  # class
            print("nc: {}".format(len(TAO.get_cat_ids())))
            # print("catagories: \n {}".format(TAO.load_cats(ids=None)))

        extract_cats = False
        if extract_cats:
            # TODO obj catagories extraction
            # https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
            with open(taofile, 'r') as f:
                js = json.loads(f.read())
                print(json.dumps(js['categories'], sort_keys=True, indent=4))
                # print("nc: {}".format(len(js['categories'])))
                cat_names = []
                for cat in js['categories']:
                    cat_names.append(cat['name'])
                print(cat_names)
                print(len(cat_names))
                catid = []
                for an in js['annotations']:
                    catid.append(an['category_id'])
                catid.sort()
                print(catid)
                print(len(catid))



    # zip results
    # os.system('zip -r ../coco.zip ../coco')
