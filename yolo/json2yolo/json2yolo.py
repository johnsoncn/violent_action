import json
from shutil import copyfile
import os
import tqdm
import argparse

import glob
import shutil
from pathlib import Path
import numpy as np
from tqdm import tqdm

def make_folders(path='../out/'): #from utralytics/json2yolo/utils
    # Create folders
    if os.path.exists(path):
        shutil.rmtree(path)  # delete output folder
    os.makedirs(path)  # make new output folder
    os.makedirs(path + os.sep + 'labels')  # make new labels folder
    os.makedirs(path + os.sep + 'images')  # make new labels folder
    return path

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
            for x in tqdm(data['annotations'], desc='Annotations %s' % Path(json_file).stem):
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
                            img_new_path = '%s\n' % Path(os.path.join(datasets_root_dir, f)) #original datasets path
                            if copy_images or only_labels: # yolv5.utils.datasets.LoadImagesAndLabels f += [x.replace('./', parent) if x.startswith('./') else x for x in t]  # local to global path
                                img_new_path = '%s\n' % Path('./'+imgdir_path + img_new_fn + img_ext) #local path
                                #img_new_path = '%s\n' % Path(outdir + imgdir_path + img_new_fn + img_ext)  # global path
                            file.write(img_new_path)
                        img_counter += 1
                        lb_d[str(x['image_id'])]=[]
                        save_img_a.append(str(x['image_id']))  # only images that are saved
                        #print(str(x['image_id']))
                    # write labels
                    if  (str(x['image_id']) in save_img_a) and (not str(box) in lb_d[str(x['image_id'])]):  # don't write same labels (even if annotations are duplicated)
                        lb_d[str(x['image_id'])].append(str(box))
                        if copy_images or only_labels:
                            with open(dir_label + img_new_fn + '.txt', 'a') as file:
                                file.write('%g %.6f %.6f %.6f %.6f\n' % (x['category_id'] - 1, *box)) #category_id starts with 1
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
    parser.add_argument('--copy_images', type=int, default=0, help='copy images to folder /images and add new path to .txt . If 0 no image is copied and .txt has the original paths; if ')
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

