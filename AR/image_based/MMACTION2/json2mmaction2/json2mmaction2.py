import json
from shutil import copyfile
import os
import argparse

import glob
import shutil
from pathlib import Path
import numpy as np
from tqdm import tqdm

def make_folders(path='../out/'):
    # Create folders
    if os.path.exists(path):
        shutil.rmtree(path)  # delete output folder
    os.makedirs(path)  # make new output folder
    os.makedirs(path + os.sep + 'labels_with_ids')  # make new labels folder #FairMOT.src.lib.datastets.dataset.jde self.label_files = [x.replace('images', 'labels_with_ids').replace('.png', '.txt').replace('.jpg', '.txt')for x in self.img_files]
    os.makedirs(path + os.sep + 'images')  # make new labels folder
    return path


# Convert motionLab JSON file into  labels --------------------------------
def convert_mola_json(datasets_root_dir=None, json_dir='../mola/annotations/', outdir='out/', copy_images=True, only_labels=False, img_number=None, track_id=None):
    '''
    #WARNING copy_images="1" #1=copy images from original datasets, easiest way to create  dataset
    # "0"= don't copy images use the ones from original #WARNING labels need to be organized
    » copy labels to the same root dir as "images" in original dataset and also have the same path as the images

    # Define label paths as a function of image paths
    sa, sb = os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep  # /images/, /labels/ substrings

    #track_id: None: using counter; str: the key of track id (e.g.: TAO is "track_id" - WARNING: MCMOT giving BUGs)
    '''
    outdir = make_folders(path=outdir)  # output directory
    jsons = glob.glob(json_dir + '*.json')
    # tao_cat_idx = [x for x in range(833)]  # object catagories index - 833 cat = 488 LVIS + 345 free-form

    # Import json
    for json_file in sorted(jsons):
        labeldir_path = 'labels_with_ids/%s/' % Path(json_file).stem  # folder name (train, val, test) remove other info
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
        import time
        start=time.time()

        #track id
        tid=0
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

                # The Labelbox bounding box format is [top left x, top left y, width, height] | [x,y,w,h]
                box = np.array(x['bbox'], dtype=np.float64)
                box[:2] += box[2:] / 2  # xy top-left corner to center
                box[[0, 2]] /= w  # normalize x & w
                box[[1, 3]] /= h  # normalize y & h

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
                            img_new_path = '%s\n' % Path(os.path.join(datasets_root_dir, f))  # original datasets path
                            if copy_images or only_labels:  #FairMOT.src.lib.cfg -jsons
                                img_new_path = '%s\n' % Path(imgdir_path + img_new_fn + img_ext)  # local path
                                # img_new_path = '%s\n' % Path(outdir + imgdir_path + img_new_fn + img_ext)  # global path
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
                                if track_id: file.write('{:d} {:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(x['category_id'] - 1, x[track_id], *box)) #category_id starts with 1, track_id starts in 0
                                else: file.write('{:d} {:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(x['category_id'] - 1, tid, *box))
                                # label_str = '{:d} {:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(class_id, tid_curr, x / img_width, y / img_height, w / img_width, h / img_height)
                        else:
                            with open(dir_label + os.path.splitext(f)[0]+ '.txt', 'a+') as file:
                                if track_id:
                                    file.write('{:d} {:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(x['category_id'] - 1,
                                                                                                x[track_id],
                                                                                                *box))  # category_id starts with 1, track_id starts in 0
                                else:
                                    file.write(
                                        '{:d} {:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(x['category_id'] - 1, tid,
                                                                                         *box))
                        tid+=1

                # STOP
                if img_number and img_counter >= img_number: break

        #Config file: root of export dataset; train paths to use
        cfg={
            "root": os.path.dirname(outdir),
            "train":
                {
                    "train": outdir+"train.txt",
                    "val": outdir+ "val.txt",
                },
            "test_emb":
                {
                    "test": outdir+"test.txt"
                },
            "test":
                {
                    "test": outdir+"test.txt"
                }
        }
        with open(outdir + 'cfg.json', 'w') as f:
            json.dump(cfg, f)

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
                        default="D:/external_datasets/MOLA/fairmotformat/mola_train/",
                        help='fairmotformat dataset output path')
    parser.add_argument('--img_number', type=int, default=None, help='number of images to convert. None=convert all')
    parser.add_argument('--copy_images', type=int, default=0, help='copy images to folder /images and add new path to .txt . If 0 no image is copied and .txt has the original paths')
    parser.add_argument('--only_labels', type=int, default=0,
                        help='write labels only as if you were copying images to folder /images and add new path to .txt .')
    parser.add_argument('--track_id', type=str,
                        default=None,
                        help='track id json key in annotations')

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
    track_id = opt.track_id

    if not datasets_root_dir: raise RuntimeError('Select datasets_root_dir')

    if source == 'mola':
        # CREATE LABELS and IMAGES FOLDER
        convert_mola_json(datasets_root_dir=datasets_root_dir, json_dir=json_dir,
                          outdir=outdir, img_number=img_number, copy_images=copy_images, only_labels=only_labels, track_id=track_id)

