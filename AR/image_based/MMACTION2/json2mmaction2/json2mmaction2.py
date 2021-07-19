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



def convert_mola_json(datasets_root_dir=None, json_dir='../mola/annotations/', outdir='out/', copy_images=True, copy_videos=False, only_labels=False, img_number=None, track_id=None):
    # Convert motionLab JSON file into  labels --------------------------------
    '''
    #WARNING copy_images="1" #1=copy_images images from original datasets, easiest way to create  dataset
    # "0"= don't copy_images images use the ones from original #WARNING labels need to be organized
    » copy_images labels to the same root dir as "images" in original dataset and also have the same path as the images

    # Define label paths as a function of image paths
    sa, sb = os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep  # /images/, /labels/ substrings

    #track_id: None: using counter; str: the key of track id (e.g.: TAO is "track_id" - WARNING: MCMOT giving BUGs)
    '''
    outdir = make_folders(path=outdir)  # output directory
    jsons = glob.glob(json_dir + '*.json')
    # tao_cat_idx = [x for x in range(833)]  # object catagories index - 833 cat = 488 LVIS + 345 free-form

    # Import json
    for json_file in sorted(jsons):
        # make dirs
        videodir_path = 'videos_%s/' % Path(json_file).stem  # folder name (train, val, test) remove other info
        imgdir_path = 'rawframes_%s/' % Path(json_file).stem  # folder name (train, val, test) remove other info
        dir_video = outdir + videodir_path
        os.mkdir(dir_video)
        dir_img = outdir + imgdir_path
        os.mkdir(dir_img)

        #json file
        with open(json_file) as f:
            data = json.load(f)

        # image lists
        img_l = []
        save_img_l= []
        imglist = []
        
        # video lists
        video_l = []
        videolist = []

        # image counter
        img_counter = 0

        # Create image dict {id: image}
        images = {'%g' % x['id']: x for x in data['images']}

        # Create video dict {id: video}
        videos = {'%g' % x['id']: x for x in data['videos']}

        # WRITE
        method="for"
        import time
        start=time.time()

        #track id
        tid=0
        if method=="for":
            for x in tqdm(data['annotations'], desc='Annotations %s' % json_file):
                image_id='%g' % x['image_id']
                img = images[image_id]
                h, w, imgf = img['height'], img['width'], img['file_name']
                _, img_ext = os.path.splitext(imgf)
                img_fn = Path(imgf).stem
                img_new_fn = image_id #original filename - if the images are not copied mantain original path

                video_id = '%g' % img["video_id"]
                video = videos[video_id]
                videof= video["name"]
                video_fn = Path(videof).stem
                video_new_fn = video_id

                # The Labelbox bounding box format is [top left x, top left y, width, height] | [x,y,w,h]
                box = np.array(x['bbox'], dtype=np.float64)
                box[:2] += box[2:] / 2  # xy top-left corner to center
                box[[0, 2]] /= w  # normalize x & w
                box[[1, 3]] /= h  # normalize y & h

                if (box[2] > 0.) and (box[3] > 0.):  # if w > 0 and h > 0
                    # write images - 1st because if copy_images fails the
                    if not image_id in img_l: # don't repeat write images #if not (any([True for im in img_l if im.find(str(x['image_id'])) > -1])):
                        img_l.append(image_id)  # all images
                        # copy_images
                        if copy_images or only_labels:
                            src = os.path.join(datasets_root_dir, imgf)
                            dst = os.path.join(dir_img, img_new_fn + img_ext)
                            try:
                                if copy_images: copyfile(src, dst)  # missing files
                                if only_labels and not os.path.exists(src): raise
                                save_img_l.append(image_id)  # only images that are saved
                            except:
                                img_l[-1] = img_l[-1] + '_MISSING'
                                print("\n>> missing : {}".format(Path(os.path.join(datasets_root_dir, imgf))))
                                print(img_l[-1])
                                continue #if image missing from dataset when copy_images dont write nothing more
                        # rawframe annotation file list: json to txt [ frame_directory total_frames label  ]
                        imgline = "{} {} {}".format(category + '/' + video_id + '/' + img_new_fn, total_frames, label)
                        imglist.append(imgline)
                        img_counter += 1
                        print(image_id)
                    # write videos
                    if not video_id in video_l:  # don't repeat write images #if not (any([True for im in img_l if im.find(str(x['image_id'])) > -1])):
                        video_l.append(video_id)  # all images
                        # copy_videos
                        if copy_videos or only_labels:
                            src = os.path.join(datasets_root_dir, imgf)
                            dst = os.path.join(dir_img, img_new_fn + img_ext)
                            try:
                                if copy_videos: copyfile(src, dst)  # missing files
                                if only_labels and not os.path.exists(src): raise
                                save_img_l.append(image_id)  # only images that are saved
                            except:
                                img_l[-1] = img_l[-1] + '_MISSING'
                                print("\n>> missing : {}".format(Path(os.path.join(datasets_root_dir, videof))))
                                print(img_l[-1])
                                continue  # if image missing from dataset when copy_images dont write nothing more
                        #video annotation file list: json to txt [fielpath label]
                        videoline = "{} {} {}".format(category + '/' + video_id + '/' + img_new_fn, total_frames, label)
                        videolist.append(videoline)
                        print(video_id)
                # STOP
                if img_number and img_counter >= img_number: break

        #save imglist : mola_{train,val}_rawframes.txt
        with open(outdir + Path(json_file).stem + '.txt', 'w') as file:
            for i in imglist:
                file.write(i)
        #save videolist : mola_{train,val}_rawframes.txt
        with open(outdir + Path(json_file).stem + '.txt', 'w') as file:
            for v in videolist:
                file.write(v)

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
    parser.add_argument('--copy_images', type=int, default=0, help='copy_images images to folder /images and add new path to .txt . If 0 no image is copied and .txt has the original paths')
    parser.add_argument('--only_labels', type=int, default=0,
                        help='write labels only as if you were copy_imagesing images to folder /images and add new path to .txt .')
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

