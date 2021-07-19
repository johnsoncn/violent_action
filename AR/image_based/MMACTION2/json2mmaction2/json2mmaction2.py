import json
import os
import argparse
import time

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
    return

def extract_file(src,dst,copy=True):
    extracted=True
    try:
        if copy:
            if not os.path.exists(os.path.dirname(dst)): os.makedirs(os.path.dirname(dst)) #make sure dir exists
            shutil.copyfile(src, dst)  # raises if missing files
        else: #if not copy only extracting filelist from json
            if not os.path.exists(src): raise
    except:
        print("\n>> missing : {}".format(src))
        extracted=False
    return extracted

def write_from_annotation(json_file, data, images, videos, categories, copy_images, copy_videos, outdir_img, outdir_video, img_number=None, level=2):
    # WRITE FILES (COPY & GENERATE FILELIST)
    # image lists
    img_l = []
    saved_img_l= []
    imglist = []
    img_counter = 0 # image counter
    # video lists
    video_l = []
    saved_video_l= []
    videolist = []
    # write files
    method="for" #TODO: parfor method
    start=time.time()
    if method=="for":
        #WRITE IMAGES
        for x in tqdm(data['annotations'], desc='Annotations %s' % json_file):
            # extract image info from x['image_id']
            image_id='%g' % x['image_id']
            if image_id in img_l: continue # continue to next loop if repeated image_id
            img_l.append(image_id)
            img = images[image_id]
            h, w, imgf = img['height'], img['width'], img['file_name']
            _, img_ext = os.path.splitext(imgf)
            img_fn = Path(imgf).stem
            img_new_fn = "img_"+image_id #img_imgid.jpg (imgid with zeros 00001: image_id.zfill(5) ) Problem is I don't no the maximum of images
            # extract video info from img['video_id']
            video_id = '%g' % img["video_id"]
            video = videos[video_id]
            videof= video["name"]
            video_fn = Path(videof).stem
            video_new_fn = "video_"+video_id
            # extract label and category
            catid = '%g' % x['category_id']
            label = catid
            category = categories[catid]
            # extract total label frames
            total_frames = '%g' % x['label_frames']
            # extract category
            category = categories[label]['name']
            # extract bounding box format is [top left x, top left y, width, height] | [x,y,w,h]
            box = np.array(x['bbox'], dtype=np.float64)
            box[:2] += box[2:] / 2  # xy top-left corner to center
            box[[0, 2]] /= w  # normalize x & w
            box[[1, 3]] /= h  # normalize y & h
            if (box[2] > 0.) and (box[3] > 0.):  # if w > 0 and h > 0
                # write images - 1st because if copy_images fails the rest should not be done
                src = os.path.join(datasets_root_dir, imgf)
                dst = os.path.join(outdir_img, video_new_fn, img_new_fn + img_ext)
                if level==2: dst = os.path.join(outdir_img, category, video_new_fn, img_new_fn + img_ext)
                ext=extract_file(src,dst,copy=copy_images)
                if not ext: continue #if image missing from dataset when extracting images dont write nothing more
                # img list:
                imgline = f'{video_new_fn}/{img_new_fn}\n' # f'{video_new_fn}/{img_new_fn} {total_frames} {label}\n'
                if level==2: imgline = f'{category}/{video_new_fn}/{img_new_fn}\n' # f'{category}/{video_new_fn}/{img_new_fn} {total_frames} {label}\n'
                imglist.append(imgline)
                img_counter += 1
                # rawframe annotation file list: json to txt [ frame_directory total_frames label  ]
                vidline = f'{video_new_fn} {total_frames} {label}\n'
                if level==2: vidline = f'{category}/{video_new_fn} {total_frames} {label}\n'
                videolist.append(vidline)
            # STOP conditions
            if img_number and img_counter >= img_number:
                print("STOP CONDITION")
                break
        #remove duplicate paths
        imglist=list(dict.fromkeys(imglist))
        videolist=list(dict.fromkeys(videolist))
    stop = time.time()
    elapsed=stop-start
    print("time elapsed:", elapsed)
    return imglist, videolist, saved_img_l, saved_video_l

def mola2mmaction2(datasets_root_dir=None, json_file='mola.json', outdir='out/', copy_images=True, copy_videos=False, img_number=None, level=2):
    # MAKE ROOT DIRS
    videodir_path = 'videos_%s/' % Path(json_file).stem  # folder name (train, val, test) remove other info
    imgdir_path = 'rawframes_%s/' % Path(json_file).stem  # folder name (train, val, test) remove other info
    outdir_video = os.path.join (outdir, videodir_path)
    outdir_img = os.path.join (outdir, imgdir_path)
    if copy_videos: make_folders(path=outdir_video)
    if copy_images: make_folders(path=outdir_img)
    # PARSE JSON ANNOTATIONS
    data=None
    with open(json_file) as f:
        data = json.load(f)
    if not data: raise
    # create image dict {id: image}
    images = {'%g' % x['id']: x for x in data['images']}
    # create video dict {id: video}
    videos = {'%g' % x['id']: x for x in data['videos']}
    # create category dict {id: category}
    categories = {'%g' % x['id']: x for x in data['categories']}
    # WRITE FILES (COPY & GENERATE FILELIST)
    method="from_annotation"
    if method=="from_annotation":
        imglist, videolist, saved_img_l, saved_video_l=write_from_annotation(json_file, data, images, videos, categories, copy_images, copy_videos, outdir_img, outdir_video, img_number=img_number, level=level)

    return imglist, videolist, saved_img_l, saved_video_l

def convert_mola_json(dataset="mola", datasets_root_dir=None, json_dir='../mola/annotations/', outdir='out/', copy_images=True, copy_videos=False, img_number=None, level=2):
    # Convert motionLab JSON file into  labels --------------------------------
    make_folders(path=outdir)  # output directory
    jsons = glob.glob(json_dir + '*.json')
    # Import json
    for json_file in sorted(jsons):
        imglist, videolist, saved_img_l, saved_video_l = mola2mmaction2(datasets_root_dir=datasets_root_dir,
                                                                        json_file=json_file,
                                                                        outdir=outdir,
                                                                        copy_images=copy_images,
                                                                        copy_videos=copy_videos,
                                                                        img_number=img_number,
                                                                        level=level)
        # GENERATE FILELISTS
        #save videolist rawframes annotations : mola_{train,val}_rawframes.txt
        dataset_type='rawframes'
        filename=f'{dataset}_{Path(json_file).stem}_{dataset_type}.txt'
        with open(outdir + filename, 'w') as f:
            f.writelines(videolist)


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
    parser.add_argument('--copy_images', type=int, default=0, help='copy_images images to folder /rawframes_{json_file} and add new path to .txt .')
    parser.add_argument('--copy_videos', type=int, default=0,
                        help='copy_videos to folder /videos_{json_file} and add new path to .txt ')
    parser.add_argument('--level', type=int, default=1,
                        help='write directory level: 1, video_id/; 2, category/video_id')

    opt = parser.parse_args()

    source = opt.source
    datasets_root_dir = opt.datasets_root_dir
    json_dir = opt.json_dir
    outdir = opt.outdir
    img_number = opt.img_number
    copy_images = False
    if opt.copy_images == 1: copy_images = True
    if opt.only_videos == 1: copy_videos = True #TODO
    level=opt.level
    print('\n>>' + str(opt))

    if not datasets_root_dir: raise RuntimeError('Select datasets_root_dir')

    if source == 'mola':
        # CREATE LABELS and IMAGES FOLDER
        convert_mola_json(datasets_root_dir=datasets_root_dir, json_dir=json_dir,
                          outdir=outdir, copy_images=copy_images,
                          copy_videos=copy_videos, img_number=img_number, level=level)

