import os.path as osp
import os
import cv2
import json
import numpy as np
import argparse

def mkdirs(d):
    if not osp.exists(d):
        os.makedirs(d)


def load_func(fpath): #TODO json
    print('fpath', fpath)
    assert os.path.exists(fpath)
    with open(fpath, 'r') as fid:
        lines = fid.readlines()
    records =[json.loads(line.strip('\n')) for line in lines]
    return records


def gen_labels(data_root, label_root, ann_root): #TODO
    mkdirs(label_root)
    anns_data = load_func(ann_root)

    tid_curr = 0
    for i, ann_data in enumerate(anns_data):
        print(i)
        image_name = '{}.jpg'.format(ann_data['ID'])
        img_path = os.path.join(data_root, image_name)
        anns = ann_data['gtboxes']
        img = cv2.imread(
            img_path,
            cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
        )
        img_height, img_width = img.shape[0:2]
        for i in range(len(anns)):
            if 'extra' in anns[i] and 'ignore' in anns[i]['extra'] and anns[i]['extra']['ignore'] == 1:
                continue
            x, y, w, h = anns[i]['fbox']
            x += w / 2
            y += h / 2
            label_fpath = img_path.replace('images', 'labels_with_ids').replace('.png', '.txt').replace('.jpg', '.txt')
            label_str = '0 {:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
                tid_curr, x / img_width, y / img_height, w / img_width, h / img_height)
            with open(label_fpath, 'a') as f:
                f.write(label_str)
            tid_curr += 1


if __name__ == '__main__': #TODO
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_rdir', type=str,
                        default=None,
                        help='root dir for all the images/')
    parser.add_argument('--json_dir', type=str,
                        default=None,
                        help='json annotations input path')
    opt = parser.parse_args()
    images_rdir= opt.images_rdir
    json_dir = opt.json_dir

    labels_to_gen=["train","val","test"]
    for l in labels_to_gen:
        try:
            data= images_rdir + '/images/'+l
            label = images_rdir + '/labels_with_ids/'+l
            ann= json_dir + '/'+l+'.json'
            gen_labels(data, label, ann)
        except Exception as e:
            print("\n>> Exception: ", e)
            continue


