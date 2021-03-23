import os.path as osp
import os
import cv2
import json
import numpy as np
import argparse
import tqdm

def mkdirs(d):
    if not osp.exists(d):
        os.makedirs(d)


def load_func_cocoformat(fpath):
    print('fpath', fpath)
    assert os.path.exists(fpath)
    return json.load(open(fpath))


def gen_labels_cocoformat(data_root, label_root, ann_root, check_img_size=True, track_id=None):
    mkdirs(label_root)
    json_data = load_func_cocoformat(ann_root)
    anns_data = json_data['annotations']
    images = {'%g' % x['id']: x for x in json_data['images']}  # Create image dict {id: image}

    tid_curr = 0 #track_id
    for i, ann_data in enumerate(tqdm(anns_data)):
        try:
            if ann_data['iscrowd']: continue
        except:
            print('missing "iscrowd" key')
        #print('\n>> ', i)
        #image
        img_js = images['%g' % ann_data['image_id']]
        h, w, f = img_js['height'], img_js['width'], img_js['file_name'] #annotations iamge size
        _, img_ext = os.path.splitext(f)
        image_name = '%g' % (ann_data['image_id']) + img_ext
        img_path = os.path.join(data_root, image_name)
        if not os.path.exists(img_path):
            print('\n>> image missing: ', img_path)
            continue
        #actual image size
        if check_img_size:
            img = cv2.imread(
                img_path,
                cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
            )
            img_height, img_width = img.shape[0:2]
            assert img_height==h
            assert img_width==w

        # The Labelbox bounding box format is [top left x, top left y, width, height] | [x,y,w,h]
        box = np.array(ann_data['bbox'], dtype=np.float64)
        box[:2] += box[2:] / 2  # xy top-left corner to center
        box[[0, 2]] /= w  # normalize x & w
        box[[1, 3]] /= h  # normalize y & h

        if (box[2] > 0.) and (box[3] > 0.):  # if w > 0 and h > 0
            label_fpath = img_path.replace('images', 'labels_with_ids').replace(img_ext, '.txt')
            if track_id:
                label_str ='{:d} {:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(ann_data['category_id'] - 1, ann_data[track_id],
                                                                            *box)  # category_id starts with 1, track_id starts in 0
            else:
                label_str ='{:d} {:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(ann_data['category_id'] - 1, tid_curr, *box)
            with open(label_fpath, 'a') as f:
                f.write(label_str)
            tid_curr += 1





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_rdir', type=str,
                        default=None,
                        help='root dir for all the images/')
    parser.add_argument('--json_dir', type=str,
                        default=None,
                        help='json annotations input path')
    parser.add_argument('--track_id', type=str,
                        default=None,
                        help='track id json key in annotations')
    opt = parser.parse_args()
    images_rdir= opt.images_rdir
    json_dir = opt.json_dir
    track_id = opt.track_id

    if not images_rdir or not json_dir: raise AssertionError("add arguments --images_rdir and --json_dir")

    labels_to_gen=["train","val","test"]
    for l in labels_to_gen:
        try:
            data= images_rdir + '/images/'+l
            label = images_rdir + '/labels_with_ids/'+l
            ann= json_dir + '/'+l+'.json'
            gen_labels_cocoformat(data, label, ann, track_id=track_id)
        except Exception as e:
            print("\n>> Exception: ", e)
            continue


