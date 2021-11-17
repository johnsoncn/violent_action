#!/usr/bin/env python3
"""
This script converts the original object instance annotations of the MS COCO data set from json to pickle
A description of the original annotations can be found here: http://cocodataset.org/#download

You can call the script as follows:

python  mscoco_pickle_generator.py <input file> [<output file>]

If you do not explicitly specify an output file name, the output file name is automatically generated from the
input file name by changing its file ending to pickle.


The script reorganizes the data of the original file. The original file is organized as follows::

    {
      "info": info,
      "images": [image],
      "annotations": [annotation],
      "licenses": [license],
    }

where "images" is a list of image descriptions, and "annotations" contains a list of object annotations.
In contrast to the original JSON, the images will be stored sorted in increasing order of their id in the pickle file.
Moreover, the annotations are stored in the pickle file as a list of lists of annotations, such that each sublist
contains all annotations of an image. The sublists are again sorted by the corresponding image id.

I.e., the format is::

    {
      "info": info,
      "images": [image_i, image_j, image_k, ... ],
      "annotations": [ [annotation_image_i], [annotation_image_j], [annotation_image_j], ... ]
      "licenses": [license],
    }

with i < j < k


"""
import numpy as np
from itertools import groupby
import json
import bolt.utils.pickle as pickle
import sys
import os
import glob
import datetime
from tqdm import tqdm

def PixelpairToAbsolutePixel(pixels_list, width):
    """
    receive a list of pixel pairs (x,y) and converts them into absolute pixels of the image 

    return absolute pixel values
    """
    abs_pixel_list = []
    for i in range(0,len(pixels_list),2):
        x = pixels[i]
        y = pixels[i+1]

        abs_pixel_list.append(y * width + x + 1)

    return abs_pixel_list


# This function takes a list of pixels, and returns them in run-length encoded format.
def PixelsToRLenc(pixels, width, height, order='F', format=False):
    """
    Based off code by https://www.kaggle.com/alexlzzz
    pixels is a list of pixels [x1,y1,x2,y2,...,xn,yn] values which need to be converted to absolute pixel values. (1-(xn*yn))
        - Done by function PixelpairToAbsolutePixel(pixels_list, width)
    order is down-then-right, i.e. Fortran
    format determines if the order needs to be preformatted (according to submission rules) or not
    
    returns run length as an array or string (if format is True)
    """
    
    # Initialse empty array
    bytes = []
    for _ in range(0, height * width):
        bytes.append(0)

    # list of pixels [x1,y1,x2,y2,...,xn,yn] values are converted to absolute pixel values.
    pixels = PixelpairToAbsolutePixel(pixels, width)
    
    # Place values from input list into the array
    for x in pixels:
        p = x - 1
        bytes[p] = 1
    
    runs = [] ## list of run lengths
    r = 0     ## the current run length
    pos = 1   ## count starts from 1 per WK
    for c in bytes:
        if ( c == 0 ):
            if r != 0:
                runs.append((pos, r))
                pos+=r
                r=0
            pos+=1
        else:
            r+=1

    #if last run is unsaved (i.e. data ends with 1)
    if r != 0:
        runs.append((pos, r))
        pos += r
        r = 0

    if format:
        z = ''
    
        for rr in runs:
            z+='{} {} '.format(rr[0],rr[1])
        return z[:-1]
    else:
        return runs
        

# This function takes a string in run-length encoded format, and returns a list of pixels
def RLencToPixels(runs):
    p1 = [] # Run-start pixel locations
    p2 = [] # Run-lengths
    
    # Separate run-lengths and pixel locations into seperate lists
    x = str(runs).split(' ')
    i = 0
    for m in x:
        if i % 2 == 0:
            p1.append(m)
        else:
            p2.append(m)
        i += 1
        
    # Get all absolute pixel values
    pixels = []
    for start, length in zip(p1, p2):
        i = 0
        length = int(length)
        pix = int(start)
        while i < length:
            pixels.append(pix)
            pix += 1
            i += 1
            
    return pixels

def read_split_file(split_filename):
    # Reading an excel file using Python
    import xlrd
    
    train_list, validation_list, test_list = [],[],[]
    sheet_names = ['train','validation','test']

    # To open Workbook
    xl_workbook = xlrd.open_workbook(split_filename)

    # Explore each sheet of the excel file
    for i in sheet_names:
        xl_sheet = xl_workbook.sheet_by_name(i)

        # Verify in each sheet the first cell of each row (filename)
        # Append to the respective list
        for j in range(xl_sheet.nrows):
            if i == 'train':
                train_list.append(xl_sheet.cell_value(j, 0))
            elif i == 'validation':
                validation_list.append(xl_sheet.cell_value(j, 0))
            elif i == 'test':
                test_list.append(xl_sheet.cell_value(j, 0))

    return train_list, validation_list, test_list

def lbo2mscoco(input_path, split_filename, verbose = True):
    """
    Generates dict file from multiple JSON files, whose structure corresponds to the format used in the original MS COCO object instance annotation files

    Three dicts are return with the respective annotations of train, validation and test set, on the MSCOCO annotation format
    
    Parameters
    ----------
    input_path : str
        File path of input JSON files (lbo annotation format)
    split_filename : str
        File path of xls input file (defines the distribution of each file per train/validation/test set)
    """

    train_list, validation_list, test_list = read_split_file(split_filename)

    if len(train_list) + len(validation_list) + len(test_list) == 0:
        print("No split filepath was found")
        quit()

    files = glob.glob(os.path.join(input_path,'**','*.json'), recursive=True)
    
    # Create a dict for train/test/valid
    list_dict = []
    list_category = []
    list_subcategory = []
    
    for i in range(0,3):
        new_dict = {}
        new_dict['info'] = {}
        new_dict['info']['year'] = int(2019)
        new_dict['info']['version'] = '2'
        new_dict['info']['description'] = 'Bosch Car Multimedia - IVS LBO Dataset'
        new_dict['info']['contributor'] = 'Bosch Car Multimedia - IVS'
        new_dict['info']['url'] = 'https://www.bosch.pt/a-nossa-empresa/bosch-em-portugal/braga/'
        new_dict['info']['date_created'] = datetime.datetime(2019, 10, 15)

        new_dict['images'] = []
        new_dict['categories'] = []
        new_dict['annotations'] = []
        new_dict['licenses'] = []
        
        # Add ID_Categories
        # 0: NOWEAPON
        # 1: WEAPON
        new_entry_category = {}
        new_entry_category['id'] = int(1)
        new_entry_category['name'] = 'no_weapon'
        new_entry_category['supercategory'] = 'no_weapon'
        new_dict['categories'].append(new_entry_category)

        new_entry_category = {}
        new_entry_category['id'] = int(2)
        new_entry_category['name'] = 'weapon'
        new_entry_category['supercategory'] = 'weapon'
        new_dict['categories'].append(new_entry_category)

        # Add ID_Licenses
        # 0: N/A
        new_entry_license = {}
        new_entry_license['id'] = int(1)
        new_entry_license['name'] = 'Bosch Car Multimedia License'
        new_entry_license['url'] = 'https://www.bosch.pt/a-nossa-empresa/bosch-em-portugal/braga/'
        new_dict['licenses'].append(new_entry_license)
        
        list_dict.append(new_dict)


    # Take care of json files and create pickle
    image_id, annotation_id = 1, 1
    n_weapons_train, n_no_weapons_train, images_train = 0, 0, 0
    n_weapons_valid, n_no_weapons_valid, images_valid = 0, 0, 0
    n_weapons_test, n_no_weapons_test, images_test = 0, 0, 0

    for input_filename in tqdm(files):
        with open(input_filename, "r") as file:
            
            filename = os.path.basename(input_filename)
            data = json.loads(file.read())

            if len(data['annotations']) == 0:
                pass

            annotations_weapons, annotations_no_weapons = 0, 0

            # Create dict with a defined structure for each json
            # Filter only relevant info.
            new_entry_image = {}
            new_entry_image['id'] = int(image_id) #data['image_info']['id']
            new_entry_image['width'] = int(data['image_info']['width'])
            new_entry_image['height'] = int(data['image_info']['height'])
            new_entry_image['file_name'] = str(data['image_info']['file_name']).replace('png','jpg')
            new_entry_image['file_path'] = str(input_filename).replace('json','jpg')
            new_entry_image['license'] = int(1)
            new_entry_image['flickr_url'] = ''
            new_entry_image['coco_url'] = ''
            new_entry_image['url'] = str(data['image_info']['url']).replace('png','jpg')

            # 2019-10-26T12:15:04.458981+01:00

            if 'date_captured' in data['image_info']:
                date = data['image_info']['date_captured'].split('T')[0].split('-')
                new_entry_image['date_captured'] = datetime.datetime(int(date[0]), int(date[1]), int(date[2]))
            else:
                new_entry_image['date_captured'] = datetime.datetime(2019, 10, 15)

            new_entry_image['lighting_conditions'] = str(data['image_info']['lighting_conditions'])
            new_entry_image['is_empty'] = str(data['image_info']['is_empty'])
            new_entry_image['is_labeled'] = str(data['image_info']['is_labeled'])
            new_entry_image['car'] = str(data['acquisition_info']['car'])
            new_entry_image['hardware'] = str(data['acquisition_info']['hardware'])
            new_entry_image['location'] = str(data['acquisition_info']['location'])

            dict_annotations = []

            for annotation in data['annotations']:
                new_entry_annotation = {}
                new_entry_annotation['id'] = int(annotation_id)
                new_entry_annotation['image_id'] = int(image_id)
                if annotation['category_id'] == 'Weapons':
                    new_entry_annotation['category_id'] = int(2)
                    annotations_weapons = annotations_weapons + 1
                else:
                    new_entry_annotation['category_id'] = int(1)
                    annotations_no_weapons = annotations_no_weapons + 1

                new_entry_annotation['area'] = float((annotation['bbox']['xmax'] - annotation['bbox']['xmin']) * (annotation['bbox']['ymax'] - annotation['bbox']['ymin']))
                # bbox = [x,y,width,height]
                new_entry_annotation['bbox'] = [int(annotation['bbox']['xmin']), int(annotation['bbox']['ymin']), int(annotation['bbox']['xmax'] - annotation['bbox']['xmin']), int(annotation['bbox']['ymax'] - annotation['bbox']['ymin'])]
                
                # if len(data['annotations']) > 1:
                #     new_entry_annotation['iscrowd'] = int(1)
                #     new_entry_annotation['segmentation'] = {}
                #     new_entry_annotation['segmentation']['counts'] = []
                #     new_entry_annotation['segmentation']['size'] = []

                #     pixels = [annotation['bbox']['xmin'], annotation['bbox']['ymin'], annotation['bbox']['xmin'], annotation['bbox']['ymax'], annotation['bbox']['xmax'], annotation['bbox']['ymax'], annotation['bbox']['xmax'], annotation['bbox']['ymin']]

                #     PixelsToRLenc(pixels, new_entry_image['width'], new_entry_image['height'])

                # else:
                new_entry_annotation['iscrowd'] = int(0)

                new_entry_annotation['segmentation'] = [[annotation['bbox']['xmin'], annotation['bbox']['ymin'], annotation['bbox']['xmin'], annotation['bbox']['ymax'], annotation['bbox']['xmax'], annotation['bbox']['ymax'], annotation['bbox']['xmax'], annotation['bbox']['ymin']]]

                new_entry_annotation['subcategory_id'] = str(annotation['subcategory_id'])
                
                dict_annotations.append(new_entry_annotation)
                annotation_id = annotation_id + 1

                if annotation['category_id'] not in list_category:
                    list_category.append(annotation['category_id'])

                if annotation['subcategory_id'] not in list_subcategory:
                    list_subcategory.append(annotation['subcategory_id'])

            # Distribute json between different pickles - train / validation / test
            if new_entry_image['file_name'] in train_list:
                #aux_dict = list_dict[0]
                list_dict[0]['images'].append(new_entry_image)
                list_dict[0]['annotations'].extend(dict_annotations)
                n_weapons_train = n_weapons_train + annotations_weapons
                n_no_weapons_train = n_no_weapons_train + annotations_no_weapons
                images_train = images_train + 1

            elif new_entry_image['file_name'] in validation_list:
                list_dict[1]['images'].append(new_entry_image)
                list_dict[1]['annotations'].extend(dict_annotations)
                n_weapons_valid = n_weapons_valid + annotations_weapons
                n_no_weapons_valid = n_no_weapons_valid + annotations_no_weapons
                images_valid = images_valid + 1

            elif new_entry_image['file_name'] in test_list:
                list_dict[2]['images'].append(new_entry_image)
                list_dict[2]['annotations'].extend(dict_annotations)
                n_weapons_test = n_weapons_test + annotations_weapons
                n_no_weapons_test = n_no_weapons_test + annotations_no_weapons
                images_test = images_test + 1

            image_id = image_id + 1

    # Rearrange images in pickle so that they are sorted by their image_id
    for i in range(0,3):
        list_dict[i]['images'] = sorted(list_dict[i]['images'], key=lambda x: x['id'])
        list_dict[i]['annotations'] = sorted(list_dict[i]['annotations'], key=lambda x: x['image_id'])

    if verbose:
        #print('Object Category List:', list_category)
        #print('Object Sub-category List:', list_subcategory)
        print('Images (train-valid-test-total): ', images_train, images_valid, images_test, images_train + images_valid + images_test)
        print('Weapon BBox Annotations (train-valid-test-total): ', n_weapons_train, n_weapons_valid, n_weapons_test, n_weapons_train + n_weapons_valid + n_weapons_test)
        print('Non-weapon BBox Annotations (train-valid-test-total): ', n_no_weapons_train, n_no_weapons_valid, n_no_weapons_test, n_no_weapons_train + n_no_weapons_valid + n_no_weapons_test)

    return list_dict

def mscoco_dict2pckle(data, output_filename):
    """
    Generates pickle file from mscoco dict, whose structure corresponds to the format used in the original 
    MS COCO object instance annotation files
    
    Parameters
    ----------
    data : dict
        MS COCO dict annotation
    output_filename : str
        File name of output file
    """

    # Rearrange images in pickle so that they are sorted by their image id
    data["images"] = sorted(data["images"], key=lambda x: x["id"])
    # Rearrange annotations so that in the result annotation list,
    # each entry holds exactly those annotations of the corresponding entry (same index)
    # in the "images"-list. An entry can be the empty list (no annotation available for that image)
    annotations = data.pop("annotations")
    image_annotations = {image["id"]: [] for image in data["images"]}
    for annotation in annotations:
        image_annotations[annotation["image_id"]].append(annotation)
    data["annotations"] = [image_annotations[key] for key in sorted(image_annotations.keys())]

    # Sanity check: Do we have index correspondence in "images" and "annotations" array?
    assert(len(data["images"]) == len(data["annotations"]))
    for image, annotations in zip(data["images"], data["annotations"]):
        for annotation in annotations:
            assert(image["id"] == annotation["image_id"])

    with open(output_filename, "wb") as file:
        file.write(pickle.dumps(data))


if __name__ == "__main__":

    if len(sys.argv) < 4:
        print("Too few arguments (expected 4).")
        print("Usage: python {} <input path> <split_xlsx filepath> <output file>".format(sys.argv[0]))
        quit()

    else:
        input_filepath = sys.argv[1]
        xlsx_split_filename = sys.argv[2]
        output_path = sys.argv[3]
        
        dict_list = lbo2mscoco(input_filepath, xlsx_split_filename)

        for i in range(len(dict_list)):
            if i == 0:
                output_filepath = os.path.join(output_path,'train.pickle')
            elif i == 1:
                output_filepath = os.path.join(output_path,'valid.pickle')
            elif i == 2:
                output_filepath = os.path.join(output_path,'test.pickle')

            if len(dict_list[i]['annotations']) > 0:
                mscoco_dict2pckle(dict_list[i], output_filepath)
