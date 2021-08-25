'''
author: Feng Yidan
email: fengyidan1995@126.com
'''
import os
import pickle
import numpy as np
import cv2
import random
import glob
import json
# import pycocotools.mask as pm
from pycocotools import _mask
import scipy.io as scio
from skimage.transform import matrix_transform
from cv2 import getRotationMatrix2D, warpAffine
import math
import threading
import scipy.io as scio
import argparse
import matplotlib.pyplot as plt
from transform_from_virtual import get_rotated_obj
import glob
import progressbar
dataset = '01'


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='01', required=False)
    parser.add_argument('--phase', type=str, default='train', required=False)
    parser.add_argument('--right', type=float, default=0.8, help='right margin of the effective range = right (ratio) * real width')
    parser.add_argument('--left', type=float, default=0.2)
    parser.add_argument('--upper', type=float, default=0.2)
    parser.add_argument('--bottom', type=float, default=0.8)
    parser.add_argument('--max_tem', type=int, default=80, help='maximum number of templates on a background')
    parser.add_argument('--min_tem', type=int, default=20, help='minimum number of templates on a background')
    parser.add_argument('--gen_num_per_base', type=int, default=2, help='generate how many images for each background image. total number of iamges will be num_bg*gen_nem_per_base')
    parser.add_argument("--oc", type=float, default=0.88, help='threshold of visibility')
    return parser.parse_args()

def prepare_template(args):
    json_paths = glob.glob(os.path.join(args.name, 'tm', '*.json'))
    tem_masks = []
    tem_dicts = []
    for json_path in json_paths:
        with open(json_path, 'r') as f:
            full_dict = json.load(f)
        annos = full_dict['shapes']
        gray_path = glob.glob(os.path.join(args.name, 'tm')+ '/%s.*p*'%(json_path.split('/')[-1].split('.')[0]))[0]
        gray = cv2.imread(gray_path)
        for an_dict in annos:
            poly_list = []
            for xy in an_dict['points']:
                poly_list.append(xy[0])
                poly_list.append(xy[1])
            rle = _mask.frPoly(poly=[poly_list], h=full_dict['imageHeight'], w=full_dict['imageWidth'])
            mask = _mask.decode(rle)[:,:,0]
            xs, ys = np.where(mask == True)
            # compute relative coordinates
            left_margin = np.min(xs)
            new_xs = xs - left_margin+1
            upper_margin = np.min(ys)
            new_ys = ys - upper_margin+1
            new_mask = np.zeros((903,1621))
            new_mask[new_xs,new_ys] = 1
            tem_masks.append(new_mask)
            dict = {}
            for x,y,nx,ny in zip(xs, ys, new_xs, new_ys):
                dict[(nx,ny)] = gray[x, y, :]
            tem_dicts.append(dict)
    return tem_masks, tem_dicts

def copy_pasteN_per_base(b,dataset_dicts,args):
    base_original = cv2.imread(os.path.join(args.name,'bg', b))
    height,width,c = base_original.shape
    RIGHT =  int(width * args.right)
    LEFT = int(width * args.left)
    BOTTOM = int(height * args.bottom)
    UPPER = int(height * args.upper)

    image_id = dataset_dicts[-1]['image_id'] + 1
    bar = progressbar
    for i in bar.progressbar(range(args.gen_num_per_base)):
        record = {}
        idx = 1
        sample_num = random.randint(args.min_tem, args.max_tem) # sample random N (a~b) templates
        base_img = base_original.copy()
        base_label = np.zeros((height,width,3))
        tem_addrs = {}
        tem_num = len(tm_masks)
        enlarge_ratio = 5
        while enlarge_ratio * tem_num < args.max_tem:
            enlarge_ratio *= 2
        candidates = enlarge_ratio * list(range(tem_num))
        sample_temps = random.sample(candidates, sample_num)
        random.shuffle(sample_temps)
        for tem_idx in sample_temps:
            tmask = tm_masks[tem_idx]
            xs, ys = np.where(tmask == True)
            x_min, x_max, y_min, y_max = np.min(xs), np.max(xs), np.min(ys), np.max(ys)
            left_margin = y_min -LEFT
            righ_margin = RIGHT - y_max
            top_margin = x_min - UPPER
            bottom_margin = BOTTOM - x_max
            tdict = tm_dicts[tem_idx]
            '''translate --> rotate--> paste'''
            tem_depth_image = np.zeros((height, width, 3)).astype('uint8')
            t_x = random.randrange(-top_margin, bottom_margin)
            t_y = random.randrange(-left_margin, righ_margin)
            tt = np.zeros((height, width))
            for x, y in zip(xs, ys):
                x1 = x + t_x
                y1 = y + t_y
                tt[x1, y1] = 1
                tem_depth_image[x1, y1, :] = tdict[(x, y)]
            tmask = (tt == 1)

            angle = random.randint(0, 36) * 10
            x1s, y1s = np.where(tmask == True)
            x1_min, x1_max, y1_min, y1_max = np.min(x1s), np.max(x1s), np.min(y1s), np.max(y1s)
            rot_mat = getRotationMatrix2D(center=((y1_min + y1_max) / 2, (x1_min + x1_max) / 2), angle=angle,
                                          scale=1)
            rotated_m = warpAffine(tmask.astype('uint8'), rot_mat, (width, height))
            rotated_d = warpAffine(tem_depth_image.astype('uint8'), rot_mat, (width, height))

            mask = (rotated_m == 1)
            base_img[mask] = rotated_d[mask]
            base_label[mask] = idx

            tem_addrs[idx] = mask

            idx += 1

        # transform copy-paste generated objects into detectron2 annotation format
        objs = []
        for label_id, tem_mask in tem_addrs.items():
            inst_full_mask = tem_mask
            inst_mask = (base_label[:,:,0]==label_id)
            visibility = np.sum(inst_mask) / np.sum(inst_full_mask)
            if visibility > args.oc:
                objs.append(get_rotated_obj(inst_mask))

        record["annotations"] = objs
        fname = b.split('.')[0] + '_' + str(i)+'.png'
        gray_outdir = os.path.join(os.getcwd(), args.name,'train')
        os.makedirs(gray_outdir, exist_ok=True)
        gray_outpath = os.path.join(gray_outdir,fname)
        cv2.imwrite(os.path.join(gray_outpath), base_img)
        record['file_name'] = gray_outpath
        record['image_id'] = image_id
        image_id += 1
        record['height'] = height
        record['width'] = width
        dataset_dicts.append(record)


if __name__ == '__main__':
    args = get_args()
    json_path = args.name + f'/json/{args.phase}.json'
    old_json_path = args.name + f'/json/{args.phase}_old.json'
    with open(json_path, 'r') as f:
        dataset_dicts = json.load(f)
    base_dir = os.path.join(args.name, 'bg')
    tm_masks, tm_dicts = prepare_template(args)
    print('templates prepared!!')
    for b in os.listdir(base_dir):
        copy_pasteN_per_base(b, dataset_dicts, args)

    os.rename(json_path, old_json_path)
    print('old json renamed!!')
    with open(json_path, 'w') as f:
        json.dump(dataset_dicts, f)
        print(f"json file: {args.phase} accomplished!")
