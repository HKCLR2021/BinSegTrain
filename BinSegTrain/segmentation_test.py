import os
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import cv2
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import HookBase
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
import torch
import argparse
import numpy as np
import time
import glob
from config import add_rmRCNN_config
import matplotlib.pyplot as plt
import tools
import math
##
import low_solidity_support as loso
if loso.IS_LOW:
    import part_aware_rot_maskrcnn
else:
    import rotated_maskrcnn
##



def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", default="outfiles/10/config.yaml", metavar="FILE", help="path to config file")
    parser.add_argument("--model_path", default='outfiles/10/model_0000999.pth', type=str)
    parser.add_argument("--score_thresh_test", default=0.1, type=float) # 0.5, 0.7, 0.9
    parser.add_argument("--nms_thresh_test", default=0.5, type=float, help='iou threshold when conducting NMS') #0.5, 0.8, 0.9
    parser.add_argument("--kernel", default=15, type=int)  # test files have to follow certain file structure --depth --gray --ply (ply for depth scaling)
    return parser

def setup_cfg(args):
    cfg = get_cfg()
    add_rmRCNN_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.MODEL.WEIGHTS = args.model_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.score_thresh_test
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = args.nms_thresh_test
    cfg.freeze()
    return cfg



def segment_image(model, img, seg_candidate_mask_num=None):
    """
    If seg_candidate_mask_num is given,
    sort the detected objs by area and return the topN candidates's mask
    """
    ##
    if len(img.shape) == 2:
        img = np.expand_dims(img, 2).repeat(3, 2)
    ## pred_mask ##
    height, width = img.shape[:2]
    input_tensor = torch.as_tensor(img.astype("float32").transpose(2, 0, 1))
    inputs = {"image": input_tensor, "height": height, "width": width}
    outputs = model([inputs])[0]
    
    ##
    if loso.IS_LOW:
        pred_masks = loso.aggregate(outputs, img)
    else:
        pred_masks = outputs['instances'].get('pred_masks').detach().cpu().numpy()
    ##
    
    if seg_candidate_mask_num is None: return pred_masks

    mask_sizes = []
    for mask in pred_masks:
        mask_sizes.append(np.sum(mask))
        # pred_masks_idx = [i for i in range(len(pred_masks)) if (pred_masks[i].sum() > 1000)]
    print(len(mask_sizes))
    picked_idx_list = np.argsort(mask_sizes)[::-1][:seg_candidate_mask_num]
    picked_mask_list = pred_masks[picked_idx_list]
    # print(np.sum(picked_mask))
    return picked_mask_list

if __name__ == '__main__':
    args = argument_parser().parse_args()
    cfg = setup_cfg(args)
    model = build_model(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    model.eval()
    img = cv2.imread('handle_virtual_img.png')
    mask_list = segment_image(model, img, 1)
    for m in mask_list:
        img[m] = np.random.uniform(0,255,(3))
    plt.imshow(img)
    plt.show()
