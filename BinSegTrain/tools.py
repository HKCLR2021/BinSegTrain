'''
author: Feng Yidan
email: fengyidan1995@126.com
'''
import h5py
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from skimage.measure import label

import pycocotools.mask as pm
import scipy.io as scio
import argparse
import json
import os
import numpy.ma as ma
from matplotlib.pyplot import *
import glob
import pickle
import open3d as o3d
import progressbar


def is_low_solidity(obj_path='datasets/part10.obj', data_root='example_virtual_data_output/10/simple/0000'):
    meta_dir = f'{data_root}/meta'
    meta_path = meta_dir+'/'+os.listdir(meta_dir)[0]
    meta = scio.loadmat(meta_path)
    intrinsics = meta['intrinsic_matrix_CameraLeft']
    obj_mesh = o3d.io.read_triangle_mesh(obj_path)
    model_pc = obj_mesh.sample_points_uniformly(number_of_points=200000)
    inst_model_pc = np.asarray(model_pc.points)
    solidities = []
    for id in range(len(meta['obj_idx'][0])):
        inst_r = meta['poses'][:, :, id][:, 0:3]
        inst_t = np.array([meta['poses'][:, :, id][:, 3:4].flatten()])
        inst_model_array = np.add(np.dot(inst_model_pc, inst_r.T), inst_t)
        inst_full_mask = object_back_projection(intrinsics, inst_model_array)
        y_idxs, x_idxs = np.where(inst_full_mask)
        object_points = np.array([[x, y] for x, y in zip(x_idxs, y_idxs)])
        (c_x, c_y), (w, h), a = cv2.minAreaRect(object_points)
        bg_area = w*h
        fg_area = np.sum(inst_full_mask)
        sol = fg_area/bg_area
        solidities.append(sol)
    solidity = np.min(solidities)
    return True if solidity < 0.35 else False

def object_back_projection(intrinsics, pc_selected):
    """
    map the point cloud onto a 2D scene

    args:
    pc_selected: sampled point cloud from object CAD model, already translated by pose
    img_shape:   shape of the 2D scene

    return:
    object mask without occlusion
    """
    h = 1080
    w = 1920
    obj_mask = np.zeros((h, w), dtype=bool)
    pc_non_zero = pc_selected[np.where(np.all(pc_selected[:, :] != [0.0, 0.0, 0.0], axis=-1) == True)[0]]
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    # depth = pc_non_zero[:, 2]  # point_z
    coords_x = (pc_non_zero[:, 0] / pc_non_zero[:, 2] * fx + cx).astype(np.uint16)
    coords_y = (pc_non_zero[:, 1] / pc_non_zero[:, 2] * fy + cy).astype(np.uint16)
    # check index range:
    new_x = np.delete(coords_x, coords_x>=w)
    new_y = np.delete(coords_y, coords_x>=w)
    new_x = np.delete(new_x, new_y>=h)
    new_y = np.delete(new_y, new_y>=h)
    obj_mask[new_y, new_x] = True
    return obj_mask




def get_sampled_cnt(mask):
    binary = mask.astype('uint8').copy() * 255
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    max_pnum = 0
    max_id = 0
    for cid in range(len(contours)):
        pnum, _, _ = contours[cid].shape
        if pnum > max_pnum:
            max_pnum = pnum
            max_id = cid
    cnt = contours[max_id][:, 0, :]
    return cnt[::1, :]

def is_approx_convex(mask, threshold):
    sampled_cnt = get_sampled_cnt(mask)
    len_cnt = sampled_cnt.shape[0]
    point2id = {}
    id2point = {}
    for id, point in enumerate(sampled_cnt):
        point2id[tuple(point)] = id
        id2point[id] = point
    ##
    hull = cv2.convexHull(sampled_cnt)
    hull_points = hull[:, 0, :]
    hull_ids = []
    for hull_point in hull_points:
        hull_ids.append(point2id[tuple(hull_point)])
    hull_ids.sort(reverse=True)
    # for i in range(1,len(hull_ids)):
    #     if hull_ids[i] < hull_ids[i - 1]:
    #         hull_ids[i] = len_cnt - 1 - hull_ids[i]
    hull_pair_ids = [[hull_ids[i], hull_ids[i + 1]] for i in range(len(hull_ids) - 1)]
    # hull_pair_ids.append([len_cnt - 1 - hull_ids[len(hull_ids) - 1], hull_ids[0]])
    hull_pair_ids.append([hull_ids[len(hull_ids) - 1], hull_ids[0]])
    hull_pair_ids = np.array(hull_pair_ids)
    _concavities = []
    for point_id in range(len_cnt):
        if point_id in hull_ids:
            _concavities.append(0)
        else:
            judge_up = (point_id <= hull_pair_ids[:, 0])
            judge_down = (point_id > hull_pair_ids[:, 1])
            judge = np.logical_and(judge_up, judge_down)
            if np.sum(judge) == 0:
                judge[-1] = True
            bridge_ids = hull_pair_ids[judge]
            try:
                p1 = id2point[bridge_ids[0][0]]
                p2 = id2point[bridge_ids[0][1]]
            except IndexError:
                print(0)
            d = np.cross(p2 - p1, id2point[point_id] - p1) / np.linalg.norm(p2 - p1)
            _concavities.append(abs(d))
    return True if max(_concavities) < threshold else False

def mask_split_single(mask):
    sampled_cnt = get_sampled_cnt(mask)
    len_cnt = sampled_cnt.shape[0]
    point2id = {}
    id2point = {}
    for id, point in enumerate(sampled_cnt):
        point2id[tuple(point)] = id
        id2point[id] = point
    ##
    hull = cv2.convexHull(sampled_cnt)
    hull_points = hull[:, 0, :]
    hull_ids = []
    for hull_point in hull_points:
        hull_ids.append(point2id[tuple(hull_point)])
    hull_ids.sort(reverse=True)
    hull_pair_ids = [[hull_ids[i], hull_ids[i + 1]] for i in range(len(hull_ids) - 1)]
    hull_pair_ids.append([hull_ids[len(hull_ids) - 1], hull_ids[0]])
    hull_pair_ids = np.array(hull_pair_ids)

    id2concavity = {}
    for point_id in range(len_cnt):
        if point_id in hull_ids:
            id2concavity[point_id] = 0
        else:
            judge_up = (point_id <= hull_pair_ids[:, 0])
            judge_down = (point_id > hull_pair_ids[:, 1])
            judge = np.logical_and(judge_up, judge_down)
            if np.sum(judge) == 0:
                judge[-1] = True
            try:
                bridge_ids = hull_pair_ids[judge]
                p1 = id2point[bridge_ids[0][0]]
            except IndexError:
                print(0)
            p2 = id2point[bridge_ids[0][1]]
            d = np.cross(p2 - p1, id2point[point_id] - p1) / np.linalg.norm(p2 - p1)
            id2concavity[point_id] = abs(d)
    tmp = sorted(id2concavity.items(), key=lambda x: x[1])
    max_concavity_id = tmp[-1][0]
    max_concavity_point = id2point[max_concavity_id]
    possible_end_ids = [i for i in range(len_cnt) if abs(i - max_concavity_id) > 50]
    candidates = sampled_cnt[possible_end_ids]
    vec2max = candidates - max_concavity_point
    dists = np.sqrt(vec2max[:, 0] ** 2 + vec2max[:, 1] ** 2)
    concavities = np.array([v for k, v in id2concavity.items()])
    # scores = (1 + 0.1 * concavities[possible_end_ids]) / (1 * dists)
    scores = dists
    try:
        end_id = possible_end_ids[np.argmin(scores)]
    except ValueError:
        print(0)
    end_point = id2point[end_id]

    mask_map = mask.astype('uint8').copy() * 255
    cv2.line(mask_map, tuple(max_concavity_point), tuple(end_point), 0, 2)
    parts_map = label(mask_map, connectivity=1)
    part_masks = []
    for id in np.unique(parts_map):
        if id != 0 and np.sum(parts_map == id) > 20:
            part_masks.append(parts_map == id)
    return part_masks

def mask_split(mask, th_convex):
    parts = []
    def seg_node(mask):
        if is_approx_convex(mask, th_convex):  # 10
            parts.append(mask)
        else:
            parts_ = mask_split_single(mask)  ##
            if len(parts_) == 1:
                parts.append(parts_[0])
                return 0
            else:
                for p in parts_:
                    seg_node(p)
    seg_node(mask)
    return parts

'''
append part-level segmentation annotations to List obj
maximum part number is set as 3
'''
def get_fix_len_rotated_obj(bitmask,class_id,offset):
    parts_num = 3*2
    while len(offset) < parts_num:
        offset.append(0.0)
    offset = offset[:parts_num]
    y_idxs, x_idxs = np.where(bitmask)
    object_points = np.array([[x, y] for x, y in zip(x_idxs, y_idxs)])
    (c_x, c_y), (w, h), a = cv2.minAreaRect(object_points)
    rle = pm.encode(np.asarray(bitmask, order="F"))
    rle['counts'] = rle['counts'].decode()
    obj = {
        "bbox": [int(c_x), int(c_y), int(w), int(h), -a],
        "segmentation": rle,
        "category_id": class_id,
        "offset": offset
    }
    return obj

def add_part_obj(inst_mask, inst_full_mask, objs):
    parts = mask_split(mask=inst_full_mask, th_convex=20)
    if len(parts) == 1:
        uv_offset = np.hstack(((np.array(np.mean(np.where(inst_full_mask), axis=1)) - np.array(np.mean(np.where(inst_mask), axis=1))) / 50.0, np.zeros(4,)))
        objs.append(get_fix_len_rotated_obj(inst_mask, 0, (uv_offset).tolist()))
    else:
        part_centres = np.array([np.mean(np.where(h), axis=1) for h in parts])
        part_ids = [i for i, x in enumerate(parts)]
        actual_part_num = len(part_ids)
        len_offsets = (actual_part_num - 1) * 2
        for cur_id in part_ids:
            other_ids = [i for i in part_ids if i != cur_id]
            full_part_mask = parts[cur_id].astype('uint8')
            cur_part_centre = part_centres[cur_id]
            other_part_offsets = part_centres[other_ids] - cur_part_centre
            other_part_offsets = other_part_offsets.reshape((len_offsets,))
            observed_part_mask = np.logical_and(full_part_mask, inst_mask)
            observed_part_centre = np.array(np.mean(np.where(observed_part_mask), axis=1))
            uv_offset = np.hstack(((cur_part_centre - observed_part_centre) / 50.0, other_part_offsets / 200.0))
            part_visibility = np.sum(observed_part_mask) / np.sum(full_part_mask)
            if part_visibility > 0.7:
                objs.append(get_fix_len_rotated_obj(observed_part_mask, 0, (uv_offset).tolist()))

def aggregate(outputs, img):
    collected_masks = []
    pred_masks = outputs['instances'].get('pred_masks').cpu().numpy()
    valid_idxs = [i for i in range(len(pred_masks)) if np.sum(pred_masks[i]) != 0]
    if len(valid_idxs) == 0:
        return np.zeros(img.shape)
    pred_masks = pred_masks[valid_idxs]
    pred_all_offsets = outputs['instances'].get('pred_offsets').detach().cpu().numpy()[valid_idxs]
    pred_offsets = pred_all_offsets[:, 2:]
    aaa, bbb = pred_offsets.shape
    len_offset = int(bbb / 2)
    pred_centre_offsets = pred_all_offsets[:, :2] / 4.0
    assembled_idxs = []
    success_idxs = []


    centres = np.array([np.mean(np.where(h), axis=1) for h in pred_masks])[:, ::-1]
    centres = centres - pred_centre_offsets

    valid_offset_nums = np.sum(abs(pred_offsets) > 10, axis=1)
    valid_offset_nums = np.ceil(valid_offset_nums/2)
    part_num_keys = np.unique(valid_offset_nums)

    th_distv = 1e-1
    th_dist_1st = 50
    th_dist_2nd = 50
    kernel = np.ones((15, 15), np.uint8)

    def multiple_part_pairing(group_idxs, actual_len_offset):
        all_idxs = group_idxs
        res_idxs = all_idxs
        while len(res_idxs) > 0:
            cur_id = res_idxs[0]
            cur_centre = centres[cur_id]
            cur_offsets = pred_offsets[cur_id]
            cur_offsets = np.reshape(cur_offsets, (len_offset, 2))[:, ::-1]

            assembled_idxs.append(cur_id)
            res_idxs = [i for i in all_idxs if i not in assembled_idxs]

            cur_assembled_parts = []
            cur_assembled_ids = []

            for oi in range(actual_len_offset):
                res_offsets = pred_offsets[res_idxs]
                res_offsets = np.reshape(res_offsets[:,:actual_len_offset*2], (-1, actual_len_offset, 2))[:, :, ::-1]
                res_masks = pred_masks[res_idxs]
                res_centres = centres[res_idxs]
                cur_offset = cur_offsets[oi]

                res_offsets_normed = res_offsets / np.linalg.norm(res_offsets, axis=2,
                                                                  keepdims=True)  # .squeeze(axis=1)
                cur_offset_normed = cur_offset / np.linalg.norm(cur_offset, axis=0, keepdims=True)
                dists_v = (res_offsets_normed + cur_offset_normed) ** 2
                dists_v = dists_v[:, :, 0] + dists_v[:, :, 1]
                #
                if np.sum(dists_v < th_distv) == 0:
                    break  ####
                v_idxs = [i for i, x in enumerate(dists_v < th_distv) if np.sum(x) > 0]



                v_idxs = np.array(v_idxs)
                #
                des_point = np.add(cur_centre, cur_offset)
                dists = (res_centres - des_point) ** 2
                dists = dists[:, 0] + dists[:, 1]
                dists = np.sqrt(dists)
                d_v_idxs = [i for i, x in enumerate(dists[v_idxs] < th_dist_2nd) if x]
                if len(d_v_idxs) == 0:
                    break
                candidate_res_idxs = v_idxs[d_v_idxs]
                r_des_points = np.repeat(np.expand_dims(res_centres[candidate_res_idxs], axis=1), actual_len_offset, axis=1) + \
                               res_offsets[candidate_res_idxs]
                r_dists = (r_des_points - cur_centre) ** 2
                r_dists = r_dists[:, :, 0] + r_dists[:, :, 1]
                r_dists = np.sqrt(r_dists)
                r_sel_id = np.argmin(r_dists)
                rd_h, rd_w = r_dists.shape
                r_sel_id_row = int(r_sel_id / rd_w)
                r_sel_id_col = r_sel_id % rd_w
                if r_dists[r_sel_id_row, r_sel_id_col] > th_dist_1st:
                    break
                finded_res_idx = candidate_res_idxs[r_sel_id_row]
                #############
                finded_idx = res_idxs[finded_res_idx]
                res_mask = res_masks[finded_res_idx]
                cur_assembled_ids.append(finded_idx)
                # assembled_idxs.append(finded_idx)
                cur_assembled_parts.append(res_mask)
                ##
                success_idxs.append(finded_idx)
                ##
            if len(cur_assembled_parts) < actual_len_offset:
                continue
            for id in cur_assembled_ids:
                assembled_idxs.append(id)
            res_idxs = [i for i in all_idxs if i not in assembled_idxs]
            ##
            success_idxs.append(cur_id)
            ##
            cur_mask = pred_masks[cur_id]
            for rmask in cur_assembled_parts:
                cur_mask = np.logical_or(cur_mask, rmask)
            dilated_mask = cv2.dilate(cur_mask.astype('uint8'), kernel, iterations=1)
            eroded_mask = cv2.erode(dilated_mask, kernel, iterations=1)
            mask = (eroded_mask == 1)
            collected_masks.append(mask)
    for group_key in part_num_keys.astype('int'):
        group_idxs = [i for i,x in enumerate(valid_offset_nums==group_key) if x]
        if group_key == 0:
            for j in group_idxs:
                collected_masks.append(pred_masks[j])
        else:
            multiple_part_pairing(group_idxs, group_key)
    return np.array(collected_masks)

# IS_LOW = is_low_solidity(dataset_name='10')
IS_LOW = False

if __name__ == '__main__':
    if is_low_solidity():
        print('low')
