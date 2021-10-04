import cv2
import json
from matplotlib.pyplot import *
import os
from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog
import argparse
##
import low_solidity_support as loso
##

def get_dicts(json_path):
    with open(json_path, 'r') as f:
        j = json.load(f)
    for i in range(len(j)):
        for k in range(len(j[i]['annotations'])):
            j[i]['annotations'][k]['segmentation']['counts'] = j[i]['annotations'][k]['segmentation']['counts'].encode()
    return j

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_path', help="json file of the train dataset", default="/home/smartgrasping/Roy/TrainPipeLine6D/workspace/test_loso/train_seg/input/json/train.json")
    parser.add_argument('--loso', help="enable loso mode", action='store_true')
    args = parser.parse_args()
    
    json_path = args.json_path
    loso.IS_LOW = args.loso
    
    dicts = get_dicts(json_path)
    DatasetCatalog.register('screw', lambda : dicts)
    MetadataCatalog.get('screw').set(thing_classes=["object"])
    meta = MetadataCatalog.get('screw')


    for d in dicts:
        img = cv2.imread(d['file_name'])
        visualizer = Visualizer(img[:, :, ::-1], metadata=meta)
        out = visualizer.draw_dataset_dict(d)
        vis_out = out.get_image()[:,:,::-1]/255.0
        vis_out = vis_out.astype('float32')
        ##
        if loso.IS_LOW:
            for obj in d['annotations']:
                c_x,c_y,_,_,_ = obj["bbox"]
                len_off = len(obj['offset'])/2 - 1
                for i in range(int(len_off)):
                    des = (int(c_x + 50*obj['offset'][0] + 200.0 * obj['offset'][2*(i+1)+1]), int(c_y + 50*obj['offset'][1] + 200.0 * obj['offset'][2*(i+1)]))
                    vis_out = cv2.arrowedLine(img= vis_out, pt1=(c_x,c_y), pt2=des,color=[1.0,0,0],thickness=2)
        ##      
        imshow(vis_out)
        show()


