import sys
import os
import json
import torch
import argparse
import detectron2.utils.comm as comm
import cv2
import random
import math
import numpy as np
from PIL import Image
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_train_loader
from detectron2.data import transforms as T
from detectron2.engine import DefaultTrainer, default_setup, hooks, launch,HookBase
from detectron2.engine.hooks import PeriodicCheckpointer
from detectron2 import model_zoo
from skimage.util import random_noise
from detectron2.evaluation import (
COCOEvaluator,
verify_results,
DatasetEvaluators
)
import matplotlib.pyplot as plt
from config import add_rmRCNN_config 

##
import low_solidity_support as loso
##


def argument_parser():
    """
    Create a parser with some common arguments used by detectron2 users.

    Args:
        epilog (str): epilog passed to ArgumentParser describing the usage.

    Returns:
        argparse.ArgumentParser:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default='./datasets/01')
    parser.add_argument("--base_config_file", default="./configs/base_rotated_maskRCNN.yaml", metavar="FILE", help="path to config file")
    parser.add_argument("--user_config_file", default="./configs/user_rmRCNN.yaml", metavar="FILE", help="path to config file")
    parser.add_argument(
        "--resume",
        default= True,
        action="store_true",
        help="Whether to attempt to resume from the checkpoint directory. "
        "See documentation of `DefaultTrainer.resume_or_load()` for what it means.",
    )
    parser.add_argument('--num_gpus', type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument("--num-machines", type=int, default=1, help="total number of machines")
    parser.add_argument(
        "--machine-rank", type=int, default=0, help="the rank of this machine (unique per machine)"
    )

    # PyTorch still may leave orphan processes in multi-gpu training.
    # Therefore we use a deterministic way to obtain port,
    # so that users are aware of orphan processes by seeing the port occupied.
    port = 2 ** 15 + 2 ** 14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    parser.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:{}".format(port),
        help="initialization URL for pytorch distributed backend. See "
        "https://pytorch.org/docs/stable/distributed.html for details.",
    )
    parser.add_argument(
        "opts",
        help="Modify config options by adding 'KEY VALUE' pairs at the end of the command. "
        "See config references at "
        "https://detectron2.readthedocs.io/modules/config.html#config-references",
        default=None,
        nargs=argparse.REMAINDER,
    )

    parser.add_argument("--debug", action="store_true")

    return parser

def random_range_noise(img, mode='pepper', range=(0.0, 0.1)):
    amount = random.uniform(range[0], range[1])
    if mode == 'pepper':
        return random_noise(img, mode=mode, amount=amount, clip=True)
    elif mode == 'gaussian':
        return random_noise(img, mode=mode, clip=True, var=amount)

class RandomPepperNoise(T.Augmentation):
    def get_transform(self, image):
        return T.ColorTransform(lambda x: (random_range_noise(x, mode='pepper', range=(0.0, 0.1))*255).astype('uint8'))
class RandomGaussianNoise(T.Augmentation):
    def get_transform(self, image):
        return T.ColorTransform(lambda x: (random_range_noise(x, mode='gaussian', range=(0.0005, 0.01))*255).astype('uint8'))
class RandomResize(T.Augmentation):

    def __init__(
        self,
        min_scale: float,
        max_scale: float,
        interp: int = Image.BILINEAR,
    ):
        super().__init__()
        self._init(locals())

    def get_transform(self, image: np.ndarray) -> T.Transform:
        # Compute the image scale and scaled size.
        input_size = image.shape[:2]
        random_scale = np.random.uniform(self.min_scale, self.max_scale)
        scaled_size = np.round(np.multiply(input_size, random_scale)).astype(int)
        return T.ResizeTransform(
            input_size[0], input_size[1], scaled_size[0], scaled_size[1], self.interp
        )

class MyCheckpointer(PeriodicCheckpointer):
    """
    A custom hook that save extra meta data to the .pth checkpoint file
    """
    def after_step(self):
        custom_info = {
        "use_loso" : loso.IS_LOW
        }
        self.step(self.trainer.iter, **custom_info)

class Trainer(DefaultTrainer):
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=DatasetMapper(cfg, is_train=True, augmentations=[
                T.RandomBrightness(0.3, 2.0),
                T.RandomContrast(0.3, 2.5),
                # RandomGaussianNoise(),
                RandomPepperNoise(),
                # T.RandomRotation([-90,90]),
                # RandomResize(0.5,1.5),
                # T.RandomCrop('relative_range',(0.3,0.3)),
                # T.RandomExtent(scale_range=(0.3, 1), shift_range=(1, 1))

    ]))

    def build_hooks(self):
        hooks = super().build_hooks()
        targetIdx = -1
        for idx,hook in enumerate(hooks): 
            if isinstance(hook, PeriodicCheckpointer):
                targetIdx = idx

        if targetIdx>=0:
            print("replace with custom checkpointer")
            hooks.pop(targetIdx)
            hooks.append(MyCheckpointer(self.checkpointer, self.cfg.SOLVER.CHECKPOINT_PERIOD))
        return hooks


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_rmRCNN_config(cfg)
    cfg.merge_from_file(args.base_config_file)
    cfg.merge_from_file(args.user_config_file)
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml") # train from ImageNet pretrained model
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def register_dataset(cfg, data_dir):  # datadir must be dataset_large
    def get_box_dicts(json_path):
        # load json file to dict and change rle str to bytes
        with open(json_path, 'r') as f:
            j = json.load(f)
        for i in range(len(j)):
            for k in range(len(j[i]['annotations'])):
                j[i]['annotations'][k]['segmentation']['counts'] = j[i]['annotations'][k]['segmentation'][
                    'counts'].encode()
        return j
    DatasetCatalog.register(cfg.DATASETS.TRAIN[0],
                            lambda: get_box_dicts(os.path.join(data_dir,'json','train.json')))
    MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).set(thing_classes=["box"])

    info = os.path.join(data_dir,'json','info.yaml')
    if os.path.exists(info):
        import yaml
        with open(info, "r") as f: 
            info = yaml.load(f, Loader=yaml.FullLoader)
            loso.IS_LOW = info.get("use_loso", False)
    print (f"use_loso : {loso.IS_LOW}")

    global DatasetMapper
    if loso.IS_LOW:
        from part_aware_rot_maskrcnn import DatasetMapper
    else:
        from rotated_maskrcnn import DatasetMapper



    # DatasetCatalog.register(cfg.DATASETS.TEST[0],
    #                         lambda: get_box_dicts(os.path.join(data_dir,'json','val.json')))
    # MetadataCatalog.get(cfg.DATASETS.TEST[0]).set(thing_classes=["box"])

def main(args):
    cfg = setup(args)
    register_dataset(cfg, data_dir=args.data_dir)
    # trainer = DefaultTrainer(cfg)
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    trainer.train()


if __name__ == '__main__':
    args = argument_parser().parse_args()

    if args.debug:
        import debugpy
        print("wait for debugger to attach to port 5678")
        debugpy.listen(5678)
        debugpy.wait_for_client()  # blocks execution until client is attached


    # main(args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )



