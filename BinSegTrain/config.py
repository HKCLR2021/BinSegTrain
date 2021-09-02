'''
author: Feng Yidan
email: fengyidan1995@126.com
'''

def add_rmRCNN_config(cfg):
    """
        Add configs for rotated MaskRCNN
    """
    cfg.INPUT.IS_ROTATED = True
