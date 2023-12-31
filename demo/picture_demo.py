import re
import os,sys 
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
sys.path.insert(0,parentdir)  
import cv2
import math
import time
import scipy
import argparse
import matplotlib
import numpy as np
import pylab as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from scipy.ndimage.morphology import generate_binary_structure
from scipy.ndimage.filters import gaussian_filter, maximum_filter

from my_lib.network.rtpose_swin import get_model
from my_lib.network import im_transform
from evaluate.coco_eval import get_outputs, handle_paf_and_heat
from my_lib.utils.common import Human, BodyPart, CocoPart, CocoColors, CocoPairsRender, draw_humans
from my_lib.utils.paf_to_pose import paf_to_pose_cpp
from my_lib.config import cfg, update_config
from my_lib.datasets import coco, transforms, datasets

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', help='experiment configure file name',
                    default='../experiments/vgg19_368x368_sgd.yaml', type=str)
parser.add_argument('--weight', type=str,
                    default='best_pose.pth')
parser.add_argument('opts',
                    help="Modify config options using the command-line",
                    default=None,
                    nargs=argparse.REMAINDER)
args = parser.parse_args()

# update config file
update_config(cfg, args)



model = get_model('swin')
model.load_state_dict(torch.load(args.weight))
model = torch.nn.DataParallel(model).cuda()
model.float()
model.eval()

test_image = 'ski.jpg'
oriImg = cv2.imread(test_image) # B,G,R order
shape_dst = np.min(oriImg.shape[0:2])

# Get results of original image

with torch.no_grad():
    paf, heatmap, im_scale = get_outputs(oriImg, model,  'rtpose')
          
print(im_scale)
humans = paf_to_pose_cpp(heatmap, paf, cfg)
        
out = draw_humans(oriImg, humans)
cv2.imwrite('ski-output+.jpg',out)

