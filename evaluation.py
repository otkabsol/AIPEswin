import unittest
import sys
sys.path.append("/mnt/AIPE-swinL/")
import torch
from collections import OrderedDict
from evaluate.coco_eval import run_eval
from my_lib.network.rtpose_swin import get_model
from my_lib.network.openpose import OpenPose_Model, use_vgg
from torch import load

#Notice, if you using the 
with torch.autograd.no_grad():
    # this path is with respect to the root of the project
    #weight_name = r'D:\【毕业论文】相关资料\【22 05 22】swin-pose代码\trained_model\model_0.pth'
    # weight_name = r'D:\【毕业论文】相关资料\【22 05 22】swin-pose代码\【swin_B】Realtime_Multi-Person_Pose_Estimation\Realtime_Multi-Person_Pose_Estimation\evaluate\best_pose.pth'
    #weight_name = '/mnt/AIPE-swinL/evaluate/best_pose.pth'
    weight_name='/mnt/AIPE-swinL/b.pth'
    state_dict = torch.load(weight_name)
    print(state_dict)
    
    # new_state_dict = OrderedDict()
    # for k,v in state_dict.items():
        # name = k[6:]
        # new_state_dict[name]=v
        
    model = get_model(trunk='swin')
    #model = openpose = OpenPose_Model(l2_stages=4, l1_stages=2, paf_out_channels=38, heat_out_channels=19)
    model = torch.nn.DataParallel(model).cuda()
    model.load_state_dict(state_dict)
    model.eval()
    model.float()
    model = model.cuda()
    
    # The choice of image preprocessing include: 'rtpose', 'inception', 'vgg' and 'ssd'.
    # If you use the converted model from caffe, it is 'rtpose' preprocess, the model trained in 
    # this repo used 'vgg' preprocess
    preprocess = 'vgg'
    # run_eval(image_dir= '/home/hkl/project/My_Dataset/COCO2017/val2017', anno_file = '/home/hkl/project/My_Dataset/COCO2017/annotations/person_keypoints_val2017.json', vis_dir = '/home/hkl/project/My_Dataset/COCO2017/vis_val2017', model=model, preprocess=preprocess)
    run_eval(image_dir=r'/mnt/coco2017/val2017',
             anno_file=r'/mnt/coco2017/annotations/person_keypoints_val2017.json',
             vis_dir=r'/mnt/AIPE-swinL/vis_model', model=model, preprocess=preprocess)

# E:\swin-pose\Realtime_Multi-Person_Pose_Estimation\coco\val2017
