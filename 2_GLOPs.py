import sys
sys.path.append("/mnt/AIPE-swinL/")
import torch
from collections import OrderedDict
# from evaluate.coco_eval import run_eval
from my_lib.network.rtpose_swin import get_model
from my_lib.network.openpose import OpenPose_Model, use_vgg
from torch import load
# from torchsummary import summary
from thop import profile


# Notice, if you using the
with torch.autograd.no_grad():
    # this path is with respect to the root of the project
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # weight_name = '/mnt/AIPE-swinL/b.pth'
    weight_name = r'J:\【毕业论文】【AIPE】实验数据\【24 01 13】swin-l\AIPE-swinL\b.pth'
    state_dict = torch.load(weight_name)
    # print(state_dict)

    model = get_model(trunk='swin')
    # model = openpose = OpenPose_Model(l2_stages=4, l1_stages=2, paf_out_channels=38, heat_out_channels=19)
    # model = torch.nn.DataParallel(model).cuda()
    # model.load_state_dict(state_dict)
    # model.eval()
    # model.float()
    # model = model.cuda()

    model=model.to(device)
    model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)
    model.eval()
    model.float()

    input=torch.randn(1,3,224,224)
    input = input.to(device)

    flops,params=profile(model,inputs=(input,),verbose=False)
    print(f"Parameters (Params): {params}")
    print(f"FLOPs: {flops / 1e9} GFLOPs")






# E:\swin-pose\Realtime_Multi-Person_Pose_Estimation\coco\val2017
