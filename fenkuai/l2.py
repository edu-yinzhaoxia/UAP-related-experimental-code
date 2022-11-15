import numpy as np
import matplotlib.pyplot as plt
import os
import random
import torch
import torchvision.datasets as dsets
from PIL import Image
from utils import image_loader
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
random.seed(2020)


def l2_distance(images, adv_images):
    delta = (adv_images - images).view(len(images), -1)
    l2 = torch.norm(delta, p=2, dim=1).mean()
    return l2
p = 0
org_path = "F:/org/"
ae_dir_data = "D:/jiazao/uap-"+str(0)+"-"+str(0)+"/"
zaimi_dir_data = "D:/zaimi/uap-"+str(0)+"-"+str(0)+"/"

# for root, dirs, files in os.walk(ae_dir_data):
#             # root 表示当前正在访问的文件夹路径
#             # dirs 表示该文件夹下的子目录名list
#             # files 表示该文件夹下的文件list
#             # 遍历文件
#             # for f in files:
#             #     print(os.path.join(root, f))
#     for f in files:
#                 #print(os.path.join(root, f)[-29:-4])
#         x = random.randint(0, 9)
#         adv_images = image_loader(
#         ae_dir_data+os.path.join(root, files[x])[-29:-4]+".png", device)
#         images = image_loader(
#         org_path+os.path.join(root, files[x])[-29:-4]+".png", device)
#         l2 = l2_distance(images, adv_images)
#         L2 = float(l2.cpu().numpy())
#         p = p+L2
#         break
# sum = p/1000
# print(sum)

for b in range(0,169,14):
    for a in range(0,169,14):
        p = 0
        org_path = "F:/org/"
        ae_dir_data ="D:/jiazao/uap-"+str(a)+"-"+str(b)+"/"
        zaimi_dir_data ="D:/zaimi/uap-"+str(a)+"-"+str(b)+"/"
        for root, dirs, files in os.walk(ae_dir_data):
        # root 表示当前正在访问的文件夹路径
        # dirs 表示该文件夹下的子目录名list
        # files 表示该文件夹下的文件list
            # 遍历文件
            # for f in files:
            #     print(os.path.join(root, f))
            for f in files:
                #print(os.path.join(root, f)[-29:-4])
                x = random.randint(0,9)
                adv_images = image_loader(ae_dir_data+os.path.join(root, files[x])[-29:-4]+".png",device)
                images = image_loader(org_path+os.path.join(root, files[x])[-29:-4]+".png",device)
                l2 =l2_distance(images, adv_images)
                L2 = float(l2.cpu().numpy())
                p=p+L2
                break
        sum = p/1000
        print(sum)
