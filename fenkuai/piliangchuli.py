import os
import cv2
import numpy as np
import math
from math import inf
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

import torch
import torchvision
import torchvision.utils as vutils
import cv2
from torch._C import device
import torchvision.transforms as transforms
from PIL import Image
from utils import loader_imgnet,evaluate, transform_invert,save_image3,save_image4,tensor_to_PIL,image_loader,imshow
loader = transforms.Compose([
    transforms.ToTensor()#,
    #transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))
    ]) 
    
unloader = transforms.ToPILImage(
    )
   
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def l2_distance(images, adv_images):
    delta = (adv_images - images).view(len(images), -1)
    l2 = torch.norm(delta, p=2, dim=1).mean()
    return l2

def image_loader(image_name,device):
    image = Image.open(image_name).convert('RGB')
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)
def save_image(tensor,save_path):

        image = tensor.cpu().clone() # we clone the tensor to not do changes on it
        image = image.squeeze(0) # remove the fake batch dimension
        image = unloader(image)
        # r,g,b = image.split()
        # image = Image.merge('RGB',(b,g,r)) 
        image.save(save_path)






def fun(path1,path2,e):

    adv = image_loader("uap_ae.png",device=device) 
    adv = adv.squeeze(0)
    adv = torch.clamp(adv, -e*0.01,e*0.01).cuda()

    img = image_loader(path1,device=device) 
    

    result = torch.clamp(img + adv, 0, 1).cuda()
    #result = img + adv
    # l2 =l2_distance(img, result)
    # L2 = float(l2.cpu().numpy())
    # print(L2)
    #save_image4("result.png",result)
    save_image(result,path2)



path_org = "F:/org"
for file1 in os.listdir(path_org):
    for file2 in os.listdir(path_org+"/"+file1):
        path1 = path_org+"/"+file1+"/"+file2
        path2_11 = "F:/fenkuai_ae11"+"/"+file1+"/"+file2
        path2_12 = "F:/fenkuai_ae12"+"/"+file1+"/"+file2
        path2_13 = "F:/fenkuai_ae13"+"/"+file1+"/"+file2
        path2_14 = "F:/fenkuai_ae14"+"/"+file1+"/"+file2
        path2_15 = "F:/fenkuai_ae15"+"/"+file1+"/"+file2
        path2_16 = "F:/fenkuai_ae16"+"/"+file1+"/"+file2
        path2_17 = "F:/fenkuai_ae17"+"/"+file1+"/"+file2
        path2_18 = "F:/fenkuai_ae18"+"/"+file1+"/"+file2
        path2_19 = "F:/fenkuai_ae19"+"/"+file1+"/"+file2
        path2_20 = "F:/fenkuai_ae20"+"/"+file1+"/"+file2
        path2_21 = "F:/fenkuai_ae21"+"/"+file1+"/"+file2
        path2_22 = "F:/fenkuai_ae22"+"/"+file1+"/"+file2
        path2_23 = "F:/fenkuai_ae23"+"/"+file1+"/"+file2
        path2_24 = "F:/fenkuai_ae24"+"/"+file1+"/"+file2
        path2_25 = "F:/fenkuai_ae25"+"/"+file1+"/"+file2
        path2_26 = "F:/fenkuai_ae26"+"/"+file1+"/"+file2
        path2_27 = "F:/fenkuai_ae27"+"/"+file1+"/"+file2
        path2_28 = "F:/fenkuai_ae28"+"/"+file1+"/"+file2
        path2_29 = "F:/fenkuai_ae29"+"/"+file1+"/"+file2
        path2_30 = "F:/fenkuai_ae30"+"/"+file1+"/"+file2
        path2_31 = "F:/fenkuai_ae31"+"/"+file1+"/"+file2
        path2_32 = "F:/fenkuai_ae32"+"/"+file1+"/"+file2
        path2_33 = "F:/fenkuai_ae33"+"/"+file1+"/"+file2
        path2_34 = "F:/fenkuai_ae34"+"/"+file1+"/"+file2
        path2_35 = "F:/fenkuai_ae35"+"/"+file1+"/"+file2
        path2_36 = "F:/fenkuai_ae36"+"/"+file1+"/"+file2
        path2_37 = "F:/fenkuai_ae37"+"/"+file1+"/"+file2
        path2_38 = "F:/fenkuai_ae38"+"/"+file1+"/"+file2
        path2_39 = "F:/fenkuai_ae39"+"/"+file1+"/"+file2
        path2_40 = "F:/fenkuai_ae40"+"/"+file1+"/"+file2

        fun(path1,path2_11,11)
        fun(path1,path2_12,12)
        fun(path1,path2_13,13)
        fun(path1,path2_14,14)
        fun(path1,path2_15,15)
        fun(path1,path2_16,16)
        fun(path1,path2_17,17)
        fun(path1,path2_18,18)
        fun(path1,path2_19,19)
        fun(path1,path2_20,20)
        fun(path1,path2_21,21)
        fun(path1,path2_22,22)
        fun(path1,path2_23,23)
        fun(path1,path2_24,24)
        fun(path1,path2_25,25)
        fun(path1,path2_26,26)
        fun(path1,path2_27,27)
        fun(path1,path2_28,28)
        fun(path1,path2_29,29)
        fun(path1,path2_30,30)
        fun(path1,path2_31,31)
        fun(path1,path2_32,32)
        fun(path1,path2_33,33)
        fun(path1,path2_34,34)
        fun(path1,path2_35,35)
        fun(path1,path2_36,36)
        fun(path1,path2_37,37)
        fun(path1,path2_38,38)
        fun(path1,path2_39,39)
        fun(path1,path2_40,40)
        # img1 = cv2.imread(path1+"/"+file1)
        # img2 = cv2.imread(path2+"/"+file2)
        # p = psnr1(img1, img2)
        # if(p > 28):
        #     print(p)
        #     os.rename(os.path.join(path1, file1), os.path.join(path1, file2))
        #     continue
