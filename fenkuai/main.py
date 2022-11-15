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






def fun(e):

    adv = image_loader("uap_ae.png",device=device) 
    adv = adv.squeeze(0)
    adv = torch.clamp(adv, -e*0.01,e*0.01).cuda()

    img = image_loader("org.png",device=device) 
    save_image(adv,"test_adv.png")
    save_image(img,"img.png")
    result = torch.clamp(img + adv, 0, 1).cuda()
    #result = img + adv
    l2 =l2_distance(img, result)
    L2 = float(l2.cpu().numpy())
    print(L2)
    #save_image4("result.png",result)
    save_image(result,"%s.png" %e)
for i in range(0,40,1):
    fun(i)
# gray0=np.zeros((224,224),dtype=np.uint8)
# gray0[:,:]=255
# gray255=gray0[:,:]

# Img_rgb=cv2.cvtColor(gray255,cv2.COLOR_GRAY2RGB)

# Img_rgb[:,:,0:3]=0
# cv2.imshow('(0,0,0)',Img_rgb)
# cv2.imwrite("black.png",Img_rgb)


# img = Image.open("black.png") 
# adv = Image.open("uap1-1.png") 
# im = img.load()
# ad = adv.load()


# # # 获得某个像素点的 RGB 值，像素点坐标由 [x, y] 指定
# for a in range(0,112):
#      for b in range(0,112):
#         # c = np.array(list(ad[a,b])) + np.array(list(im[2*a,2*b]))     
#         #  ad[a,b] = im[2*a,2*b] + ad[a,b] 
#         im[2*a,2*b] = ad[a,b] 
        
#         # print(im[2*a,2*b])

# img.save("uap_ae.png")
