import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import imageio
import torch
from torch._C import device
import torchvision.transforms as transforms
from PIL import Image
from utils import loader_imgnet, model_imgnet, evaluate, transform_invert,save_image3,save_image4,tensor_to_PIL,image_loader,imshow
def test():
    # loader使用torchvision中自带的transforms函数
    loader = transforms.Compose([
    transforms.ToTensor()]) 
    
    unloader = transforms.ToPILImage()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')

    def save_image(save_path,tensor):
        dir = 'results'
        image = tensor.cpu().clone() # we clone the tensor to not do changes on it
        image = image.squeeze(0) # remove the fake batch dimension
        image = unloader(image)
        if not os.path.exists(dir):
            os.makedirs(dir)
        image.save(save_path)

    sys.path.append(os.path.realpath('..'))


    dir_data = 'F:/ae/ex-imagnet/resnet50\org-a/'
    # dir_data1 = "E:/desk/RDH/zaimi/"
    dir_uap = 'F:/coding/CW-2/uap/uaps/imagenet/'
    loader = loader_imgnet(dir_data, 10000, 1) # evaluate on 10,000 validation images

    # load model
    model = model_imgnet('resnet50')

    val_transform = transforms.Compose([
            #transforms.Resize(224),
            #transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
    

    uap = torch.load(dir_uap + 'sgd-vgg16-eps10.pth').cuda() #生成的对抗扰动
    save_image3("uap_vgg16.png",uap)
    # _, _, top1acc, top5acc, outputs, labels = evaluate(model, loader, uap = uap)
    # print(sum(outputs == labels) / len(labels))

   

test()