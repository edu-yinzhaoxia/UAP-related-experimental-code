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
def test(uap_path,a):
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


    dir_data = 'F:/org/'
    # dir_data1 = "E:/desk/RDH/zaimi/"
    dir_uap = './uaps/imagenet/'
    loader = loader_imgnet(dir_data, 10000, 1) # evaluate on 10,000 validation images

    # load model
    model = model_imgnet('resnet50')

    val_transform = transforms.Compose([
            #transforms.Resize(224),
            #transforms.CenterCrop(224),
            transforms.ToTensor()
        ])

    # uap = torch.load(dir_uap + 'sgd-resnet50-eps10.pth').cuda() #生成的对抗扰动
    # # save_image3("uap1.png",uap)
    uap_pr1 =  image_loader(uap_path,device)#场外加载的对抗扰动 .png Image转 tensor
    uap1 = uap_pr1.squeeze(0)
    uap1 = torch.clamp(uap1, -a*0.01, a*0.01)



    # #plt.imshow(uap_pr)# 显示tensor
    # uap1_max = torch.max(uap1)


    _, _, top1acc, top5acc, outputs, labels= evaluate(a,model, loader, uap = uap1)
    print(sum(outputs == labels) / len(labels))

# for b in range(0,169,14):

uap_path ="uap-89-95.png"
test(uap_path,a=0)


