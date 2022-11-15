import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import imageio
import torch
from torch._C import device
import torchvision.transforms as transforms
from PIL import Image
from utils import loader_imgnet, model_imgnet, evaluate, transform_invert, save_image3, save_image4, tensor_to_PIL, image_loader, imshow
import numpy as np

import random

import cv2

from matplotlib import pyplot as plt


def test(dir_data):

    # evaluate on 10,000 validation images
    loader = loader_imgnet(dir_data, 10000, 1)
    # load model
    model = model_imgnet('resnet50')

    _, _, top1acc, top5acc, outputs, labels = evaluate(model, loader, uap=None)
    print(sum(outputs == labels) / len(labels))

# for b in range(0,169,14):
#     for a in range(0,169,14):
#         ae_dir_data ="D:/ae/uap-"+str(a)+"-"+str(b)+"/"
#         test(ae_dir_data)

zaimi_dir_data = "F:/org/"
test(zaimi_dir_data)
print("---------------------------------------------------------")
# for b in range(0, 169, 14):
#     for a in range(0, 169, 14):
#         zaimi_dir_data = "D:/zaimi/uap-"+str(a)+"-"+str(b)+"/"
#         test(zaimi_dir_data)
# for i in range(1,40,1):
#     zaimi_dir_data = "F:/fenkuai_ae%s/" %i
#     test(zaimi_dir_data)
print("---------------------------------------------------------")
for p in range(1,40,1):
    zaimi_dir_data = "F:/ae/uap-%s/" %p
    test(zaimi_dir_data)




