
import cv2
import os
import numpy as np
import math
from skimage.measure import compare_ssim
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import torch
import torchvision
loader = transforms.Compose([
    transforms.ToTensor()#,
    #transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))
    ]) 
    
unloader = transforms.ToPILImage()
   
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
def image_loader(image_name,device):
    image = Image.open(image_name).convert('RGB')
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)
def psnr(img1, img2):
    mse = np.mean((img1/255. - img2/255.) ** 2)
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    return( 20 * math.log10(PIXEL_MAX / math.sqrt(mse)))


def ssim(imageA, imageB):
    # 为确保图像能被转为灰度图
    imageA = np.array(imageA, dtype=np.uint8)
    imageB = np.array(imageB, dtype=np.uint8)
    # 通道分离，注意顺序BGR不是RGB
    (B1, G1, R1) = cv2.split(imageA)
    (B2, G2, R2) = cv2.split(imageB)
    # convert the images to grayscale BGR2GRAY
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
    (score0, diffB) = compare_ssim(B1, B2, full=True)
    (score1, diffG) = compare_ssim(G1, G2, full=True)
    (score2, diffR) = compare_ssim(R1, R2, full=True)
    aveScore = (score0+score1+score2)/3
    print(aveScore)

def l2_distance(images, adv_images):
    delta = (adv_images - images).view(len(images), -1)
    l2 = torch.norm(delta, p=2, dim=1).mean()
    return l2
# for i in range(0, 10, 1):
#     gt = cv2.imread('org.png')
#     img = cv2.imread('%s.png' % i)
#     print(i)
#     psnr(gt, img)
# for i in range(0, 10, 1):
#     gt = image_loader('org.png',device)
#     img = image_loader('%s.png' % i,device)
#     print(i)
#     print(l2_distance(gt, img))
for a in range(1,40,1):
        p = 0
        ae_dir_data = "F:/ae/uap-0/"
        # ae_dir_data ="D:/jiazao/uap-"+str(a)+"-"+str(b)+"/"
        zaimi_dir_data ="F:/fenkuai_ae"+str(a)+"/"
        for root, dirs, files in os.walk(ae_dir_data):
        # root 表示当前正在访问的文件夹路径
        # dirs 表示该文件夹下的子目录名list
        # files 表示该文件夹下的文件list
            # 遍历文件
            for f in files:
                #x = random.randint(3,8)

                p1 = "F:/"+ zaimi_dir_data[3:]+f[:-10]+"/"+files[4][:-4]+".png"
                p2 = "F:/"+ ae_dir_data[3:]+f[:-10]+"/"+files[4][:-4]+".png"

                adv_images = cv2.imread(p1)
                images = cv2.imread(p2)
                L =psnr(images, adv_images)
                # print(dtype(L))
                #L2 = float(l2.cpu().numpy())
                p=p+L
                break
        sum = p/1000
        print(sum)