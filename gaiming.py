import os
import cv2
import numpy as np
import math


def psnr1(target, ref):
    target_data = np.array(target, dtype=np.float64)
    ref_data = np.array(ref, dtype=np.float64)
    # 直接相减，求差值
    diff = ref_data - target_data
    # 按第三个通道顺序把三维矩阵拉平
    diff = diff.flatten('C')
    # 计算MSE值
    rmse = math.sqrt(np.mean(diff ** 2.))
    # 精度
    eps = np.finfo(np.float64).eps
    if(rmse == 0):
        rmse = eps
    return 20*math.log10(255.0/rmse)
path1 = "E:/desk/org1"
path2 = "E:/desk/RDH/ae"
for file1 in os.listdir(path1):
    for file2 in os.listdir(path2):
        # print(path2+"/"+file2)
        img1 = cv2.imread(path1+"/"+file1)
        img2 = cv2.imread(path2+"/"+file2)
        p = psnr1(img1, img2)
        if(p > 28):
            print(p)
            os.rename(os.path.join(path1, file1), os.path.join(path1, file2))
            continue
