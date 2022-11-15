import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import imageio

import cv2
import numpy as np
from matplotlib import pyplot as plt
 
x = np.uint8([250])
y = np.uint8([10])
print(cv2.add(x, y))
print(x+y)                               
 
img1 = cv2.imread("111.png")  # 图片1
img2 = cv2.imread("uap-sgd-resnet50-eps10.png")  # 图片2
#add = cv2.add(img1, img2)  # 两个图像相加
subtract = cv2.subtract(img1, img2)  # 两个图像相减
#multiply = cv2.multiply(img1, img2)  # 两个图像相乘
#divide = cv2.divide(img1, img2)  # 两个图像相除
 
plt.subplot(131), plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB), 'gray'), plt.title('img1')
plt.subplot(132), plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB), 'gray'), plt.title('img2')
plt.subplot(133), plt.imshow(cv2.cvtColor(subtract, cv2.COLOR_BGR2RGB), 'gray'), plt.title('subtract')


imageio.imwrite("222.png", subtract)