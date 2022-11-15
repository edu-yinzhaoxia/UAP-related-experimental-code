import cv2
import numpy as np
from numpy.lib.function_base import _angle_dispatcher
def FunctionName(ae_path,path,save_path):
    
    img = cv2.imread(ae_path)  #224*224
    img2 = cv2.resize(img,(56,56))
    cv2.imwrite(path, img2)


    img_r = img2.copy()
    w = img_r.shape[1]
    h = img_r.shape[0]
    # layoff the pictures in rows
    sz1 = w#*2
    sz0 = h#*2
    #print(sz1,sz0 )
    # generate a blank photo
    myimg1 = np.zeros((sz0, sz1, 3), np.uint8)
    # copy each pixels'value
    x = 0
    y = 0
    for y in range(0, h*4):
        for x in range(0, w*4):
            if (y<h and x<w):
                myimg1[y, x, 0] = img[y, x, 0]
                myimg1[y, x, 1] = img[y, x, 1]
                myimg1[y, x, 2] = img[y, x, 2]
            elif(y<h and w<=x<w*2):
                myimg1[y, w*2-x-1 , 0] = img[y, x, 0]
                myimg1[y, w*2-x-1 , 1] = img[y, x, 1]
                myimg1[y, w*2-x-1 , 2] = img[y, x, 2] 
            elif(h<=y<h*2 and x<w):
                myimg1[h*2-y-1, x, 0] = img[y, x, 0]
                myimg1[h*2-y-1, x, 1] = img[y, x, 1]
                myimg1[h*2-y-1, x, 2] = img[y, x, 2]
            elif(h<=y<h*2 and w<=x<w*2):
                myimg1[h*2-y-1, w*2-x-1 , 0] = img[y, x, 0]
                myimg1[h*2-y-1, w*2-x-1 , 1] = img[y, x, 1]
                myimg1[h*2-y-1, w*2-x-1 , 2] = img[y, x, 2]
    cv2.imwrite(save_path,myimg1)
    # w = myimg1.shape[1]
    # h = myimg1.shape[0]
    # # layoff the pictures in rows
    # sz1 = w*2
    # sz0 = h*2
    # #print(sz1,sz0 )
    # # generate a blank photo
    # myimg2 = np.zeros((sz0, sz1, 3), np.uint8)
    # # copy each pixels'value
    # x = 0
    # y = 0
    # for y in range(0, h*4):
    #     for x in range(0, w*4):
    #         if (y<h and x<w):
    #             myimg2[y, x, 0] = myimg1[y, x, 0]
    #             myimg2[y, x, 1] = myimg1[y, x, 1]
    #             myimg2[y, x, 2] = myimg1[y, x, 2]
    #         elif(y<h and w<=x<w*2):
    #             myimg2[y, x, 0] = myimg1[y, w*2-x-1 , 0]
    #             myimg2[y, x, 1] = myimg1[y, w*2-x-1 , 1]
    #             myimg2[y, x, 2] = myimg1[y, w*2-x-1 , 2]
    #         elif(h<=y<h*2 and x<w):
    #             myimg2[y, x, 0] = myimg1[h*2-y-1, x, 0]
    #             myimg2[y, x, 1] = myimg1[h*2-y-1, x, 1]
    #             myimg2[y, x, 2] = myimg1[h*2-y-1, x, 2]
    #         elif(h<=y<h*2 and w<=x<w*2):
    #             myimg2[y, x, 0] = myimg1[h*2-y-1, w*2-x-1 , 0]
    #             myimg2[y, x, 1] = myimg1[h*2-y-1, w*2-x-1 , 1]
    #             myimg2[y, x, 2] = myimg1[h*2-y-1, w*2-x-1 , 2]

    # cv2.imwrite(save_path,myimg2)
# for num in range(1,17):
#     ae_path = "uap"+str(num)+".png"
#     path ="F:/coding/CW-2/uap/results/uap"+str(num)+"-1/16.png"
#     save_path ="F:/coding/CW-2/uap/results/uap"+str(num)+".png"
#     FunctionName(ae_path,path,save_path)
#     print(path)
ae_path = "sk-2.png"
path ="F:/coding/CW-2/uap/fenkuai/sk-1.png"
save_path ="F:/coding/CW-2/uap/sk2.png"
FunctionName(ae_path,path,save_path)
print(path)