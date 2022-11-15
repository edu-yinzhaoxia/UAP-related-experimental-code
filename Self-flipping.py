import cv2
import numpy as np
def FunctionName(a,b,c,d,path,save_path):
    
    img = cv2.imread("uap1.png")
    #path ="uap3-4.png"
    #save_path ="uap3-44.png"
    # print(img.shape)
    # #cropped = img[56:168, 56:168]  # 裁剪坐标为[y0:y1, x0:x1]
    cropped = img[a:b, c:d]  # 裁剪坐标为[y0:y1, x0:x1]
    cv2.imwrite(path, cropped)


    img = cv2.imread(path)
    w = img.shape[1]
    h = img.shape[0]
    # layoff the pictures in rows
    sz1 = w*2
    sz0 = h*2
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
                myimg1[y, x, 0] = img[y, w*2-x-1 , 0]
                myimg1[y, x, 1] = img[y, w*2-x-1 , 1]
                myimg1[y, x, 2] = img[y, w*2-x-1 , 2]
            elif(h<=y<h*2 and x<w):
                myimg1[y, x, 0] = img[h*2-y-1, x, 0]
                myimg1[y, x, 1] = img[h*2-y-1, x, 1]
                myimg1[y, x, 2] = img[h*2-y-1, x, 2]
            elif(h<=y<h*2 and w<=x<w*2):
                myimg1[y, x, 0] = img[h*2-y-1, w*2-x-1 , 0]
                myimg1[y, x, 1] = img[h*2-y-1, w*2-x-1 , 1]
                myimg1[y, x, 2] = img[h*2-y-1, w*2-x-1 , 2]

    w = myimg1.shape[1]
    h = myimg1.shape[0]
    # layoff the pictures in rows
    sz1 = w*2
    sz0 = h*2
    #print(sz1,sz0 )
    # generate a blank photo
    myimg2 = np.zeros((sz0, sz1, 3), np.uint8)
    # copy each pixels'value
    x = 0
    y = 0
    for y in range(0, h*4):
        for x in range(0, w*4):
            if (y<h and x<w):
                myimg2[y, x, 0] = myimg1[y, x, 0]
                myimg2[y, x, 1] = myimg1[y, x, 1]
                myimg2[y, x, 2] = myimg1[y, x, 2]
            elif(y<h and w<=x<w*2):
                myimg2[y, x, 0] = myimg1[y, w*2-x-1 , 0]
                myimg2[y, x, 1] = myimg1[y, w*2-x-1 , 1]
                myimg2[y, x, 2] = myimg1[y, w*2-x-1 , 2]
            elif(h<=y<h*2 and x<w):
                myimg2[y, x, 0] = myimg1[h*2-y-1, x, 0]
                myimg2[y, x, 1] = myimg1[h*2-y-1, x, 1]
                myimg2[y, x, 2] = myimg1[h*2-y-1, x, 2]
            elif(h<=y<h*2 and w<=x<w*2):
                myimg2[y, x, 0] = myimg1[h*2-y-1, w*2-x-1 , 0]
                myimg2[y, x, 1] = myimg1[h*2-y-1, w*2-x-1 , 1]
                myimg2[y, x, 2] = myimg1[h*2-y-1, w*2-x-1 , 2]

    cv2.imwrite(save_path,myimg2)

for b in range(0,169):
    for a in range(0,169):
        path ="F:/coding/CW (2)/uap/results/uap-"+str(a)+"-"+str(b)+"-1-16.png"
        save_path ="F:/coding/CW (2)/uap/results/uap-"+str(a)+"-"+str(b)+".png"
        
        FunctionName(b,b+56,a,a+56,path,save_path)
        print(a,b)
        print(path)