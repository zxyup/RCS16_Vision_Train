import cv2 as cv
import numpy as np
import random

img=cv.imread("C:/Users/18601/.conda/envs/opencv/sample.jpg")
#改变图像尺寸
img=cv.resize(img,None,fx=0.3,fy=0.3)
#改变ROI
tem=img[15:450,380 :805]
img=tem
#创建新背景
img1 = np.zeros(img.shape, np.uint8)
#变成灰度图后 改变阈值
img=cv.cvtColor(img,cv.COLOR_BGR2GRAY)

ret,img = cv.threshold(img,180,255,cv.THRESH_BINARY)


contours, hierarchy = cv.findContours(img,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)




RoyalBlue=(65 ,105 ,225	)
HotPink=(255 ,105 ,180)

for i in contours:
    # 通过面积
    area= cv.contourArea(i)
    # if area>9000:
    #     cv.drawContours(img1,[i],0,HotPink,3)
    # elif area>500:
    #     cv.drawContours(img1,[i],0,RoyalBlue,3)
    # elif area>300:
    #     cv.drawContours(img1, [i], 0, HotPink, 3)

    # 通过顶点个数
    if len(i)>30:

        cv.drawContours(img1,[i],0,RoyalBlue,3)

cv.imshow("image3", img1)
cv.waitKey(0)
cv.destroyAllWindows()
