#能量机关视频视频识别目标击打区域
import cv2 as cv
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
def is_circular(contour,threshold1=0.89):                                   #在这里定义一个判断轮廓是否接近圆，用参数threshold1来确定轮廓是否拟合圆，范围在[0,1]。
    perimeter=cv.arcLength(contour,True)
    area=cv.contourArea(contour)
    if perimeter==0:
        return False
    circularity=4*3.1415926*area/(perimeter*perimeter)
    return circularity>threshold1
cap = cv.VideoCapture('day3_3.MP4')
while cap.isOpened():
    ret, img = cap.read()
    img=cv.resize(img,(750,435),interpolation=cv.INTER_AREA)
    # 如果正确读取帧，ret为True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    ret,thresh=cv.threshold(gray,190,255,cv.THRESH_BINARY)

    #为了进一步筛去干扰项，这里使用漫水法
    seed_point = (100, 100)
    fill_color = (255, 0, 0)                                                # 定义填充颜色
    mask = np.zeros((thresh.shape[0] + 2, thresh.shape[1] + 2), np.uint8)   # 创建掩膜图像，全黑
    # 执行漫水法
    cv.floodFill(thresh, mask, seed_point, fill_color)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (2,2))
    closing = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)
    closing = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)
    contours, hierarchy = cv.findContours(closing, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    cnt=0
    val=160                                                                 #这里先定义一个面积的初值
    count=0                                                                 #以count记录最小面积的位置
    for contour in contours:
        if is_circular(contour) and val>cv.contourArea(contour):
            count=cnt
            val=cv.contourArea(contour)
        cnt=cnt+1
    if count!=0:                                                            #这里可以通过判断count有无变化来判断图中是否有目标区域，若无则无需处理
        cv.drawContours(img,contours,count,(0,0,255),3)
    cv.imshow('frame', img)
    if cv.waitKey(25) == ord('q'):
        break
cap.release()
cv.destroyAllWindows()