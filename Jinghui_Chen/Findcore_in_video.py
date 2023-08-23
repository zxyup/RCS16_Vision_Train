import numpy as np
import cv2 as cv

blue_color = (255, 0, 0)   # 蓝色
green_color = (0, 255, 0)  # 绿色
red_color = (0, 0, 255)    # 红色
yellow_color = (0, 255, 255)  # 黄色
white_color = (255,255,255) #白色

cap = cv.VideoCapture('video/sample.mp4')
while cap.isOpened():
    ret, img = cap.read()
    # 如果正确读取帧，ret为True
    if not ret:
        print("Can't receive img (stream end?). Exiting ...")
        break
    g_channel = img[:,:,1]  # 获取绿色通道
    ret, thresh = cv.threshold(g_channel, 220, 255, cv.THRESH_BINARY)
    
    kernel = np.ones((3, 3), np.uint8)
    dilation = cv.dilate(thresh, kernel, iterations=7)
    erosion = cv.erode(dilation,kernel,iterations=27)

    contours, hierachy = cv.findContours(erosion, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    flag=0
    # 获取最后一个白点的中心
    len_1=len(contours)
    if len_1 > 0:
        flag=1
        last_contour = contours[-1]  # 获取最后一个轮廓
        M = cv.moments(last_contour)  # 计算轮廓的矩
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])  # 计算中心点的x坐标
            cy = int(M["m01"] / M["m00"])  # 计算中心点的y坐标
            # 进一步处理中心点坐标等操作
        # contours_1, hierachy_1 = cv.findContours(dilation, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        # #原来是你!!!!!
        if flag:
            radius=40
            cv.circle(img, (cx, cy), radius, yellow_color, thickness=5)
            cv.circle(img, (cx, cy), radius+10, green_color, thickness=5)
            cv.circle(img, (cx, cy), radius+20, red_color, thickness=5)

    cv.imshow('img', img)
    if cv.waitKey(1) == ord('q'):
        break
cap.release()
cv.destroyAllWindows()