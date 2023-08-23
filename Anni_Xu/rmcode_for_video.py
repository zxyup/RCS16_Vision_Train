#基本搞定。。。
#处理视频
import numpy as np
import cv2 as cv
cap = cv.VideoCapture('../video1.MP4')
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # 逐帧捕获
    ret, img = cap.read()
    # 如果正确读取帧，ret为True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    img=cv.resize(img,(-1,-1),fx=0.5,fy=0.5)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # 二值化处理
    ret, thresh = cv.threshold(gray, 200, 255, cv.THRESH_BINARY)
    # 扩张
    kernel = np.ones((2, 1), np.uint8)
    erosion = cv.erode(thresh, kernel, iterations=1)
    # 寻找轮廓
    contours, hierarchy = cv.findContours(erosion, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    for item in contours:
        area = cv.contourArea(item)
        perimeter = cv.arcLength(item, True)
        #print(area)
        #if area > 20 and area < 40:
        #艰难的调参
        if area > 1200 and area < 1800:
            if perimeter>140 and perimeter<200:
            # 绘制轮廓
                img_contour = cv.drawContours(img, item, -1, (0, 0, 255), 5)
        # elif area>15000:


        #img_contour = cv.drawContours(img, item, -1, (0, 255, 0), 2)
    cv.imshow('image', img )
    if cv.waitKey(1) == ord('q'):
        break
# 完成所有操作后，释放捕获器
cap.release()
cv.destroyAllWindows()