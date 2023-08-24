import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
cap = cv.VideoCapture('./大能量机关8m激活过程（无10环灯效）.MP4')
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # 逐帧捕获
    ret, frame = cap.read()
    # 如果正确读取帧，ret为True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # 我们在框架上的操作到这里
    img1 = cv.imread('ColorfulTarget.png', cv.IMREAD_GRAYSCALE)  # 索引图像
    img2 = cv.resize(frame,(-1,-1),fx=0.5,fy=0.5)
    # 初始化SIFT描述符
    sift = cv.xfeatures2d.SIFT_create()
    # 基于SIFT找到关键点和描述符
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    # 默认参数初始化BF匹配器
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    # 应用比例测试
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])
    # cv.drawMatchesKnn将列表作为匹配项。
    img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    #plt.imshow(img3), plt.show()
    #if cv.waitKey(1) == ord('q'):
     #   break
    cv.imshow('img3',img3)
    cv.waitKey(0)
# 完成所有操作后，释放捕获器
cap.release()
cv.destroyAllWindows()