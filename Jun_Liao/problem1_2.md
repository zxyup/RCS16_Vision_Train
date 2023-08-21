import cv2 as cv
img = cv.imread('sample.jpg')#先读图片
resized2 = cv.resize(img,(750 , 375), interpolation=cv.INTER_AREA)     #然后调图像的大小
resized = cv.resize(img, (750 , 375), interpolation=cv.INTER_AREA)
resized = resized[0:275,178:550]                                        #先确定目标区域的大致位置，以此可以过滤大部分的干扰项
gray = cv.cvtColor(resized, cv.COLOR_BGR2GRAY)                             #把图像变成灰图才能进行
ret, thresh = cv.threshold(gray, 215, 265, cv.THRESH_TOZERO_INV) #阈值过滤,根据图片的观察目标区域的颜色多为白色确定maxval
contours, hierarchy = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
val1 = []
val2 = []
for i in contours:            #通过计算各轮廓所围成的面积来筛选掉不必要的区域
    area = cv.contourArea(i)
    len = cv.arcLength(i, True)
    print(area)                             #可以先把所有的面积输出
    if area<5000 and area>900:
        val1.append(i)
    elif area<900 and area>400:
        val2.append(i)
cv.drawContours(resized, val1, -1, (0, 255, 0), 2)
cv.drawContours(resized, val2, -1, (0, 0, 255), 2)
cv.imshow("drawing", resized)
cv.imshow("image", resized2)
cv.waitKey(0)
cv.destroyAllWindows()
