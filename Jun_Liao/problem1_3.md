#利用findContours函数中的RETR_TREE参数
import cv2 as cv
img = cv.imread('sample.jpg')
resized2 = cv.resize(img,(750 , 375), interpolation=cv.INTER_AREA)
resized = cv.resize(img, (750 , 375), interpolation=cv.INTER_AREA)
gray = cv.cvtColor(resized, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(gray, 215, 265, cv.THRESH_TOZERO)
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
val1 = []
val2 = []
cnt = 0
for i in contours:
    if hierarchy[0][cnt][2]==-1 and hierarchy[0][cnt][3]==-1:      #易得目标区域外的点既没有子轮廓也没有父轮廓，因此可将非目标轮廓排除
        cnt=cnt+1
        continue
    if hierarchy[0][cnt][3]==-1:                                   #大轮廓没有父轮廓，所以hierarchy的第四个元素为-1
        val1.append(i)
    elif hierarchy[0][cnt][2]==-1:                                 #小轮廓没有子轮廓，所以hierarchy的第三个元素为-1
        val2.append(i)
    cnt = cnt+1
cv.drawContours(resized, val1, -1, (0, 255, 0), 2)
cv.drawContours(resized, val2, -1, (0, 0, 255), 2)
cv.imshow("drawing", resized)
cv.imshow("image", resized2)
cv.waitKey(0)
cv.destroyAllWindows()