#利用findContours函数中的RETR_CCOMP参数（同problem1—_3的处理）
import cv2 as cv
img = cv.imread('sample.jpg')
resized2 = cv.resize(img,(750 , 375), interpolation=cv.INTER_AREA)
resized = cv.resize(img, (750 , 375), interpolation=cv.INTER_AREA)
gray = cv.cvtColor(resized, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(gray, 215, 265, cv.THRESH_TOZERO)
contours, hierarchy = cv.findContours(thresh, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
print(hierarchy)
val1 = []
val2 = []
cnt = 0
for i in contours:
    if hierarchy[0][cnt][2] ==-1 and hierarchy[0][cnt][3] ==-1:
        cnt=cnt+1
        continue
    if hierarchy[0][cnt][3] ==-1:
        val1.append(i)
    elif hierarchy[0][cnt][2] ==-1:
        val2.append(i)
    cnt = cnt + 1
cv.drawContours(resized, val1, -1, (0, 255, 0), 2)
cv.drawContours(resized, val2, -1, (0, 0, 255), 2)
cv.imshow("drawing", resized)
cv.imshow("image", resized2)
cv.waitKey(0)
cv.destroyAllWindows()