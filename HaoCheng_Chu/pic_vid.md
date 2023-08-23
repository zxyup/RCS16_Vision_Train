import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
img=cv.imread("C:/Users/18601/Desktop/codes/sample2.png")
tmp=img.copy()
tmp1=img.copy()
plt.figure(figsize=(12,8))
# plt.subplot(2,2,1)
# b=img[:,:,0]
# plt.imshow(b,cmap="gray")
# plt.title("b")

# plt.subplot(2,2,2)
# g=img[:,:,1]
# plt.imshow(g,cmap="gray")
# plt.title("g")

# plt.subplot(2,2,3)
# r=img[:,:,2]
# plt.imshow(r,cmap="gray")
# plt.title("r")

# plt.subplot(2,2,4)
# sub=cv.subtract(b,r)
# sub=cv.subtract(sub,g)
# plt.imshow(sub,cmap="gray")
# plt.title("sub")

# plt.show()

gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
ret,thresh=cv.threshold(gray,150,255,cv.THRESH_BINARY)
plt.subplot(2,2,1)
plt.imshow(thresh,cmap="gray")
plt.title("org")

kernel=np.ones((3,3),np.uint8)
plt.subplot(2,2,2)
erosion = cv.erode(thresh, kernel, iterations = 1)
test=cv.dilate(erosion,kernel,iterations = 1)
test=cv.erode(test, kernel, iterations = 1)
plt.imshow(erosion,cmap="gray")
plt.title("ero")

plt.subplot(2,2,3)
dilation=cv.dilate(thresh,kernel,iterations = 1)
plt.imshow(dilation,cmap="gray")
plt.title("dil")

plt.subplot(2,2,4)
dilation=cv.dilate(test,kernel,iterations=9)
dilation=cv.erode(dilation, kernel, iterations = 19)
plt.imshow(dilation,cmap="gray")
plt.title("dil2")

plt.show()
