import numpy as np
import cv2 as cv

# cv.imread(图像路径,读取图像的方式)
# 读进来的格式是BGR格式
# 读取图像的方式：
# cv.IMREAD_COLOR：-- 1 加载彩色图像。任何图像的透明度都会被忽视。它是默认标志。
# cv.IMREAD_GRAYSCALE：-- 0 以灰度模式加载图像
# cv.IMREAD_UNCHANGED：-- -1 加载图像，包括alpha通道
im = cv.imread('./sample.jpg')

# cv.resize(src, dsize, dst, fx, fy, interpolation)
# cv.resize(原图像,缩放后的图像大小(元组),目标图像dst(一般不传递参数或设置为None),x和y方向上的缩放比例,插值方式)
# 缩放后图像的大小,如(500,400)
# x 和 y方向上的缩放比例
## 如 cv.resize(im, (0,0), None, fx=0.5 , fy=0.3)
### 注意：如果要使用比例放缩，那么 缩放后的图像大小dsize 要写成一个不合法的形式(0,0)
im = cv.resize(im,(500,400))

# dimg = im[y上: y下 , x左: x右]
# dimg->被裁剪后的图片
# im -> 需要裁减的图片
# 具体的值需要慢慢调试出来
im = im[7:220,149:328]

#cv.cvtColor(原图, 转换模式)
##返回值是转换后的图片
##常用的转换通道有： cv.COLOR_BGR2GRAY(三通道图片转为灰度图片) , cv.COLOR_BGR2RGB(转为正常通道)
###注意: cv.COLOR_BGR2GRAY 此转换模式不能将灰度图片再度转为灰度图片
imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)


# ret, thresh = cv.threshold(需要处理的图像, 阈值, 分配的值, 阈值处理模式选择)
## 需要处理的图像必须为灰色图像
## 阈值： 分界值
## 分配的值： 如果一个像素的灰度值大于或小于阈值(取决于参数4的选择)，则会被赋予分配的值
''' 常见阈值处理模式: 
cv.THRESH_BINARY  -> 大于阈值则赋予分配的值, 小于则分配 0
cv.THRESH_BINARY_INV -> 小于阈值则赋予分配的值, 大于则分配 0
cv.THRESH_TRUNC -> 超过阈值则赋予被分配的值， 小于则分配原有的值
返回值:
ret->阈值
thresh->返回处理后的图片(二值化处理)     
'''
ret, thresh = cv.threshold(imgray, 180, 255, cv.THRESH_BINARY)


# contours, hierarchy = cv.findContours(二值化图像, 轮廓检索方式, 轮廓的估计方法)
# contours： 返回一个list,list中每个元素都是图像中的一个轮廓
# hierarchy: 返回一个list,其中的元素个数和轮廓个数相同
# hierachy 中每个元素的列表对应 [Next, Previous, First Child, Parent]
# contours[i]对应hierarchy[0][i][0~3]
'''
轮廓检索方式：
cv2.RETR_LIST: 所有轮廓从处于同一层级
cv2.RETR_TREE: 完整建立轮廓的层级从属关系
cv2.RETR_EXTERNAL: 只寻找最高层级的轮廓
cv2.RETR_CCOMP: 把所有轮廓分为两个层级, 不是外层就是里层
轮廓估计方式:
cv2.CHAIN_APPROX_NONE: 储存所有边界点
cv2.CHAIN_APPROX_SIMPLE: 储存边界最少的点
'''
contours, hierarchy = cv.findContours(thresh, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
print(len(contours))

intercontours=[]
for i in range(0,len(contours)):
    if hierarchy[0][i][3] != -1:
        intercontours.append(contours[i])
cv.drawContours(im,intercontours,-1,(0,1,233),2)
    


contours0, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
#dst1=cv.drawContours(im,contours0, -1, (244,25,255), 2)
cv.imshow('Display',im)
cv.waitKey()
 