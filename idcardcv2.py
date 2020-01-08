# -*- coding: utf-8 -*-
"""
 @Time       : 2019/12/25 17:53
 @Author     : Zhen Qi
 @Email      : qizhen816@163.com
 @File       : idcardcv.py
 @Description: API Script
"""
import cv2
import glob
import numpy as np



def findAttrs(fname):
    img = cv2.imdecode(np.fromfile(fname, dtype=np.uint8), -1)
    h, w, _ = img.shape
    if h > w:
        img = np.rot90(img)

    img = cv2.resize(img,(1200,800))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 高斯平滑
    Gaussian = cv2.GaussianBlur(gray, (3, 3), 0, 0, cv2.BORDER_DEFAULT)
    # 中值滤波
    gray = cv2.medianBlur(Gaussian, 5)
    gray = cv2.equalizeHist(gray)
    # gray = np.squeeze(img[:,:,2])
    # binary = ~gray
    # gray = ~gray
    gray = cv2.filter2D(gray, -1,
                               kernel=np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32))  # 对图像进行滤波,是锐化操作
    # gray = cv2.filter2D(gray, -1, kernel=np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32))
    gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=3)
    gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=3)
    img_gradient = cv2.subtract(gradX, gradY)
    Sobel = cv2.convertScaleAbs(img_gradient)  # sobel算子,计算梯度, 也可以用canny算子替代
    # absX = cv2.convertScaleAbs(gradX)  # 转回uint8
    # absY = cv2.convertScaleAbs(gradY)
    # Sobel = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    blurred = cv2.GaussianBlur(Sobel, (5, 5), 0)  # 再进行一次高斯去噪
    # blurred = cv2.medianBlur(Sobel, 5)
    # 这里改进成自适应阈值,貌似没用
    # ret, binary = cv2.threshold(blurred, 170, 255, cv2.THRESH_BINARY)
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, -3)
    # binary = cv2.medianBlur(binary,ksize=3)
    # ret, binary = cv2.threshold(gray, np.mean(gray)*0.65, 255, cv2.THRESH_BINARY)


    # # 膨胀和腐蚀操作的核函数
    # element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    # element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    # Dilation = cv2.dilate(binary, element2, iterations=1)
    # Erosion = cv2.erode(Dilation, element1, iterations=1)
    # img_closed = cv2.dilate(Erosion, element2, iterations=2)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    img_closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    img_closed = cv2.morphologyEx(img_closed, cv2.MORPH_OPEN, kernel)
    img_closed1 = cv2.erode(img_closed, None, iterations=1)
    img_closed1 = cv2.dilate(img_closed1, None, iterations=3)
    # 这里调整了kernel大小(减小),腐蚀膨胀次数后(增大),出错的概率大幅减小

    cv2.imshow('333',blurred)
    # cv2.imshow('331',binary[vertical_idx[0]:vertical_idx[-1],horizontal_idx[0]:horizontal_idx[-1]]*255)


    cv2.waitKey(2000)

imgs = glob.glob('E:/dataset/身份证分割/val_images/*.jpg')
# fname = 'E:/dataset/身份证/val_images/236107_front.jpg'
for fname in imgs[2:]:
    findAttrs(fname)