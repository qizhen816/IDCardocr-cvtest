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


def match_attr(tamp,img):
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(tamp, None)  # 找到模板图像的特征点
    kp2, des2 = sift.detectAndCompute(img, None)  # 找到识别的特征点

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=10)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # store all the good matches as per Lowe's ratio test.
    # 两个最佳匹配之间距离需要大于ratio 0.7,距离过于相似可能是噪声点
    good = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good.append(m)
    # reshape为(x,y)数组
    if len(good) > 2:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        # 用HomoGraphy计算图像与图像之间映射关系, M为转换矩阵
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
        # 使用转换矩阵M计算出img1在img2的对应形状
        h, w = cv2.UMat.get(tamp).shape
        M_r = np.linalg.inv(M)
        im_r = cv2.warpPerspective(img, M_r, (w, h))


def findAttrs(fname,tamp):
    img = cv2.imdecode(np.fromfile(fname, dtype=np.uint8), -1)
    tamp = cv2.imdecode(np.fromfile(tamp, dtype=np.uint8), -1)
    match_attr(img,tamp)
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

    gray = cv2.filter2D(gray, -1,
                               kernel=np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32))  # 对图像进行滤波,是锐化操作
    # gray = cv2.filter2D(gray, -1, kernel=np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32))
    gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=3)
    gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=3)
    img_gradient = cv2.subtract(gradX, gradY)
    Sobel = cv2.convertScaleAbs(img_gradient)  # sobel算子,计算梯度, 也可以用canny算子替代
    blurred = cv2.GaussianBlur(Sobel, (5, 5), 0)  # 再进行一次高斯去噪
    # 这里改进成自适应阈值,貌似没用
    # ret, binary = cv2.threshold(blurred, 170, 255, cv2.THRESH_BINARY)
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, -3)


    cv2.imshow('333',blurred)
    # cv2.imshow('331',binary[vertical_idx[0]:vertical_idx[-1],horizontal_idx[0]:horizontal_idx[-1]]*255)


    cv2.waitKey(2000)

imgs = glob.glob('E:/dataset/身份证分割/val_images/*.jpg')
tmp = 'imgs/id_tamp1.png'
# fname = 'E:/dataset/身份证/val_images/236107_front.jpg'
for fname in imgs[:]:
    findAttrs(fname,tmp)