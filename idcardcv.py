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

def locateText(line,cnt):
    hist_line = np.sum(line, axis=0)
    hist_line_idx = np.nonzero(hist_line > 5)[0]
    if len(hist_line_idx) == 0:
        return 0, len(hist_line)
    if cnt in [0,1,2,-1]:
        return hist_line_idx[0],hist_line_idx[-1]
    last = hist_line_idx[0]
    for idx in hist_line_idx:
        if idx - last > 30 or idx == hist_line_idx[-1]:
            break
        last = idx
    return hist_line_idx[0],last

def findAttrs(fname):
    img = cv2.imdecode(np.fromfile(fname, dtype=np.uint8), -1)
    h, w, _ = img.shape
    if h > w:
        img = np.rot90(img)

    img = cv2.resize(img,(1000,600))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gray = np.squeeze(img[:,:,0])
    # gray = cv2.equalizeHist(gray)
    # gray[gray>np.mean(gray)//2] = 255
    # binary = ~gray
    ret, binary = cv2.threshold(gray, np.mean(gray)*0.65, 255, cv2.THRESH_BINARY)

    element = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    binary = cv2.erode(~binary, element)
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    binary = cv2.dilate(binary, element)

    binary = cv2.medianBlur(binary,ksize=3)
    # connectivity = 4
    # num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity, cv2.CV_8U)
    # labels = np.array(labels, dtype=np.int8)
    # maxnum = [(i, stats[i][-1]) for i in range(len(stats))]
    # maxnum = sorted(maxnum, key=lambda s: s[1], reverse=True)
    # print(maxnum)
    # for i in range(3):
    #     x,y = maxnum[i]
    #     if y>100:
    #         binary[labels == x] = 0
    # x,y = maxnum[1]
    # binary[labels == x] = 0

    cv2.imshow('h',binary)

    # element = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    # binary = cv2.dilate(binary, element)

    binary[binary != 0] = 1
    h, w = binary.shape
    horizontal = np.sum(binary, axis=0)
    if np.sum(horizontal[:w//2])>np.sum(horizontal[w//2:]):
        binary = np.rot90(binary,k=2)
        img = np.rot90(img,k=2)
        horizontal = horizontal[::-1]
    horizontal_idx = np.nonzero(horizontal <= max(w/8,np.min(horizontal)*2))[0] #np.min(horizontal)*3
    vertical = np.sum(binary, axis=1)
    vertical_idx = np.nonzero(vertical <= max(h/8,np.min(vertical)*2))[0] #np.min(vertical)*3
    binary_half = binary[vertical_idx[0]:vertical_idx[-1],horizontal_idx[0]:int(w*0.6)]
    cv2.imshow('333',binary_half*255)
    # cv2.imshow('331',binary[vertical_idx[0]:vertical_idx[-1],horizontal_idx[0]:horizontal_idx[-1]]*255)

    hist = np.sum(binary_half, axis=1)
    text_idx = np.nonzero(hist>10)[0]
    last = text_idx[0]
    begin = text_idx[0]
    coods = []
    count = 0
    mask = np.zeros_like(img)
    for idx in text_idx:
        if idx - last > 5 or idx == text_idx[-1]:
            coods.append([begin, last])
            if count < 3:
                line = binary[begin+vertical_idx[0]:last+vertical_idx[0],horizontal_idx[0]:int(w*0.6)]
            else:
                line = binary[begin+vertical_idx[0]:last+vertical_idx[0],horizontal_idx[0]:horizontal_idx[-1]]
            if idx == text_idx[-1]:
                count = -1
            h_begin,h_last = locateText(line,count)
            x1 = horizontal_idx[0]+h_begin
            y1 = begin+vertical_idx[0]
            x2 = horizontal_idx[0]+h_last
            y2 = last+vertical_idx[0]
            mask[y1:y2,x1:x2] = img[y1:y2,x1:x2]
            begin = idx
            count += 1
        last = idx

    cv2.imshow('33',mask)

    # cv2.imshow('33',binary_half*255)
    cv2.waitKey(2000)

imgs = glob.glob('E:/dataset/身份证分割/*.jpg')
for fname in imgs[2:]:
    findAttrs(fname)