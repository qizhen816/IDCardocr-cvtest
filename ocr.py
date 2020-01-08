import time
import re
import numpy as np
import cv2
import threading
# import pytesseract

FLANN_INDEX_KDTREE = 0
MIN_MATCH_COUNT = 10
MASK_IMG_NAME = 'img/idcard_mask.jpg'
x = 1280.00 / 3840.00
pixel_x = int(x * 3840)


class FindAddress(threading.Thread):
    def __init__(self, crop_gray):
        threading.Thread.__init__(self)
        self.crop_gray = crop_gray
        self.coods = None

    def run(self):
        template = cv2.UMat(cv2.imread('img/address_mask_%s.jpg' % pixel_x, 0))
        w, h = cv2.UMat.get(template).shape[::-1]
        res = cv2.matchTemplate(self.crop_gray, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        top_left = (max_loc[0] + w, max_loc[1] - int(20 * x))
        bottom_right = (top_left[0] + int(1700 * x), top_left[1] + int(650 * x))
        # result = cv2.UMat.get(self.crop_org)[top_left[1] - 10:bottom_right[1], top_left[0] - 10:bottom_right[0]]
        # cv2.rectangle(self.crop_gray, top_left, bottom_right, 255, 2)
        self.coods = top_left, bottom_right

    def result(self):
        return self.coods

class FindBirthday(threading.Thread):
    def __init__(self, crop_gray):
        threading.Thread.__init__(self)
        self.crop_gray = crop_gray
        self.coods = None

    def run(self):
        template = cv2.UMat(cv2.imread('img/name_mask_%s.jpg' % pixel_x, 0))
        w, h = cv2.UMat.get(template).shape[::-1]
        res = cv2.matchTemplate(self.crop_gray, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        top_left = (max_loc[0] + w, max_loc[1] - int(20 * x))
        bottom_right = (top_left[0] + int(700 * x), top_left[1] + int(300 * x))

        # result = cv2.UMat.get(self.crop_org)[top_left[1] - 10:bottom_right[1], top_left[0] - 10:bottom_right[0]]
        # cv2.rectangle(self.crop_gray, top_left, bottom_right, 255, 2)
        self.coods = top_left, bottom_right

    def result(self):
        return self.coods

class FindName(threading.Thread):
    def __init__(self, crop_gray):
        threading.Thread.__init__(self)
        self.crop_gray = crop_gray
        self.coods = None

    def run(self):
        template = cv2.UMat(cv2.imread('img/name_mask_%s.jpg' % pixel_x, 0))
        w, h = cv2.UMat.get(template).shape[::-1]
        res = cv2.matchTemplate(self.crop_gray, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        top_left = (max_loc[0] + w, max_loc[1] - int(20 * x))
        bottom_right = (top_left[0] + int(700 * x), top_left[1] + int(300 * x))

        # result = cv2.UMat.get(self.crop_org)[top_left[1] - 10:bottom_right[1], top_left[0] - 10:bottom_right[0]]
        # cv2.rectangle(self.crop_gray, top_left, bottom_right, 255, 2)
        self.coods = top_left, bottom_right

    def result(self):
        return self.coods

class FindIdNumber(threading.Thread):
    def __init__(self, crop_gray):
        threading.Thread.__init__(self)
        self.crop_gray = crop_gray
        self.coods = None

    def run(self):
        template = cv2.UMat(cv2.imread('img/id_number_mask_%s.jpg' % pixel_x, 0))
        w, h = cv2.UMat.get(template).shape[::-1]
        res = cv2.matchTemplate(self.crop_gray, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        top_left = (max_loc[0] + w, max_loc[1] - int(20 * x))
        bottom_right = (top_left[0] + int(2300 * x), top_left[1] + int(300 * x))
        # result = cv2.UMat.get(self.crop_org)[top_left[1] - 10:bottom_right[1], top_left[0] - 10:bottom_right[0]]
        # cv2.rectangle(self.crop_gray, top_left, bottom_right, 255, 2)
        self.coods = top_left, bottom_right

    def result(self):
        return self.coods


class FindSex(threading.Thread):
    def __init__(self, crop_gray):
        threading.Thread.__init__(self)
        self.crop_gray = crop_gray
        self.coods = None

    def run(self):
        template = cv2.UMat(cv2.imread('img/sex_mask_%s.jpg' % pixel_x, 0))
        w, h = cv2.UMat.get(template).shape[::-1]
        res = cv2.matchTemplate(self.crop_gray, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        top_left = (max_loc[0] + w, max_loc[1] - int(20 * x))
        bottom_right = (top_left[0] + int(300 * x), top_left[1] + int(300 * x))
        # result = cv2.UMat.get(self.crop_org)[top_left[1] - 10:bottom_right[1], top_left[0] - 10:bottom_right[0]]
        # cv2.rectangle(self.crop_gray, top_left, bottom_right, 255, 2)
        self.coods = top_left, bottom_right

    def result(self):
        return self.coods


class IDCard(object):
    def __init__(self, parse_img_name):
        self.parse_img_name = parse_img_name

    def get_result(self):
        id_card = self.find_idcard()
        if id_card == None:
            return None
        gray_img, org_img = get_gray_img(id_card)

        find_name = FindName(gray_img)
        find_id_number = FindIdNumber(gray_img)
        find_sex = FindSex(gray_img)
        find_address = FindAddress(gray_img)
        # find_birthday = FindBirthday(gray_img)

        # 多线程
        data = [find_name, find_id_number, find_sex, find_address]
        for thread in data:
            thread.start()

        for thread in data:
            thread.join()

        id_card = {
            'name': find_name.result(),
            'id_number': find_id_number.result(),
            'address': find_address.result(),
            'sex': find_sex.result(),
            # 'birthday': find_birthday.result()
        }
        print(id_card)

        for k in id_card.keys():
            org_img = self.draw_contors(id_card[k],org_img)
        cv2.imshow('r',img_resize(org_img,600))
        cv2.waitKey(200)


    def draw_contors(self, resu, img):
        if resu != None:
            top_left,bottom_right = resu
            img = cv2.rectangle(img, top_left, bottom_right, 255, 2)
        return img

    def find_idcard(self):

        mask_img = cv2.imdecode(np.fromfile(MASK_IMG_NAME, dtype=np.uint8), -1)
        mask_img = img_resize(cv2.UMat(cv2.cvtColor(mask_img, cv2.COLOR_RGB2GRAY)), 640) #640

        parse_img_read = cv2.imdecode(np.fromfile(self.parse_img_name, dtype=np.uint8), -1)
        if parse_img_read.shape[0] > parse_img_read.shape[1]:
            parse_img_read = np.rot90(parse_img_read,k=1)
            m1 = img_resize(cv2.UMat(parse_img_read),640)
            f1 = FindName(get_gray_img(m1)[0])
            f1.start()
            f1.join()
            re = f1.result()
            if re!=None:
                c,_ = re
                if c[0]>320:
                    parse_img_read = np.rot90(parse_img_read, k=2)
            else:
                parse_img_read = np.rot90(parse_img_read, k=2)
        parse_img = img_resize(cv2.UMat(cv2.cvtColor(parse_img_read, cv2.COLOR_RGB2GRAY)), 1920) #1920
        img_org = img_resize(cv2.UMat(parse_img_read), 1920) #1920

        sift = cv2.xfeatures2d.SIFT_create()
        # 特征点检测
        kp1, des1 = sift.detectAndCompute(mask_img, None)
        kp2, des2 = sift.detectAndCompute(parse_img, None)

        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=10)

        matches = cv2.FlannBasedMatcher(index_params, search_params).knnMatch(des1, des2, k=2)

        good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
        print(len(good_matches))
        if len(good_matches) < MIN_MATCH_COUNT:
            cv2.imshow('f',img_resize(img_org,600))
            cv2.waitKey(200)
            return None

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        m, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        h, w = cv2.UMat.get(mask_img).shape
        m_r = np.linalg.inv(m)
        result_img = cv2.warpPerspective(img_org, m_r, (w, h))

        # 找到两幅图片特征点
        # draw_params = dict(matchColor=(0, 255, 0),
        #                    singlePointColor=None,
        #                    matchesMask=mask.ravel().tolist(),
        #                    flags=2)
        # img3 = cv2.drawMatches(mask_img, kp1, parse_img, kp2, good_matches, None, **draw_params)
        # cv2.imshow('2s',img3)
        # cv2.waitKey(12)

        return result_img


def img_resize(imggray, dwidth):
    crop = imggray
    size = crop.get().shape
    height = size[0]
    width = size[1]
    height = height * dwidth / width
    crop = cv2.resize(src=crop, dsize=(dwidth, int(height)), interpolation=cv2.INTER_CUBIC)

    return crop


def get_gray_img(org_img):
    # 图片尺寸 (410, 640, 3)
    height, width, color = cv2.UMat.get(org_img).shape
    height = int(height * 3840 * x / width)

    # 拉伸原图尺寸与模板一致
    org_img = cv2.resize(src=org_img, dsize=(int(3840 * x), height), interpolation=cv2.INTER_CUBIC)
    _gray_img = cv2.cvtColor(org_img, cv2.COLOR_RGB2GRAY)
    _gray_img = cv2.GaussianBlur(_gray_img, (3, 3), 0, 0, cv2.BORDER_DEFAULT)
    return _gray_img, org_img
