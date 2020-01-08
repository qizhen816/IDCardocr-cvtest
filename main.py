import datetime
import time

from ocr import IDCard
import glob

if __name__ == "__main__":
    # imgs = glob.glob('E:/dataset/身份证分割/val_images/*.jpg')
    imgs = glob.glob('test_img/*.jpg')

    for img in imgs[7:]:
        print(img)
        starttime = time.time()
        ocr = IDCard(img)
        ocr.get_result()
        endtime = time.time()
        print("运行时间:", (endtime - starttime))
