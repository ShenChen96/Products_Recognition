import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

def homo_diff(pts, homograph_1, homograph_2):
    src_pts = np.float32(pts).reshape(-1, 1, 2)

    src_pts_1 = cv2.perspectiveTransform(src_pts, homograph_1)
    src_pts_2 = cv2.perspectiveTransform(src_pts, homograph_2)

    ttr = np.squeeze(src_pts_1)
    ggt = np.squeeze(src_pts_2)
    diff = ttr - ggt

    total_cdist = 0

    for each_point in diff:
        x = each_point[0]
        y = each_point[1]
        dist = x**2 + y **2
        total_cdist += dist ** 0.5

     diff_value =  total_cdist / len(diff)
	
    return diff_value 

def get_img_interest_point(img_path):
    img = cv2.imread(img_path)
    h, w, _ = img.shape

    # print("the h is : {}".format(str(h)))
    max_resolution = 960
    if (h > max_resolution):
        scale_factor = float(max_resolution) / float(h)
    else:
        scale_factor = 1.0
    img = cv2.resize(img, (int(w * scale_factor), int(h * scale_factor)),
                     interpolation=cv2.INTER_CUBIC)
    # img = cv2.resize(img,(1080,1920))
    # h,w,_ = img.shape
    # img[1920//3*2:,1080//3*2:,:] = 0
    # descriptor = cv2.KAZE_create()
    descriptor = cv2.AKAZE_create()
    # masks = np.zeros((h, w, 1), np.uint8)
    # masks[:h//3*2,:w//3*2]=255
    # kps = descriptor.detect(img,mask=masks)
    kps = descriptor.detect(img)
    vector_size = 2000
    kps = sorted(kps, key=lambda x: -x.response)[:vector_size]
    # kps, des = descriptor.detectAndCompute(img, kps)
    kps, des = descriptor.compute(img, kps)

    return kps, des

class Point(object):
    def __init__(self, x=0.0, y=0.0, z=1.0):
        self.x = x
        self.y = y
        self.z = z

    def calculatenewpoint(self, homo):
        point_old = np.array([self.x, self.y, self.z]).reshape(3, 1)
        point_new = np.dot(homo, point_old)
        point_new /= point_new[2, 0]
        self.x = point_new[0, 0]
        self.y = point_new[1, 0]
        self.z = point_new[2, 0]


class Corner(object):
    def __init__(self):
        self.ltop = Point()
        self.lbottom = Point()
        self.rtop = Point()
        self.rbottom = Point()

    def calculatefromimage(self, img):
        rows = img.shape[0]
        cols = img.shape[1]
        self.ltop.x = 0.0
        self.ltop.y = 0.0
        self.lbottom.x = 0.0
        self.lbottom.y = float(rows)
        self.rtop.x = float(cols)
        self.rtop.y = 0.0
        self.rbottom.x = float(cols)
        self.rbottom.y = float(rows)

    def calculatefromhomo(self, homo):
        self.ltop.calculatenewpoint(homo)
        self.lbottom.calculatenewpoint(homo)
        self.rtop.calculatenewpoint(homo)
        self.rbottom.calculatenewpoint(homo)

    def getoutsize(self):
        lx = min(self.ltop.x, self.lbottom.x)
        rx = max(self.rtop.x, self.rbottom.x)
        uy = min(self.ltop.y, self.rtop.y)
        dy = max(self.lbottom.y, self.rbottom.y)
        return lx, rx, uy, dy


def calculatecorners(imgs, homos):
    c = Corner()
    c.calculatefromimage(imgs)
    c.calculatefromhomo(homos)

    return c


