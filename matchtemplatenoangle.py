import cv2
import numpy as np
from matplotlib import pyplot as plt

def rotation(img,M):
    rows, cols , depth= img.shape
    rotate = cv2.getRotationMatrix2D((cols / 2, rows / 2), M, 1)
    dst = cv2.warpAffine(img,rotate,(cols,rows))
    return dst

img_rgb = cv2.imread('sperm1.jpg')
# img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
template_1 = cv2.imread('spermsample4.jpg')
img_B, img_G, img_R = cv2.split(img_rgb)
for i in range(1,36):
    temp=rotation(template_1,10*(i-1))
    template=temp[9:30,10:30]
    # print(template.shape)
    # cv2.imshow('template',template)
    # cv2.waitKey(0)


    template_B, template_G, template_R = cv2.split(template)

    w, h = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY).shape[::-1]
    res_B = cv2.matchTemplate(img_B, template_B, cv2.TM_CCOEFF_NORMED)
    res_G = cv2.matchTemplate(img_G, template_G, cv2.TM_CCOEFF_NORMED)
    res_R = cv2.matchTemplate(img_R, template_R, cv2.TM_CCOEFF_NORMED)
    threshold = 0.9
    res = (res_B + res_G + res_R*1.2)

    loc_B = np.where(res_B >= threshold)
    loc_G = np.where(res_G >= threshold)
    loc_R = np.where(res_R >= threshold)
    loc = np.where(res >= 3.2 * threshold)
    # loc=[loc_B,loc_G,loc_R]

    for pt in zip(*loc_B[::-1]):
        # cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
        cv2.rectangle(img_B, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
    for pt in zip(*loc_G[::-1]):
        # cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
        cv2.rectangle(img_G, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
    for pt in zip(*loc_R[::-1]):
        # cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
        cv2.rectangle(img_R, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
cv2.imwrite('B.png',img_B)
cv2.imwrite('G.png',img_G)
cv2.imwrite('R.png',img_R)
cv2.imwrite('rgb.png',img_rgb)
cv2.imshow('B.png',img_B)
cv2.imshow('G.png',img_G)
cv2.imshow('R.png',img_R)
cv2.imshow('rgb.png',img_rgb)
cv2.waitKey(0)



