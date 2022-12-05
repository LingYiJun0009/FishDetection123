import cv2
import numpy as np

# read image
img = cv2.imread('scene00001.jpg')

# read from text file
with open('scene00001.txt') as f:
    lines = f.readlines()

# return array of bbox coordinates from txt file
def convert(lines):
    ilines = []
    bbox = []
    for i in range(len(lines)):
        ilines.append(list(map(float,lines[i].split( ))))
        plc = ilines[-1]

        # convert to bbox format
        bboxcur = list(map(int,
                       [plc[1]*img.shape[1], plc[2]*img.shape[0], plc[3]*img.shape[1], plc[4]*img.shape[0]]))
        bbox.append(bboxcur)

    return bbox

# draw array of bbox to the image file
def drawbbx(img, bboxarr):
    for i in bboxarr:
        p1 = (i[0], i[1])
        p2 = (i[0] + i[2], i[1] + i[3])
        cv2.rectangle(img, p1, p2, (255, 0, 0), 2, 1)

    # img = cv2.resize(img, (int(img.shape[1] / 2), int(img.shape[0] / 2)), interpolation=cv2.INTER_AREA)
    # cv2.imshow('res', img)
    # cv2.waitKey()
    return img


# array of bbox coordinates per img
bboxarr = convert(lines)
cv2.imshow('fwe', drawbbx(img, bboxarr))
cv2.waitKey()
f.close()