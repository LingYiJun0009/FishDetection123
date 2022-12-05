# 3rd November 2021
# same with stringofdiff2
# added video writer (saved images is the unresized one)
# cancelled waitKey since I don't want to press the button for every frame
# also added a bunch of other file path

# File path history:
# path 1: BB
#   input: "../24thReport2021/H_a_waterfallfishyTrim.mp4"
#   output: 'BBTrim_Diff.avi'   [OK]
# path 2: FF
#   input: "./FF_Trim.mp4"
#   output: 'FFTrim_Diff.avi' [OK]
#path 3: snorkel
#   input: "./snorkelTrim.MOV"
#   output: 'snorkelTrim_Diff.avi' [OK]
#path 4: pedestrian
#   input: "./crosswalk.avi"
#   output: 'crosswalk_Diff.avi'    [OK]
#path 5:
#   input: "./f4k_Trim.mp4"
#   output: "f4k_Trim_Diff.avi"     [OK]
#path 6: vehicles
#   input: "./cars.mp4"
#   output: "./cars_diff.avi"

# path 7: F4K gt_106.flv
# input: D:/f4k_detection_tracking/gt_106.flv
# output: './NoCLAHEgt106.avi'

# path 8: F4K gt_107.flv
# input: D:/f4k_detection_tracking/gt_107.flv
# output: './NoCLAHEgt107.avi'

# path 9: F4K gt_109.flv
# input: D:/f4k_detection_tracking/gt_109.flv
# output: './NoCLAHEgt109.avi'

# path 10: F4K gt_110.flv
# input: D:/f4k_detection_tracking/gt_110.flv
# output: './NoCLAHEgt110.avi'

# path 11: F4K gt_111.flv
# input: D:/f4k_detection_tracking/gt_111.flv
# output: './NoCLAHEgt111.avi'

# path 12: F4K gt_112.flv
# input: D:/f4k_detection_tracking/gt_112.flv
# output: './NoCLAHEgt112.avi'

# path 13: F4K gt_113.flv
# input: D:/f4k_detection_tracking/gt_113.flv
# output: './NoCLAHEgt113.avi'

# path 14: F4K gt_114.flv
# input: D:/f4k_detection_tracking/gt_114.flv
# output: './NoCLAHEgt114.avi'

# path 15: F4K gt_116.flv
# input: D:/f4k_detection_tracking/gt_116.flv
# output: './NoCLAHEgt116.avi'

# path 16: F4K gt_117.flv
# input: D:/f4k_detection_tracking/gt_117.flv
# output: './NoCLAHEgt117.avi'

# path 17: F4K gt_118.flv
# input: D:/f4k_detection_tracking/gt_118.flv
# output: './NoCLAHEgt118.avi'

# path 18: F4K gt_119.flv
# input: D:/f4k_detection_tracking/gt_119.flv
# output: './NoCLAHEgt119.avi'

# path 19: LCF gt_200.flv       # 15th May 2022
# input: D:/f4kBIG/fishclef_2015/training_set/videos/gt_200.flv
# output: './NoCLAHEgt200.avi'

# path 19: LCF gt_201.flv       # 15th May 2022
# input: D:/f4kBIG/fishclef_2015/training_set/videos/gt_201.flv
# output: './NoCLAHEgt201.avi'

# path 20: LCF gt_202.flv       # 15th May 2022
# input: D:/f4kBIG/fishclef_2015/training_set/videos/gt_202.flv
# output: './NoCLAHEgt202.avi'

# path 21: LCF gt_203.flv       # 15th May 2022
# input: D:/f4kBIG/fishclef_2015/training_set/videos/gt_203.flv
# output: './NoCLAHEgt203.avi'

# path 22: LCF gt_204.flv       # 15th May 2022
# input: D:/f4kBIG/fishclef_2015/training_set/videos/gt_204.flv
# output: './NoCLAHEgt204.avi'

# path 23: LCF gt_205.flv       # 15th May 2022
# input: D:/f4kBIG/fishclef_2015/training_set/videos/gt_205.flv
# output: './NoCLAHEgt205.avi'

# path 24: LCF gt_206.flv       # 15th May 2022
# input: D:/f4kBIG/fishclef_2015/training_set/videos/gt_206.flv
# output: './NoCLAHEgt206.avi'

# path 25: LCF gt_207.flv       # 15th May 2022
# input: D:/f4kBIG/fishclef_2015/training_set/videos/gt_207.flv
# output: './NoCLAHEgt207.avi'

# path 26: LCF gt_208.flv       # 15th May 2022
# input: D:/f4kBIG/fishclef_2015/training_set/videos/gt_208.flv
# output: './NoCLAHEgt208.avi'

# path 27: LCF gt_209.flv       # 15th May 2022
# input: D:/f4kBIG/fishclef_2015/training_set/videos/gt_209.flv
# output: './NoCLAHEgt209.avi'

# path 28: LCF gt_210.flv       # 15th May 2022
# input: D:/f4kBIG/fishclef_2015/training_set/videos/gt_210.flv
# output: './NoCLAHEgt210.avi'

# path 29: LCF gt_211.flv       # 15th May 2022
# input: D:/f4kBIG/fishclef_2015/training_set/videos/gt_211.flv
# output: './NoCLAHEgt211.avi'

# path 30: LCF gt_212.flv       # 15th May 2022
# input: D:/f4kBIG/fishclef_2015/training_set/videos/gt_212.flv
# output: './NoCLAHEgt212.avi'

# path 31: LCF gt_213.flv       # 15th May 2022
# input: D:/f4kBIG/fishclef_2015/training_set/videos/gt_213.flv
# output: './NoCLAHEgt213.avi'

# path 32: LCF gt_214.flv       # 15th May 2022
# input: D:/f4kBIG/fishclef_2015/training_set/videos/gt_214.flv
# output: './NoCLAHEgt214.avi'

# path 33: LCF gt_215.flv       # 15th May 2022
# input: D:/f4kBIG/fishclef_2015/training_set/videos/gt_215.flv
# output: './NoCLAHEgt215.avi'

# path 34: LCF gt_216.flv       # 15th May 2022
# input: D:/f4kBIG/fishclef_2015/training_set/videos/gt_216.flv
# output: './NoCLAHEgt216.avi'

# path 35: LCF gt_217.flv       # 15th May 2022
# input: D:/f4kBIG/fishclef_2015/training_set/videos/gt_217.flv
# output: './NoCLAHEgt217.avi'

# path 36: LCF gt_218.flv       # 15th May 2022
# input: D:/f4kBIG/fishclef_2015/training_set/videos/gt_218.flv
# output: './NoCLAHEgt218.avi'

# path 37: LCF gt_219.flv       # 15th May 2022
# input: D:/f4kBIG/fishclef_2015/training_set/videos/gt_219.flv
# output: './NoCLAHEgt219.avi'

import cv2
import numpy as np

lof = 5   #length of frame
video = cv2.VideoCapture("D:/f4kBIG/fishclef_2015/training_set/videos/gt_219.flv")

framewidth = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frameheight = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('NoCLAHEgt219_5f_v2.avi',fourcc, 30.0, (framewidth,frameheight))
def clahe_me(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(v)
    # return cl
    merged = cv2.merge((h,s,cl))
    final = cv2.cvtColor(merged, cv2.COLOR_HSV2BGR)
    return final

# def LABfi_me(frames, framepos, lof):
#     lab = cv2.cvtColor(frames, cv2.COLOR_BGR2LAB)
#     l,a,b = cv2.split(lab)
#     # b = b*(255/2)
#
#     a = a*((255/lof)*framepos)
#     a = a.astype('uint8')
#     b = b.astype('uint8')
#     merged = cv2.merge((l,a,b))
#     final = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
#     return final
#     #img1=img1.astype('uint8')

#  labfyme test

def LABfi_me(frames, framepos, lof):
    hsv = cv2.cvtColor(frames, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    h = h * 0
    h = h + ((255/lof) * framepos)
    s = s*0
    s = s+255
    # l=l*255
    # a = a * ((255 / lof) * framepos)

    h = h.astype('uint8')
    s = s.astype('uint8')
    v = v.astype('uint8')
    merged = cv2.merge((h, s, v))
    final = cv2.cvtColor(merged, cv2.COLOR_HSV2BGR)
    return final
    # img1=img1.astype('uint8')

def resizemeh(frame):
    rsframe = cv2.resize(frame, (int(frame.shape[1] / 2), int(frame.shape[0] / 2)), interpolation=cv2.INTER_AREA)
    return rsframe

arrayoframes = []
fordiff = []

while True:
    ok, frame = video.read()

    # initialize the first 2 frames
    if len(fordiff)<2:
        print("was here")
        fordiff.append(frame)
        continue

    # initialize the first 5 diff frames
    if len(arrayoframes) < lof:
        framediff = cv2.subtract(fordiff[1], fordiff[0])
        #framediff = clahe_me(framediff)         # to clahe here
        arrayoframes.append(framediff)
        if len(arrayoframes) < lof:
            fordiff.pop(0)
            fordiff.append(frame)
            continue


    #  convert into LAB format and combine 5(lof) frames to display
    if len(arrayoframes) == lof:

        arroLAB = []
        for i in range(lof):
            # convert to LAB
            labby = LABfi_me(arrayoframes[i], i, lof)
            arroLAB.append(labby)

            # display test
            # res = resizemeh(labby)
            # cv2.imshow('labby', res)
            # res2 = resizemeh(arrayoframes[i])
            # cv2.imshow('claheddiff', res2)
            # cv2.waitKey()

        sum = arroLAB[0]
        print(len(arroLAB))
        for i in range(1,lof):
            # sum the converted array of frames
            sum = cv2.add(sum, arroLAB[i])

        towrite = sum
        #sumre = resizemeh(sum)
        # cv2.imshow('sum1', arroLAB[0])
        # cv2.imshow('sum2', arroLAB[1])
        # cv2.imshow('sum3', arroLAB[2])
        # cv2.imshow('sum4', arroLAB[3])
        # cv2.imshow('sum5', arroLAB[4])
        #
        # cv2.imshow('sum', towrite)
        # print("boop")

        #write here
        out.write(towrite)

        # display


        cv2.waitKey()


        arrayoframes.pop(0)
        # break
    # update fordiff
    fordiff.pop(0)
    fordiff.append(frame)


out.release()
cv2.destroyAllWindows()




