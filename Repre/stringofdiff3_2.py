# same with stringofdiff3, but because test set has too many files so have to employ looping (responsible for LCF15)
# 18th August 2022: still need to cut done video files into whatever folder
import cv2
import numpy as np
def clahe_me(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(v)
    # return cl
    merged = cv2.merge((h,s,cl))
    final = cv2.cvtColor(merged, cv2.COLOR_HSV2BGR)
    return final
def LABfi_me(frames, framepos, lof):
    #  labfyme test
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

# manual settings
lof = 3   #length of frame
testortrain =  "training" #"test" # "train"


if testortrain == "test": inrange = range(220, 293)
elif testortrain ==  "training": inrange = range(200, 220)
else: raise Exception("train or test? Val is a subset of original LCF15 dataset")

for gtno in inrange:
    progressbar = 0
    print("gtno")
    print(gtno)
    video = cv2.VideoCapture("D:/f4kBIG/fishclef_2015/{testortrain}_set/videos/gt_{gtno}.flv".format(gtno = gtno, testortrain = testortrain))
    framewidth = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameheight = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('NoCLAHEgt{gtno}_f{lof}_v2.avi'.format(gtno = gtno, lof = lof ),fourcc, 30.0, (framewidth,frameheight))

    arrayoframes = []
    fordiff = []
    ok = True
    while ok == True:
        ok, frame = video.read()

        # initialize the first 2 frames
        if len(fordiff)<2:
            print("was here")
            fordiff.append(frame)
            continue

        # initialize the first 5 diff frames
        if len(arrayoframes) < lof:
            try:
                framediff = cv2.subtract(fordiff[1], fordiff[0])
            except:
                print("should be the end of file")
                continue
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
            # print(len(arroLAB))
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
            print("progress bar")
            progressbar+=1
            print(progressbar)


            # display


            cv2.waitKey()


            arrayoframes.pop(0)
            # break
        # update fordiff
        fordiff.pop(0)
        fordiff.append(frame)


    out.release()
    cv2.destroyAllWindows()
