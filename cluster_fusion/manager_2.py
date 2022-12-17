# remember to look for todo(s)

# in terms of the whole pipeline manager_2.py starts from after making inference using YOLOv5 RGB & Repre
# rough copy of manager.py [3rd December]
# new features:
    # 1) cleared up unecessary code, clusteroutput and manager code

import os
import time
import inspect
import shutil
from os import path
import numpy as np
from cluster_fusion.utils import *
import math
import cv2
from natsort import natsorted
# what if I change something
# get the latest written test run folder
# test if it did get pushed attempt2
pathtorunsdata_test = "../Data/Runs/Test/"
latesttest = natsorted(os.listdir(pathtorunsdata_test))[-1]
toggle = 2
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    MEHGREEN = '\033[32m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# the clusterParams isn't really cluster params since
class clusterParams:
    # ground truth image path "placeholder"
    gtpath = "D:/f4kBIG/fishclef_2015_withval_v3/labels/f4kBIG_test_v3"  # test
    imgpath = "D:/f4kBIG/fishclef_2015_withval_v3/images/f4kBIG_test_v3"  # test

    # after YOLOv5 RGB and Repre
    # variables that's dynamically set/ changeable
    # input output txt folders for clusteroutput module
    # todo: change these variables later to fit module to run CNN testing + manager_2 in one go
    cf_0001_path_folder = "../Data/Runs/Test/{}/RGB_labels/labels".format(latesttest)
    repre005_path_folder = "../Data/Runs/Test/{}/Repre_labels/labels".format(latesttest)

    # for after a_clusteroutput
    cloutput_nf = "../Data/Runs/Test/{}/cloutput".format(latesttest)

    # how many frame for repre
    repre_f = 5

    # a_clusteroutput params
    a_1 = 0.05  # remove too small repre boxes
    a_2 = 0.5  # remove too big repre boxes
    a_3 = 20  # distance threshold between 2 points in a cluster
    a_4 = 1000  # linalg dist between boxes within a cluster
    a_5 = 0.7  # get 80% most similar angle
    a_6 = 0.1  # std dev accepted for angle

    # automatictroubleshoot param
    bestMAPconf = 0.541 # this will depend on training data

    # todo: IOU_cutoff and conf_cutoff should be updated to read train0's future output file (have yet to be implemented)
    IOU_cutoff = 0.002
    conf_cutoff = 0.15

    # output folder after clusteroutput_fusion (so it can be sent to test mAP)
    finalresult = "../Data/Runs/Test/{}/finalresult".format(latesttest)


def a_clusteroutput(a_1, a_2, a_3, a_4, a_5, a_6, filenamepath, cloutput_nf, imgpath):
    """
    process YOLOv5_Repreoutput
    the output of this module will be further processed by automatictroubleshoot function

    :param a_1:
    :param a_2:
    :param a_3:
    :param a_4:
    :param a_5:
    :param a_6:
    :param filenamepath: clusterParams.repre005_path_folder   # read from this folder
    :param cloutput_nf: cloutput_nf = clusterParams.cloutput_nf # output to this folder
    :param imgpath:
    :param imgoutput:
    :return: return nothing
    """

    def rmvcluster(listocluster, boxmat):

        for loind, sg2cluster in enumerate(listocluster):

            # with or against concept, checking the first line of distancemat2 would suffice
            # rmv boxes that has exactly the same overlap
            relboxmat = boxmat[sg2cluster]
            relboxmat = relboxmat.clip(min=0)
            height = relboxmat[:, 2] - relboxmat[:, 0]
            width = relboxmat[:, 3] - relboxmat[:, 1]
            hwratio = [height / width]
            area = [height * width]
            distancemat2 = distancemat_2(hwratio, area)
            # print("distancemat2")
            # print(distancemat2)

            unique, counts = np.unique(distancemat2[0], return_counts=True)

            unique = unique[counts > 1]

            if len(unique) > 0:
                rmlist = []
                for iu in unique:
                    rmlist.append(np.where(distancemat2[0] == iu)[0][1:])

                # flat_list = [item for sublist in t for item in sublist]
                rmlist = [item for sublist in rmlist for item in sublist]
                rmlist = list(np.sort(np.array(rmlist)))
                todeduct = 0

                for ir, r in enumerate(rmlist):
                    r = int(r)
                    r = r - todeduct
                    sg2cluster.pop(r)
                    todeduct += 1

                listocluster.pop(loind)
                listocluster.insert(loind, sg2cluster)

        # print("listocluster")
        # print(listocluster)
        return listocluster
    def splitcluster2(listocluster, boxmat):
        # distance metric for both split cluster is made through a few sample cases
        toremove = []
        appendall = []

        for loind, sg2cluster in enumerate(listocluster):

            # with or against concept, checking the first line of distancemat2 would suffice
            relboxmat = boxmat[sg2cluster]
            relboxmat = relboxmat.clip(min=0)
            height = relboxmat[:, 2] - relboxmat[:, 0]
            width = relboxmat[:, 3] - relboxmat[:, 1]
            hwratio = [height / width]
            area = [height * width]
            distancemat2 = distancemat_2(hwratio, area)

            if np.max(distancemat2[:, :]) == 0:
                toremove.append(loind)

            if np.any(distancemat2[:, :] > a_4):

                stuff = np.where(distancemat2[0] > a_4)[0]
                stuff2 = np.where(distancemat2[0] < a_4)[0]
                if len(stuff) == len(distancemat2[0]):
                    toremove.append(loind)
                    continue

                if len(stuff) >= 3 or len(stuff2) >= 3:
                    # listocluster = listocluster[:loind-1] + listocluster[loind + 1:]

                    toremove.append(loind)
                    print("look here")
                    print(np.std(distancemat2[0, stuff]))
                    if len(stuff) > 3 and np.std(distancemat2[0, stuff]) < a_4:
                        tinyclust = []
                        for s in stuff:
                            tinyclust.append(sg2cluster[s])
                        appendall += [tinyclust]
                    if len(stuff2) > 3 and np.std(distancemat2[0, stuff2]) < a_4:
                        tinyclust = []
                        for s in stuff2:
                            tinyclust.append(sg2cluster[s])
                        appendall += [tinyclust]

                else:
                    toremove.append(loind)

            # remove the array where there are more than 1 0

        toremove = np.array(toremove)

        if len(toremove) > 0:
            for i in range(len(toremove)):
                listocluster = listocluster[:toremove[i]] + listocluster[toremove[i] + 1:]
                toremove = toremove - 1
        listocluster += appendall

        return listocluster
    def splitcluster(listocluster, boxmat, listocenter):
        toremove = []
        appendall = []
        for lind, cluster in enumerate(listocluster):
            relboxmat = boxmat[cluster]
            relboxmat = relboxmat.clip(min=0)
            # print("relboxmat")
            # print(relboxmat)

            allx = np.append(relboxmat[:, 0], relboxmat[:, 2])
            ally = np.append(relboxmat[:, 1], relboxmat[:, 3])

            maxx = np.max(allx)
            minx = np.min(allx)
            maxy = np.max(ally)
            miny = np.min(ally)

            roi = np.zeros((int(maxy - miny), int(maxx - minx)))
            # cv2.rectangle(img, (round(topx), round(topy)), (round(bottomx), round(bottomy)), (0, 0,255), 1)

            norm_relboxmat = relboxmat - np.array([minx, miny, minx, miny])

            for rec in norm_relboxmat:
                # cv2.rectangle(roi, (round(rec[0]), round(rec[1])), (round(rec[2]), round(rec[3])), (0), 1, cv2.FILLED)
                roi[int(rec[1]): int(rec[3]), int(rec[0]): int(rec[2])] = 255
            roi = np.uint8(roi)
            num_labels, labels_im = cv2.connectedComponents(roi)
            splitclust = [[] for j in range(num_labels)]
            clustercenter = listocenter[cluster] - np.array([minx, miny])  # assign every value in cluster

            if num_labels >= 3:
                for ic, cctr in enumerate(clustercenter):  # splitclust[].append(cluster[ic])
                    # print("cctr")
                    # print(cctr)
                    # print(labels_im.shape)
                    splitclust[labels_im[int(cctr[1]) - 1][int(cctr[0]) - 1]].append(cluster[ic])
                for sc in splitclust:
                    if len(sc) >= 3:
                        # listocluster = listocluster[:lind] + listocluster[lind + 1:] + [sc]
                        appendall += [sc]
                toremove.append(lind)
        toremove = np.array(toremove)

        if len(toremove) > 0:
            for i in range(len(toremove)):
                listocluster = listocluster[:toremove[i] - 1] + listocluster[toremove[i] + 1:]
                toremove = toremove - 1

        listocluster += appendall

        return listocluster
    def calccoords(len, centerx, centery, rad):  # calculate coordinates given radians 25th Feb 2022
        rad = (float(rad))
        # "unnormalize angle"  (tolossangle+math.pi)/(math.pi*2)
        rad = (rad * (math.pi * 2)) - math.pi
        x = centerx + (len * math.cos(rad))
        y = centery + (len * math.sin(rad))

        if x < 0:
            x = 0
        if y < 0:
            y = 0

        return x, y
    def distancemat_2(whratio, size):
        whratio = whratio[0]
        size = size[0]
        dismat = np.zeros(((len(whratio), len(whratio))))
        for index, (wh, sz) in enumerate(zip(whratio, size)):
            for index2, (wh2, sz2) in enumerate(zip(whratio, size)):
                dismat[index][index2] = np.linalg.norm((abs(wh2 - wh), abs(sz2 - sz)))
        # dismat = dismat/np.max(dismat) * 10
        return dismat
    def distancemat(listocenter):
        dismat = np.zeros(((len(listocenter), len(listocenter))))
        for index, coords in enumerate(listocenter):
            for index2, coords2 in enumerate(listocenter):
                dismat[index][index2] = np.linalg.norm(coords2 - coords)

        return dismat
    def centercoords(boxmat):
        a = np.zeros((len(boxmat), 2))
        for index, boxline in enumerate(boxmat):
            centerx = (boxline[0] + boxline[2]) / 2
            centery = (boxline[1] + boxline[3]) / 2

            a[index][0] = centerx
            a[index][1] = centery
        return a

    txt_list = []
    # txt_list = ["gt_112_frame(491).txt"]
    for file in os.listdir(filenamepath):
        if file.endswith(".txt"):
            txt_list.append(file)
    txt_list = natsorted(txt_list)  # sort the list 11th August 2022
    bla = 0

    # make sure there is no cloutput folder during reruns, or else it'll just append to existing txt files
    if os.path.isdir(cloutput_nf) == True:
        shutil.rmtree(cloutput_nf)

    for filename in txt_list:
        path1 = os.path.join(filenamepath, filename)

        with open(path1) as file:
            lines = file.readlines()
            img = cv2.imread(os.path.join(imgpath, filename[:-4] + ".jpg"))
            # print("why is img so shit")
            # print(os.path.join(imgpath, filename[:-4] + ".jpg"))
            # cv2.imshow("sdcsdc", img)
            # cv2.waitKey()

            if os.path.isdir(cloutput_nf) == False:
                os.makedirs(cloutput_nf)
            with open(os.path.join(cloutput_nf, filename), 'a') as f:

                if bool(lines) == False:
                    #cv2.imwrite(os.path.join(imgoutput, filename[:-4] + ".jpg"), img)
                    continue

                thearray = pro2nparr(lines)  # get xywh coordinates

                # # filter out boxes that are too small
                xmask = thearray[:, 2] > a_1
                thearray = thearray[xmask]
                ymask = thearray[:, 3] > a_1
                thearray = thearray[ymask]
                lines = np.array(lines)
                lines = lines[xmask]
                lines = lines[ymask]

                if lines.size == 0:
                    #cv2.imwrite(os.path.join(imgoutput, filename[:-4] + ".jpg"), img)
                    continue

                # filter out boxes that are too big
                xmask = thearray[:, 2] < a_2
                thearray = thearray[xmask]
                ymask = thearray[:, 3] < a_2
                thearray = thearray[ymask]
                lines = lines[xmask]
                lines = lines[ymask]

                if lines.size == 0:
                    #cv2.imwrite(os.path.join(imgoutput, filename[:-4] + ".jpg"), img)
                    continue

                getangle = getang(lines)
                boxmat = xywhn2xyxy(thearray)  # get xyxy coordinates
                listocenter = centercoords(boxmat)

                # for ind, blue in enumerate(listocenter):
                #     #print(boxmat)
                #     #print((round(blue[0]), round(blue[1]), round(blue[2]), round(blue[3])))
                #     #print(img.shape)
                #     cv2.circle(img, (round(blue[0]), round(blue[1])), 5, (255, 0, 0), 1)
                #     cv2.imshow('img', img)
                #     cv2.waitKey()

                distmat = distancemat(listocenter)
                listocluster = []
                seen = []

                # arbritrary distance 40
                for index, distline in enumerate(distmat):
                    flatlist = sum(listocluster, [])  # flatten the clusterlist so can use count
                    if flatlist.count(index) == False:
                        currcluster = []
                        clustindices = np.where(distline < a_3)
                        clustindices = list(clustindices[0])
                        currcluster = clustindices

                        count = 0
                        newindices = 1
                        while newindices != []:  # recursively add index until there is no point/indices that belongs to the cluster

                            clustindices = np.where(distmat[currcluster[count]] < a_3)
                            clustindices = list(clustindices[0])
                            for i in clustindices:

                                if currcluster.count(i) == False:
                                    newindices = i
                                    currcluster.append(i)
                                if count == (len(currcluster) - 1):
                                    newindices = []
                            count += 1
                        if len(currcluster) > 3:
                            listocluster.append(currcluster)

                print("listocluster")
                print(listocluster)
                listocluster = splitcluster2(listocluster,
                                             boxmat)  # split cluster that has different ratio and size (27th March 2022)
                print("listocluster after distance mat")
                print(listocluster)
                listocluster = splitcluster(listocluster, boxmat,
                                            listocenter)  # split cluster that does not intersect (26th March 2022)
                print("listocluster after non overlapping box")
                print(listocluster)
                # (31st March 2022) remove boxes that has redundant iou
                listocluster = rmvcluster(listocluster, boxmat)
                print("listocluster after omiting redundant box")
                print(listocluster)

                confirmfish = []
                for locind, sgcluster in enumerate(listocluster):  # listocluster indices, singlecluster
                    # print("sgcluter")
                    # print(sgcluster)
                    if len(sgcluster) >= 3:
                        relangle = getangle[sgcluster]  # check relevant angle
                        print("relangle")
                        print(relangle)
                        std = np.std(relangle)

                        print("Std")
                        print(std)

                        # check if 80% of points within the cluster has similar angle value
                        sortedangle = np.argsort(relangle.flatten())  # sortedangle are indices of cluster
                        # print("sorted angle")
                        # print(sortedangle)
                        eightylen = round(a_5 * len(relangle))  # length of eighty percent
                        # print("eightylen")
                        # print(eightylen)
                        # make sure after filtering have more than 3 angles is about the same
                        sorted = relangle[sortedangle]

                        if len(sgcluster) == 3 and std > a_6:
                            continue
                        top80_ind = sortedangle[0:eightylen]  # top angle 80 percent indices
                        topeighty = relangle[top80_ind]
                        bottom80_ind = sortedangle[-(eightylen):]  # bottom angle 80 percent indices
                        bottomeighty = relangle[bottom80_ind]
                        topeighty_std = np.std(topeighty)
                        bottomeighty_std = np.std(bottomeighty)
                        #             print("sorted")
                        #             print(sorted)
                        #             print("topeighty")
                        #             print(topeighty)
                        # print("topeighty std")
                        #             print(np.std(topeighty))
                        #             print("bottomeighty")
                        #             print(bottomeighty)
                        # print("bottomeightystd")
                        # print(topeighty_std)
                        # print(bottomeighty_std)
                        # we can start finding confirmed clusters
                        if topeighty_std <= a_6 or bottomeighty_std <= a_6:
                            eighty_std = np.argmin([topeighty_std, bottomeighty_std])
                            if eighty_std == 0:
                                eighty_ind = top80_ind
                            if eighty_std == 1:
                                eighty_ind = bottom80_ind  # eighty_ind is in relation to sgcluster, not listocenter
                            if std < a_6:  # if 3 is already less than 0.10 std
                                eighty_ind = np.arange(len(sgcluster))

                            sgcluster = np.array(sgcluster)
                            eighty_dm = distancemat(listocenter[sgcluster[np.array(eighty_ind)]])
                            #               print("")
                            #               print("eighty_dm")
                            #               print(eighty_dm)
                            #               print(eighty_dm.shape)

                            # from here we start to try and predict where the fish is according to cluster
                            # top eighty distance matrix
                            maxno = np.argmax(eighty_dm)
                            maxvalue = np.max(eighty_dm)
                            xmax = math.floor(
                                maxno / len(eighty_dm))  # listoclusters coordinate's indices for the current cluster
                            ymax = maxno % len(eighty_dm)

                            # get the coordinates of the largest distance in the current cluster
                            # print(xmax)
                            # print(ymax)
                            # print(maxvalue)

                            coords_ind1 = sgcluster[eighty_ind[xmax]]
                            coords_ind2 = sgcluster[eighty_ind[ymax]]

                            # image = cv2.circle(image, center_coordinates, radius, color, thickness)

                            coords1 = listocenter[coords_ind1]
                            coords2 = listocenter[coords_ind2]  # got it



                            cv2.circle(img, (round(coords1[0]), round(coords1[1])), 5, (255, 0, 0), 1)
                            cv2.circle(img, (round(coords2[0]), round(coords2[1])), 5, (255, 255, 255), 1)

                            # plan is to extrapolate the distance into the possible range of where the fish could be(so like a huge box)
                            # resulting box
                            resulting_ang = np.mean(relangle[eighty_ind])

                            xy1 = np.array(calccoords(maxvalue / 2, coords1[0], coords1[1], resulting_ang))
                            xy2 = np.array(calccoords(maxvalue / 2, coords2[0], coords2[1], resulting_ang))

                            # getting better box coords
                            # 28th March 2022
                            dy = coords2[1] - coords1[1]
                            dx = coords2[0] - coords1[0]
                            rad2coord = math.atan2(dy, dx)
                            rad2coord_flip = rad2coord + math.pi  # flipping rad2coord, consider flipping angle of 2 coordinats
                            nmlr = ((rad2coord + math.pi) % (math.pi * 2)) / (math.pi * 2)  # normalized radian
                            nmlr_flip = ((rad2coord_flip + math.pi) % (math.pi * 2)) / (math.pi * 2)

                            angdiff = abs(resulting_ang - nmlr)
                            angdiff2 = abs(resulting_ang - nmlr_flip)

                            # get the point where it is pointing to the direction of mean angle
                            coordind = np.argmin([angdiff, angdiff2])  # coord angle diff

                            # 23rd August 2022
                            # decide which point (relpoint) is in the direction of mean angle coords1 or coords2
                            if coordind == 0:
                                cooradiff = angdiff
                                coordang = nmlr
                                rbcent = coords2

                            if coordind == 1:
                                cooradiff = angdiff2
                                coordang = nmlr_flip
                                rbcent = coords1

                            #agreeability = 1 - (
                            #(cooradiff / 0.5))  # if coords angle match with angle, then it will be longer
                            #result_len = maxvalue * agreeability  # if not, resulting box will not be too far away
                            coocenterx = (coords1[0] + coords2[0]) / 2
                            coocentery = (coords1[1] + coords2[1]) / 2
                            cv2.circle(img, (round(coocenterx), round(coocentery)), 5, (255, 155, 74), 1)

                            #rbcent = calccoords(result_len, coocenterx, coocentery,
                            #                    (resulting_ang + coordang) / 2)  # resulting box center
                            forrbboxmat = boxmat[sgcluster]
                            w = np.mean(forrbboxmat[:, 2] - forrbboxmat[:, 0])
                            h = np.mean(forrbboxmat[:, 3] - forrbboxmat[:, 1])
                            rbxtop = rbcent[0] + (w / 2)  # resultbox xytop and xybottom
                            rbxbottom = rbcent[0] - (w / 2)
                            rbytop = rbcent[1] + (h / 2)
                            rbybottom = rbcent[1] - (h / 2)
                            cv2.rectangle(img, (round(rbxtop), round(rbytop)), (round(rbxbottom), round(rbybottom)),
                                          (0, 0, 255), 1)
                            cv2.circle(img, (round(rbcent[0]), round(rbcent[1])), 5, (255, 255, 0), 1)

                            # print("xy1 xy2")
                            # print(xy1)
                            # print(xy2)
                            # can start writing to txt file
                            # [x1, y1, x2, y2]
                            # coords = np.expand_dims(np.array([xy1[0], xy1[1], xy2[0], xy2[1]]), axis = 0)
                            # apparantly up til here need to detect which is top xy and whichi s bottom xy

                            # topxy and bottom xy processing-------------------------------------------------
                            # topx, topy = np.min([xy1[0], xy2[0]]), np.min([xy1[1], xy2[1]])
                            # bottomx, bottomy = np.max([xy1[0], xy2[0]]), np.max([xy1[1], xy2[1]])

                            # cv2.rectangle(img, (round(topx), round(topy)),
                            #               (round(bottomx), round(bottomy)), (0, 0,255), 1)

                            # convert2xywhn = np.expand_dims(np.array([topx, topy, bottomx, bottomy]), axis=0)
                            convert2xywhn = np.expand_dims(np.array([rbxtop, rbytop, rbxbottom, rbybottom]), axis=0)

                            towritecoords = xyxy2xywhn(convert2xywhn)

                            # print(towritecoords)
                            # print("")

                            towriteline = "0" + " " + str(towritecoords[0][0]) + " " + str(
                                towritecoords[0][1]) + " " + str(
                                towritecoords[0][2]) + " " + str(towritecoords[0][3]) + " " + str(0.5) + "\n"
                            f.write(towriteline)

                print("y")
                print(filename)
                # cv2.imshow('img', img)
                #cv2.imwrite(os.path.join(imgoutput, filename[:-4] + ".jpg"), img)
                # cv2.waitKey()
                bla += 1

                # if bla == 210:
                #     break

                #  seen should be somewhere where the whole clustindices will be deposited there or we don't even need that, just query listoclusters

def automatictroubleshoot(bestMAPconf, cf_0001_path_folder, cloutput_nf, gtpath, imgpath, IOU_cutoff, conf_cutoff):
    """
    process output from a_clusteroutput and fuse it with low_conf YOLORGB output

    :param bestMAPconf:
    :param cf_0001_path_folder: YOLORGB output
    :param cloutput_nf: output from a_clusteroutput
    :return:
    """
    def drawstuff_forcf0541(verify, filename, imgpath, color):
        verify_file = os.path.join(verify, filename)
        with open(verify_file) as file:
            verlines = file.readlines()
            if verlines == []:
                verlines = ['0 0 0 0 0 0\n']
        confarray = getconf(verlines)
        ver_array = pro2nparr(verlines)
        ver_boxmat = xywhn2xyxy(ver_array)
        filter = (confarray > bestMAPconf).ravel()
        boxmat_0541 = ver_boxmat[filter]

        tempimg = cv2.imread(os.path.join(imgpath, filename[:-4] + '.jpg'))

        for cind, fin in enumerate(boxmat_0541):
            cv2.rectangle(tempimg, (round(fin[0]), round(fin[1])), (round(fin[2]), round(fin[3])), color, 1)
            cv2.putText(tempimg, str(np.round(confarray[cind], 3)), (round(fin[0]), round(fin[1]) - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        return tempimg
    def get_unique(FP_IOU, FP_conf, TP_IOU, TP_conf, a=IOU_cutoff, b=conf_cutoff, within=True, cf0001ind_towardori=None,
                   cf0001ind_towardori_FP=None):
        """
        get array of FPTP IOU array and conf array
        get parameters a - IOU cutoff, b - conf cutoff
        Return:
            1) FP IOU and FP conf unique array (filtered array)
            2) TP IOU and TP conf unique array (filtered array)
        """
        i1 = np.where(FP_IOU > a)[0]
        c1 = np.where(FP_conf > b)[0]
        i2 = np.where(TP_IOU > a)[0]
        c2 = np.where(TP_conf > b)[0]
        # print("is it index or numbers?") # it's index
        # print(i2)
        # FP_unique = np.unique(np.concatenate((i1, c1), axis=0))
        # TP_unique = np.unique(np.concatenate((i2, c2), axis=0))
        FP_unique = np.intersect1d(i1, c1)
        TP_unique = np.intersect1d(i2, c2)
        if within == True:
            return cf0001ind_towardori_FP[FP_unique], cf0001ind_towardori[TP_unique]
        if within == False:
            return FP_unique, TP_unique

    # initialize some parameters fpr "sofar" variables
    IOUsofar = np.array([])
    confsofar = np.array([])
    IOUsofar_FP = np.array([])
    confsofar_FP = np.array([])
    FNsofardict = {}  # a dictionary that stores image name and the array of FN gt
    gtcount = 0
    filteredcount = 0
    gtFNcount = 0
    afterfilterFP = 0
    afterfilterTP = 0
    #bestMAPconf = 0.541  # best mAP conf, 0.541 for val data, 0.532 for train data
    fiveFeaturesnpy_FP = np.empty((1, 5))
    fiveFeaturesnpy_TP = np.empty((1, 5))

    txt_list = []
    # txt_list = ["gt_217_frame(259).txt"]
    for file in os.listdir(cf_0001_path_folder):
        if file.endswith(".txt"):
            txt_list.append(file)

    for fi, filename in enumerate(txt_list):
        print(bcolors.HEADER + "FILENAME: " + str(filename) + bcolors.ENDC)
        cf_0001_path = os.path.join(cf_0001_path_folder, filename)

        # sometimes, there is no cloutput file, so need to create one, that way FN and annotation calculation can continue
        if os.path.exists(os.path.join(cloutput_nf, filename)) == False:
            # create empty missing cl file
            with open(os.path.join(cloutput_nf, filename), 'a') as f:
                pass

        cl_output_path = os.path.join(cloutput_nf, filename)
        gt_filepath = os.path.join(gtpath, filename)

        # process gt_lines
        with open(gt_filepath) as gt_file:
            gt_lines = gt_file.readlines()
            if len(gt_lines) == 0:
                print("no gt lines in " + filename)
                continue
            gtcount += len(gt_lines)
            thearray_gt = pro2nparr(gt_lines)
            boxmat_gt = xywhn2xyxy(thearray_gt)

        # process cf_0001_lines
        with open(cf_0001_path) as cf_file:
            cf_0001_lines = cf_file.readlines()
            if len(cf_0001_lines) == 0:
                print("no cf 0001 lines in " + filename)

                ''' '''
                # 1st October: code block related to FN no. 1
                # increment gtFNcount since there's no cf
                gtFNcount += len(gt_lines)
                FNsofardict[filename] = np.array([])
                print("")
                print("FN test--------------")
                print(len(gt_lines))
                print(" ")
                ''' '''
                continue
            thearray_cf0001 = pro2nparr(cf_0001_lines)
            boxmat_cf0001 = xywhn2xyxy(thearray_cf0001)
            conf_cf0001 = getconf(cf_0001_lines)

        # make sure cl_output_lines isn't empty
        with open(cl_output_path) as cl_file:
            cl_output_lines = cl_file.readlines()
            if len(cl_output_lines) == 0:
                print("no cl lines in " + filename)
                ''' '''
                # 1st October: code block related to FN no. 2
                # increment gtFNcount since there's no cl

                filtered_cf0001_ind_inv = np.where(conf_cf0001.ravel() > bestMAPconf)[
                    0]  # index in relation to original cf lines
                filteredcount += len(filtered_cf0001_ind_inv)
                filt_boxmat_cf0001_inv = boxmat_cf0001[filtered_cf0001_ind_inv]
                tempmat = def_ioumat(filt_boxmat_cf0001_inv, boxmat_gt)
                if len(filt_boxmat_cf0001_inv) != 0:
                    maxcfIOU = np.max(tempmat, axis=0)
                    indwhere = np.where(maxcfIOU > 0.5)[0]  # which index of gt overlaps with high_cf > 0.5
                    indwhere_inv = np.where(maxcfIOU < 0.5)[0]
                    gtFNcount += (len(gt_lines) - len(indwhere))  # len(indwhere) current TP for this frame
                    FNsofardict[filename] = boxmat_gt[indwhere_inv]
                if len(filt_boxmat_cf0001_inv) == 0:
                    gtFNcount += len(gt_lines)
                    FNsofardict[filename] = boxmat_gt
                ''' '''
                continue
            # process cl_output_lines
            thearray_cl = pro2nparr(cl_output_lines)
            boxmat_cl = xywhn2xyxy(thearray_cl)

            # choose highest clusteroutput IOU overlap with low_cf if there are
            # multiple
            # but first, need to convert cf and cloutput to boxmat format

            filtered_cf0001_ind = np.where(conf_cf0001.ravel() < bestMAPconf)[
                0]  # index in relation to original cf lines
            filtered_cf0001_ind_inv = np.where(conf_cf0001.ravel() > bestMAPconf)[
                0]  # index in relation to original cf lines
            filteredcount += len(filtered_cf0001_ind_inv)
            filt_boxmat_cf0001 = boxmat_cf0001[filtered_cf0001_ind]  # boxmat where conf < bestMAPconf
            filt_boxmat_cf0001_inv = boxmat_cf0001[filtered_cf0001_ind_inv]  # boxmat where conf > bestMAPconf

            # get indices of filtered cf_0001 (lowcf) that overlaps with gt > IOU 0.5
            IOUmat_cfgt = def_ioumat(filt_boxmat_cf0001, boxmat_gt)
            if toggle == 0 or toggle == 1:
                print("lenghts of gt, filtcf0001, cloutput")
                print("len of gt: " + str(len(boxmat_gt)))
                print("len of filt cf 0001: " + str(len(filt_boxmat_cf0001)))
                print("len of filt cf 0001 inv: " + str(len(filt_boxmat_cf0001_inv)))
                print("len of cloutput: " + str(len(boxmat_cl)))
                print("IOUmat_cfgt " + filename)
                print(IOUmat_cfgt)
            maxrowIOU_gtcf = np.max(IOUmat_cfgt, axis=1)
            cfind_IOU05 = np.where(maxrowIOU_gtcf > 0.5)[0]

            '''getting FP [11th September]
            # get indices of filtered cf_0001 (lowcf) where it overlaps with gt < IOU0.5 
            '''
            cfind_IOU05_inv = np.where(maxrowIOU_gtcf < 0.5)[0]
            ''''''

            # 3rd September 2022 18:02: cancel [low cf that has IOU with gt > 0.5]
            # that has big ish overlap with high cf
            # REMINDER:
            # filt_boxmat_cf0001[cfind_IOU05] = cf0001 < bestMAPconf that overlaps with gt > IOU 0.5
            # filt_boxmat_cf0001_inv = cf > bestMAPconf
            IOUmat_cfhicflo = def_ioumat(filt_boxmat_cf0001[cfind_IOU05],
                                         filt_boxmat_cf0001_inv)  # overlap between cf high and cf low

            # to solve np.max identity problem around ln 254
            if len(filt_boxmat_cf0001[cfind_IOU05]) == 0:
                IOUmat_cfhicflo = np.empty((0, 1))
            elif len(filt_boxmat_cf0001_inv) == 0:
                IOUmat_cfhicflo = np.zeros((len(filt_boxmat_cf0001[cfind_IOU05]), 1))

            if toggle == 0 or toggle == 1:
                print("cf high cf low")
                print("filt_boxmat_cf0001_inv: " + str(len(filt_boxmat_cf0001_inv)))
                print("filt_boxmat_cf0001[cfind_IOU05]: " + str(len(filt_boxmat_cf0001[cfind_IOU05])))
                print(IOUmat_cfhicflo)

            # this index of (low cf that overlaps with gt > IOU 0.5 [filt_boxmat_cf0001[cfind_IOU05]) that does not overlaps with a high cf
            overlaphi_low_ind = np.where(np.max(IOUmat_cfhicflo, axis=1) == 0)[0]

            if toggle == 0 or toggle == 1:
                print("overlaphilow")
                print(overlaphi_low_ind)

            # later on all related cfindIOU05 needs to be edited
            # overlap of cl with [cf that overlaps > 0.5 with gt that does not overlap with high cond cf]
            IOUmat_cfcl = def_ioumat(filt_boxmat_cf0001[cfind_IOU05][overlaphi_low_ind], boxmat_cl)

            maxrowIOU_cfcl = np.max(IOUmat_cfcl, axis=1)
            cfclind_IOU01 = np.where(maxrowIOU_cfcl > 0.1)[
                0]  # for each cf that has overlap IOU > 0.1 with cl is considered
            clind_IOUmaxn01 = np.argmax(IOUmat_cfcl, axis=1)[
                cfclind_IOU01]  # cl index that has overlap IOU > 0.1 with cf
            cfcl_IOU_array = maxrowIOU_cfcl[cfclind_IOU01]

            '''getting FP [11th September]
            # cancel [low cf that has IOU with gt > 0.5] that has big ish overlap with high cf
            '''
            IOUmat_cfhicflo_v2 = def_ioumat(filt_boxmat_cf0001[cfind_IOU05_inv], filt_boxmat_cf0001_inv)

            # to solve np.max identity problem
            if len(filt_boxmat_cf0001[cfind_IOU05_inv]) == 0:
                IOUmat_cfhicflo_v2 = np.empty((0, 1))
            elif len(filt_boxmat_cf0001_inv) == 0:
                IOUmat_cfhicflo_v2 = np.zeros((len(filt_boxmat_cf0001[cfind_IOU05_inv]), 1))

            # this index of (low cf that overlaps with gt < IOU 0.5 [filt_boxmat_cf0001[cfind_IOU05_inv]) that does not overlaps with a high cf
            overlaphi_low_ind_v2 = np.where(np.max(IOUmat_cfhicflo_v2, axis=1) == 0)[0]

            # overlap of cl with [cf that overlaps < 0.5 with gt that does not overlap with high cond cf]
            IOUmat_cfcl_FP = def_ioumat(filt_boxmat_cf0001[cfind_IOU05_inv][overlaphi_low_ind_v2], boxmat_cl)
            maxrowIOU_cfcl_FP = np.max(IOUmat_cfcl_FP, axis=1)
            cfclind_IOU01_FP = np.where(maxrowIOU_cfcl_FP > 0.1)[
                0]  # for each cf_FP that has overlap IOU > 0.1 with cl is considered
            clind_IOUmaxn01_FP = np.argmax(IOUmat_cfcl_FP, axis=1)[
                cfclind_IOU01_FP]  # cl index that has overlap IOU > 0.1 with cf
            cfcl_IOU_array_FP = maxrowIOU_cfcl_FP[cfclind_IOU01_FP]
            ''''''                          ''''''

            # parameters to take note:
            # nested indices in regards to original cf lines
            # boxmat_cf0001[filtered_cf0001_ind][cfind_IOU05][overlaphi_low_ind][cfclind_IOU01]
            cf0001ind_towardori = filtered_cf0001_ind[cfind_IOU05][overlaphi_low_ind][cfclind_IOU01]

            '''getting FP [11th September] nested indices in regards to original cf lines (but this time
            it's for low cf that DOES NOT overlap with gt)'''
            cf0001ind_towardori_FP = filtered_cf0001_ind[cfind_IOU05_inv][overlaphi_low_ind_v2][cfclind_IOU01_FP]
            '''         '''         '''         '''

            # remove low cf that share the same cl, higher conf is chosen 5th September------
            if len(clind_IOUmaxn01) != 0:

                toremoveind, newcl = remove_duplicl(clind_IOUmaxn01, conf_cf0001[cf0001ind_towardori])
                if len(toremoveind) > 0:
                    clind_IOUmaxn01 = newcl
                    cf0001ind_towardori = np.delete(cf0001ind_towardori, toremoveind)
                    cfcl_IOU_array = np.delete(cfcl_IOU_array, toremoveind)

            '''getting the same variables but FP version [11th September]'''
            if len(clind_IOUmaxn01_FP) != 0:
                toremoveind_FP, newcl_FP = remove_duplicl(clind_IOUmaxn01_FP, conf_cf0001[cf0001ind_towardori_FP])
                if len(toremoveind_FP) > 0:
                    clind_IOUmaxn01_FP = newcl_FP
                    cf0001ind_towardori_FP = np.delete(cf0001ind_towardori_FP, toremoveind_FP)
                    cfcl_IOU_array_FP = np.delete(cfcl_IOU_array_FP, toremoveind_FP)
            '''         '''         '''         '''
            # --------------------------------UPDATED-----------------------------------------------

            # ----------------------------UPDATE 11TH OCTOBER 2022 ON ELIMINATING FP AND TP OVERLAP AND CHOOSING THE ONE WITH HIGHER CONF-----
            # find IOU_mat that overlaps between FP and TP
            # make sure both of them are not empty to make sure no mat error will occur
            # also if one of them is empty no overlap will occur anyways
            if len(cf0001ind_towardori_FP) != 0 and len(cf0001ind_towardori) != 0:
                FP_TP_IOU_mat = def_ioumat(boxmat_cf0001[cf0001ind_towardori_FP], boxmat_cf0001[cf0001ind_towardori])
                index_arr = np.where(FP_TP_IOU_mat > 0.1)  # arbitrary overlap IOU parameter (0.1 for now)
                rmvTPind = []
                rmvFPind = []
                if len(index_arr[0]) != 0 and len(index_arr[1]) != 0:
                    for ftind in range(len(index_arr[0])):
                        rmvFPorTP = np.argmin((conf_cf0001[cf0001ind_towardori_FP][index_arr[0][ftind]],
                                               conf_cf0001[cf0001ind_towardori][index_arr[1][ftind]]))
                        if rmvFPorTP == 0:
                            rmvFPind.append(index_arr[0][ftind])
                        if rmvFPorTP == 1:
                            rmvTPind.append(index_arr[1][ftind])
                if len(rmvTPind) > 0:
                    # convert to np array so I can use np unique
                    rmvTPind = np.unique(np.array(rmvTPind))
                    # then remove from the list, IOU array, conf and box _TP
                    clind_IOUmaxn01 = np.delete(clind_IOUmaxn01, rmvTPind)
                    cf0001ind_towardori = np.delete(cf0001ind_towardori, rmvTPind)
                    cfcl_IOU_array = np.delete(cfcl_IOU_array, rmvTPind)

                if len(rmvFPind) > 0:
                    # convert to np array so I can use np unique
                    rmvFPind = np.unique(np.array(rmvFPind))
                    # then remove from the list, IOU array, conf and box _TP
                    clind_IOUmaxn01_FP = np.delete(clind_IOUmaxn01_FP, rmvFPind)
                    cf0001ind_towardori_FP = np.delete(cf0001ind_towardori_FP, rmvFPind)
                    cfcl_IOU_array_FP = np.delete(cfcl_IOU_array_FP, rmvFPind)

            # ----------------------------11TH OCTOBER UPDATE END ----------------------------------------------------------------------------

            ''''''         '''         '''
            # 1st October 2022
            # code block responsible for False Negative (FN) no.3
            # list/array of conf0001 (high and low) boxes that is considered as detection AND also overlaps with gt > 0.5
            selectedcf = np.concatenate((boxmat_cf0001[filtered_cf0001_ind_inv], boxmat_cf0001[cf0001ind_towardori]),
                                        axis=0)
            gt_mat = def_ioumat(boxmat_gt, selectedcf)  # try to find array of gt that has not got an overlap
            # ind = np.where(np.max(randarray, axis = 1) < 0.5)
            # index of gt that DOES NOT overlap with any detection (low or high cf)
            if len(selectedcf) == 0:
                gtFNcount += len(boxmat_gt)
                FNsofardict[filename] = boxmat_gt
                print("")
                print("FN test scf==0-------------")
                print(len(boxmat_gt))
            if len(selectedcf) > 0:
                gtFNind = np.where(np.max(gt_mat, axis=1) < 0.5)
                gtFNcount += len(gtFNind[0])  # increment FN count
                FNsofardict[filename] = boxmat_gt[gtFNind]  # record down filename gt array pairs
                print("")
                print("FN test---scf>=0----------")
                print(len(gtFNind[0]))
            ''''''         '''          '''

            if toggle == 0 or toggle == 1:
                print("clind_IOUmaxn01" + str(clind_IOUmaxn01))  # which cl index
                print("cfclind_IOU01" + str(cfclind_IOU01))  # which cf index
                print("cfcl_IOU_array" + str(cfcl_IOU_array))  # its value

            # block in charge of TP details output
            if toggle == 2 or toggle == 0:
                if len(cfcl_IOU_array) == 0:
                    print("no extra TP for " + str(filename))
                else:
                    IOUsofar = np.concatenate((IOUsofar, cfcl_IOU_array), axis=None)
                    confsofar = np.concatenate((confsofar, conf_cf0001[cf0001ind_towardori]), axis=None)

                    print(
                        bcolors.OKCYAN + "STATS for TP:-------------------------------------------------- " + filename + bcolors.ENDC)
                    print(bcolors.BOLD + "Individual ROI stats:" + bcolors.ENDC)
                    print("  frame index from cf0001: " + str(
                        np.delete(filtered_cf0001_ind[cfind_IOU05][overlaphi_low_ind][cfclind_IOU01], toremoveind)))
                    print("  IOU array :" + str(cfcl_IOU_array))
                    print("  conf: " + str(conf_cf0001[cf0001ind_towardori]))
                    print("")
                    print(bcolors.BOLD + "Per frame stats: " + bcolors.ENDC)
                    print("  total no of potential extra TP for this frame: " + str(len(cfcl_IOU_array)))
                    print("  IOU range: min[" + str(np.min(cfcl_IOU_array)) + "] max[" + str(
                        np.max(cfcl_IOU_array)) + "]")
                    print("  IOU average : " + str(np.mean(cfcl_IOU_array)))
                    print("  conf range: min[" + str(np.min(conf_cf0001[cf0001ind_towardori])) + "] max[" + str(
                        np.max(conf_cf0001[cf0001ind_towardori])) + "]")
                    print("  conf average: " + str(np.mean(conf_cf0001[cf0001ind_towardori])))
                    print(" ")
                    print(bcolors.BOLD + "Total stats so far: " + bcolors.ENDC)
                    print("  total no of potential extra TP so far: " + str(len(confsofar)))
                    print("  double check: " + str(len(IOUsofar)))
                    print("  IOU range so far: min[" + str(np.min(IOUsofar)) + "] max[" + str(np.max(IOUsofar)) + "]")
                    print("  IOU average so far: " + str(np.mean(IOUsofar)))
                    print(
                        "  conf range so far: min[" + str(np.min(confsofar)) + "] max[" + str(np.max(confsofar)) + "]")
                    print("  conf average so far: " + str(np.mean(confsofar)))
                    print(" ")
                    print("no. of total gt annotations: " + str(gtcount))  # number of groundtruth annotations
                    print("no. of total detected high cf: " + str(filteredcount))  # number of detected high conf cf
                    print(bcolors.WARNING + "just remember IOU means the IOU between cf and cl" + bcolors.ENDC)
                    print(bcolors.OKCYAN + "-------------------------------------------------- " + bcolors.ENDC)
                    print(" ")

            # block in charge of FP details output
            if toggle == 2 or toggle == 0:
                if len(cfcl_IOU_array_FP) == 0:
                    print("no extra FP for" + str(filename))

                else:
                    IOUsofar_FP = np.concatenate((IOUsofar_FP, cfcl_IOU_array_FP), axis=None)
                    confsofar_FP = np.concatenate((confsofar_FP, conf_cf0001[cf0001ind_towardori_FP]), axis=None)
                    print(
                        bcolors.MEHGREEN + "STATS for FP:-------------------------------------------------- " + filename + bcolors.ENDC)
                    print(bcolors.BOLD + "Individual ROI stats:" + bcolors.ENDC)
                    print("  frame index from cf0001: " + str(
                        np.delete(filtered_cf0001_ind[cfind_IOU05_inv][overlaphi_low_ind_v2][cfclind_IOU01_FP],
                                  toremoveind_FP)))
                    print("  IOU array :" + str(cfcl_IOU_array_FP))
                    print("  conf: " + str(conf_cf0001[cf0001ind_towardori_FP]))
                    print("")
                    print(bcolors.BOLD + "Per frame stats: " + bcolors.ENDC)
                    print("  total no of potential extra FP for this frame: " + str(len(cfcl_IOU_array_FP)))
                    print("  IOU range: min[" + str(np.min(cfcl_IOU_array_FP)) + "] max[" + str(
                        np.max(cfcl_IOU_array_FP)) + "]")
                    print("  IOU average : " + str(np.mean(cfcl_IOU_array_FP)))
                    print("  conf range: min[" + str(np.min(conf_cf0001[cf0001ind_towardori_FP])) + "] max[" + str(
                        np.max(conf_cf0001[cf0001ind_towardori_FP])) + "]")
                    print("  conf average: " + str(np.mean(conf_cf0001[cf0001ind_towardori_FP])))
                    print(" ")
                    print(bcolors.BOLD + "Total stats so far: " + bcolors.ENDC)
                    print("  total no of potential extra FP so far: " + str(len(confsofar_FP)))
                    print("  double check: " + str(len(IOUsofar_FP)))
                    print("  IOU range so far: min[" + str(np.min(IOUsofar_FP)) + "] max[" + str(
                        np.max(IOUsofar_FP)) + "]")
                    print("  IOU average so far: " + str(np.mean(IOUsofar_FP)))
                    print("  conf range so far: min[" + str(np.min(confsofar_FP)) + "] max[" + str(
                        np.max(confsofar_FP)) + "]")
                    print("  conf average so far: " + str(np.mean(confsofar_FP)))
                    print(" ")
                    print("no. of total gt annotations: " + str(gtcount))  # number of groundtruth annotations
                    print("no. of total detected high cf: " + str(filteredcount))  # number of detected high conf cf
                    print(bcolors.WARNING + "just remember IOU means the IOU between cf and cl" + bcolors.ENDC)
                    print(bcolors.OKCYAN + "-------------------------------------------------- " + bcolors.ENDC)
                    print(" ")

            # VISUAL AID SO i DON'T GET CONFUSED AWFHCAWUEHAOUEHAWUIHC@&$&%@(*#&(@*$&&^@$*(#@*&$^@&*(!@#*$&^@@!*(#*$&^@*(!
            # cl ind, IOU array, index left towards original cf
            # clind_IOUmaxn01, cfcl_IOU_array(the amount of overlap, follow clindIOUmax01 so don't worry), cf0001ind_towardori
            img = cv2.imread(os.path.join(imgpath, filename[:-4] + ".jpg"))
            img2 = drawstuff_forgt(gtpath, filename, imgpath, (0, 0, 255))
            img3 = drawstuff_forcf(cf_0001_path_folder, cf0001ind_towardori, cfcl_IOU_array, filename, img2,
                                   (255, 0, 0))
            img4 = drawstuff_forcl(cloutput_nf, clind_IOUmaxn01, filename, img3, (0, 155, 0))
            # neon blue for cf that represents FP
            img5 = drawstuff_forcf(cf_0001_path_folder, cf0001ind_towardori_FP, cfcl_IOU_array_FP, filename, img4,
                                   (226, 249, 17))
            # neon green for cl that represents cl that "supported" FP_cfs
            img6 = drawstuff_forcl(cloutput_nf, clind_IOUmaxn01_FP, filename, img5, (49, 247, 14))

            # cv2.imwrite(os.path.join(outputfolder30, filename[:-4] + ".jpg"), img6)
            # cv2.imshow("hello", img4)
            # cv2.imshow("hello", img6)
            # cv2.waitKey()

            # 22nd November 2022
            # make 5 features
            # for FP
            if len(cfcl_IOU_array_FP) != 0:
                five_features_FP = make_5features(cfcl_IOU_array_FP, conf_cf0001, cf0001ind_towardori_FP,
                                                  thearray_cf0001)
            if len(
                    cfcl_IOU_array_FP) == 0:  # make an empty array, later code write condition so it won't append empty array
                five_features_FP = np.array([])

            # for TP
            if len(cfcl_IOU_array) != 0:
                five_features_TP = make_5features(cfcl_IOU_array, conf_cf0001, cf0001ind_towardori, thearray_cf0001)
            if len(
                    cfcl_IOU_array) == 0:  # make an empty array, later code write condition so it won't append empty array
                five_features_TP = np.array([])

            # 22nd Nov 2022 [appending it]
            if len(five_features_FP) != 0:
                fiveFeaturesnpy_FP = np.concatenate((fiveFeaturesnpy_FP, five_features_FP), axis=0)
            if len(five_features_TP) != 0:
                print("what is going on")
                print(fiveFeaturesnpy_TP)
                fiveFeaturesnpy_TP = np.concatenate((fiveFeaturesnpy_TP, five_features_TP), axis=0)

            # # 21st Nov 2022 edit out array to loop to filtered array
            # # # recursively write FP ROI to FP_ROI_folder
            # FP_u, TP_u = get_unique(cfcl_IOU_array_FP, conf_cf0001[cf0001ind_towardori_FP],
            #                         cfcl_IOU_array, conf_cf0001[cf0001ind_towardori], within = True,
            #                         cf0001ind_towardori = cf0001ind_towardori, cf0001ind_towardori_FP = cf0001ind_towardori_FP)
            # if len(FP_u) != 0:
            #     for FP_ind, FP_box in enumerate(boxmat_cf0001[FP_u]):
            #         # transform to int
            #         # 21st Nov check width and height of current FP ROI see if it falls in the category
            #         FP_w = thearray_cf0001[FP_u[FP_ind]][2]
            #         FP_h = thearray_cf0001[FP_u[FP_ind]][3]
            #         if FP_w < 0.3 and FP_h < 0.3:
            #             xyxy = [int(x) for x in FP_box]
            #             x1, y1, x2, y2 = xyxy
            #             cv2.imwrite(os.path.join(FP_ROI_folder_21stNov, filename[:-4] + "_{}".format(FP_ind) + ".jpg"),
            #                         img[y1:y2, x1:x2, :])
            #             afterfilterFP += 1
            #
            # # # recursively write TP ROI to TP_ROI_folder
            # if len(TP_u) != 0:
            #     for TP_ind, TP_box in enumerate(boxmat_cf0001[TP_u]):
            #         # transform to int
            #         # 21st Nov check width and height of current TP ROI see if it falls in the category
            #         TP_w = thearray_cf0001[TP_u[TP_ind]][2]
            #         TP_h = thearray_cf0001[TP_u[TP_ind]][3]
            #         if TP_w < 0.3 and TP_h < 0.3:
            #             xyxy = [int(x) for x in TP_box]
            #             x1, y1, x2, y2 = xyxy
            #             cv2.imwrite(os.path.join(TP_ROI_folder_21stNov, filename[:-4] + "_{}".format(TP_ind) + ".jpg"),
            #                         img[y1:y2, x1:x2, :])
            #             afterfilterTP += 1

    # save five features as an npy file [22nd November 2022]
    # np.save("five_features_FP_train.npy", fiveFeaturesnpy_FP[1:])
    # np.save("five_features_TP_train.npy", fiveFeaturesnpy_TP[1:])

    print(bcolors.WARNING + "no. of total gt annotations: " + str(gtcount))  # number of groundtruth annotations
    print("no. of total detected high cf: " + str(filteredcount) + bcolors.ENDC)  # number of detected high conf cf
    print(bcolors.BOLD + "Total stats so far for FP: " + bcolors.ENDC)
    print("  total no of potential extra FP so far: " + str(len(confsofar_FP)))
    print("  double check: " + str(len(IOUsofar_FP)))
    print("  IOU range so far: min[" + str(np.min(IOUsofar_FP)) + "] max[" + str(np.max(IOUsofar_FP)) + "]")
    print("  IOU average so far: " + str(np.mean(IOUsofar_FP)))
    print("  conf range so far: min[" + str(np.min(confsofar_FP)) + "] max[" + str(np.max(confsofar_FP)) + "]")
    print("  conf average so far: " + str(np.mean(confsofar_FP)))
    print(" ")
    print(bcolors.BOLD + "Total stats so far for TP: " + bcolors.ENDC)
    print("  total no of potential extra TP so far: " + str(len(confsofar)))
    print("  double check: " + str(len(IOUsofar)))
    print("  IOU range so far: min[" + str(np.min(IOUsofar)) + "] max[" + str(np.max(IOUsofar)) + "]")

    print("")
    print("  IOU average so far: " + str(np.mean(IOUsofar)))
    print("  conf range so far: min[" + str(np.min(confsofar)) + "] max[" + str(np.max(confsofar)) + "]")
    print("  conf average so far: " + str(np.mean(confsofar)))
    print(bcolors.BOLD + "Total stats so far for FN: " + bcolors.ENDC)
    print("  total no of FN: " + str(gtFNcount))
    print(" ")
    print(" ")
    print(" ")
    print(print(bcolors.WARNING + "remember variable FNsofardict, dictionary that has filename and"
                                  "FN boxmat gt pairs " + bcolors.ENDC))  # number of groundtruth annotations)

    print(" ")
    print("after filtered FPTP:")
    FP_unique, TP_unique = get_unique(IOUsofar_FP, confsofar_FP, IOUsofar, confsofar, within=False)
    print("FP_unique length: " + str(len(FP_unique)))
    print("TP_unique length: " + str(len(TP_unique)))
    print("")
    print(bcolors.MEHGREEN + "after filter FP: " + str(afterfilterFP))
    print("after filter TP: " + str(afterfilterTP))

    print("")
    print("end of automatictroubleshoot repre")

def automaticstatistics_maker(cf_0001_path_folder, bestMAPconf, gtpath ):
    txt_list = []
    # txt_list = ["gt_201_frame(105).txt"]
    for file in os.listdir(cf_0001_path_folder):
        if file.endswith(".txt"):
            txt_list.append(file)

    gtcount = 0
    FN = 0
    FP = 0
    TP = 0
    # highconftotal (it's just total number of bounding boxes > 0.532, just double checking, numbers should
    # be the same as TP + FP)
    highconftotal = 0

    for fi, filename in enumerate(txt_list):
        print(bcolors.HEADER + "FILENAME: " + str(filename) + bcolors.ENDC)
        cf_0001_path = os.path.join(cf_0001_path_folder, filename)
        gt_filepath = os.path.join(gtpath, filename)
        # process gt_lines
        with open(gt_filepath) as gt_file:
            gt_lines = gt_file.readlines()
            if len(gt_lines) == 0:
                print("no gt lines in " + filename)
                continue
            gtcount += len(gt_lines)
            thearray_gt = pro2nparr(gt_lines)
            boxmat_gt = xywhn2xyxy(thearray_gt)
        with open(cf_0001_path) as cf_file:
            cf_0001_lines = cf_file.readlines()
            if len(cf_0001_lines) == 0:
                print("no cf 0001 lines in " + filename)

                ''' '''
                # 1st October: code block related to FN no. 1
                # increment gtFNcount since there's no cf
                FN += len(gt_lines)
                print("")
                print("FN test--------------")
                print(len(gt_lines))
                print(" ")
                ''' '''
                continue
        thearray_cf0001 = pro2nparr(cf_0001_lines)
        boxmat_cf0001 = xywhn2xyxy(thearray_cf0001)
        conf_cf0001 = getconf(cf_0001_lines)

        indtow_cf0001ori = np.where(conf_cf0001.ravel() > bestMAPconf)[0]
        highconftotal += len(indtow_cf0001ori)
        highconfbox = boxmat_cf0001[indtow_cf0001ori]  # boxmat > bestMAPconf

        if len(highconfbox) == 0:  # no detections, thus only FN is increased, no TP and FP since there's no detection
            FN += len(gt_lines)
            continue

        # if high conf box overlap gt > IOU 0.5, considered TP
        # IOUmat_gtcfhi, gt horizontal (every column), high cf vertical (every row)
        IOUmat_gtcfhi = def_ioumat(highconfbox, boxmat_gt)
        # in gt axis, get highest IOU score with cf (let's say there's 4 gt, the following array should have 4)
        maxcfIOU = np.max(IOUmat_gtcfhi, axis=0)
        argmaxcfIOU = np.argmax(IOUmat_gtcfhi, axis=0)  # which index of high_cf has highest overlap with each gt

        ''' 22nd October for the purpose of cropping and saving FPTP ROIs '''
        cfIOUmaxwithgt = np.max(IOUmat_gtcfhi, axis=1)  # returns an IOU array of each cf's highest IOU with any gt
        highcf_ind05IOU = np.where(cfIOUmaxwithgt > 0.5)[0]  # index from highconf box of IOU > 0.5 (as TP)
        highcf_ind05IOU_inv = np.where(cfIOUmaxwithgt < 0.5)[0]  # index from highconf box of IOU < 0.5 (as FP)

        ROITP = highconfbox[highcf_ind05IOU]
        ROIFP = highconfbox[highcf_ind05IOU_inv]

        # img = cv2.imread(os.path.join(imgpath, filename[:-4] + ".jpg"))
        #
        # # recursively write FP ROI to FP_ROI_folder (low conf)
        # if len(ROIFP) != 0:
        #     for FP_ind, FP_box in enumerate(ROIFP):
        #         # transform to int
        #         xyxy = [int(x) for x in FP_box]
        #         x1, y1, x2, y2 = xyxy
        #         cv2.imwrite(os.path.join(FP_ROI_folder, filename[:-4] + "_{}".format(FP_ind) + ".jpg"),
        #                     img[y1:y2, x1:x2, :])
        #
        # # recursively write TP ROI to TP_ROI_folder (low conf)
        # if len(ROITP) != 0:
        #     for TP_ind, TP_box in enumerate(ROITP):
        #         # transform to int
        #         xyxy = [int(x) for x in TP_box]
        #         x1, y1, x2, y2 = xyxy
        #         cv2.imwrite(os.path.join(TP_ROI_folder, filename[:-4] + "_{}".format(TP_ind) + ".jpg"),
        #                     img[y1:y2, x1:x2, :])
        # '''     '''     '''    '''
        # cv2.waitKey()
        indwhere = np.where(maxcfIOU > 0.5)[0]  # which index of gt overlaps with high_cf > 0.5
        cur_TP = len(indwhere)  # current TP for this frame
        TP += cur_TP  # update total TP

        # check how many gt left, considered FN
        cur_FN = len(np.where(maxcfIOU < 0.5)[0])
        FN += cur_FN

        # if high conf box overlap gt < IOU 0.5 considered FP (also 2 overlapping high cf with a single gt problem has been noted and mitigated)
        cur_FP = len(indtow_cf0001ori) - len(indwhere)
        FP += cur_FP

    print(bcolors.WARNING + "no. of total gt annotations: " + str(gtcount))  # number of groundtruth annotations
    print("no. of total detected high cf: " + str(highconftotal) + bcolors.ENDC)  # number of detected high conf cf
    print("double check TP + FP (should have same value as high cf) : " + str(TP + FP) + bcolors.ENDC)
    print("total FN: " + str(FN))
    print("total FP: " + str(FP))
    print("total TP: " + str(TP))

p = clusterParams()
a_clusteroutput(p.a_1, p.a_2, p.a_3, p.a_4, p.a_5, p.a_6, p.repre005_path_folder, p.cloutput_nf,
                p.imgpath)
automatictroubleshoot(p.bestMAPconf, p.cf_0001_path_folder, p.cloutput_nf, p.gtpath, p.imgpath)
automaticstatistics_maker(p.cf_0001_path_folder, p.bestMAPconf, p.gtpath)

# 4th Dec 2022
# draft: manager_2 is the testing part
# todo: dynamic naming for after clusteroutput [DONE]
# todo: writing to YOLOtxt file to be tested with mAP module not available yet
# todo: cancel the need of gtpath other than to test mAP
# todo: when running test need to create an instruction history file to remember
#  which parameters were used (exp: weights etc etc..., saved in test run test(n) folders)
# todo: bestMAPconf, IOU needs to be read from training folder as well
# todo: do something about the make 5 feature "feature" in automatic troubleshoot
# todo: writing text file have also been commented out in automatic troubleshoot
# todo: visualization: take care of which folder will store it


# to think:
# how do I make sure the output is correct?
# run it again and run this to see if test results are similar?
# OK, then iteratively organize it to make sure it is similar

# Life
# morning: freedom for a day!
#