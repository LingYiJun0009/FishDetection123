# 26th Dec 2022
# this .py file takes [RGB, Repre labels output paths, and gtpaths] and perform
    # cluster_module, with or without fusion_module, and run all possible parameters
    # and output a graph of the parameters with its F1 scores

# Its usage involves passing paths where the YOLO output labels are
# There will always be an RGB labels output, and a repre labels output
# Can't run the mAP here, so F1 scores will be used
# But in the future mAP is needed to run
# So, todo: fusion_module add a function where it will output after fusion label txt files

import argparse
import os
import time
import inspect
import shutil
from os import path
import numpy as np
import matplotlib.pyplot as plt
from cluster_fusion.utils import *
# from utils import *
import math
import cv2
from natsort import natsorted
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from tqdm import tqdm
import copy

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
class clusterParams:
    # get the latest written test run folder (in PyCharm ToGitHub Data.Datasets.Runs.Test.test_{}.format(latesttest)
    pathtorunsdata_test = "../Data/Runs/Test/"
    latesttest = natsorted(os.listdir(pathtorunsdata_test))[-1]

    # ground truth
    gtpath = "D:/f4kBIG/fishclef_2015_withval_v3/labels/f4kBIG_test_v3"  # test
    imgpath = "D:/f4kBIG/fishclef_2015_withval_v3/images/f4kBIG_test_v3"  # test

    # after YOLOv5 RGB and Repre
    # variables that's dynamically set/ changeable
    # input output txt folders for clusteroutput module
    cf_0001_path_folder = "../Data/Runs/Test/test_0/RGB_labels/labels".format(latesttest)
    repre005_path_folder = "../Data/Runs/Test/test_0/Repre_labels/labels".format(latesttest)

    # for after a_clusteroutput
    cloutput_nf = "../Data/Runs/Test/test_0/cloutput".format(latesttest)

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
    IOU_cutoff = 0.1
    conf_cutoff = 0.001

    # output folder after clusteroutput_fusion (so it can be sent to test mAP)
    fusionoutput =  "../Data/Runs/Test/{}/fusionresult".format(latesttest) # after fusion output path
    #finalresult = "../Data/Runs/Test/{}/finalresult".format(latesttest)
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
    :return: return extra TP FP produced by it if fusion_module not included, if fusion_module is included,
    then it'll be replaced by fusion_module output
    and creates cloutput folder which is populated by txt files result)
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
                    toremove.append(loind)
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
            norm_relboxmat = relboxmat - np.array([minx, miny, minx, miny])

            for rec in norm_relboxmat:
                roi[int(rec[1]): int(rec[3]), int(rec[0]): int(rec[2])] = 255
            roi = np.uint8(roi)
            num_labels, labels_im = cv2.connectedComponents(roi)
            splitclust = [[] for j in range(num_labels)]
            clustercenter = listocenter[cluster] - np.array([minx, miny])  # assign every value in cluster

            if num_labels >= 3:
                for ic, cctr in enumerate(clustercenter):  # splitclust[].append(cluster[ic])
                    splitclust[labels_im[int(cctr[1]) - 1][int(cctr[0]) - 1]].append(cluster[ic])
                for sc in splitclust:
                    if len(sc) >= 3:
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
    # txt_list = ["gt_220_frame(133).txt"]
    for file in os.listdir(filenamepath):
        if file.endswith(".txt"):
            txt_list.append(file)
    txt_list = natsorted(txt_list)  # sort the list 11th August 2022

    # make sure there is no cloutput folder during reruns, or else it'll just append to existing txt files
    if os.path.isdir(cloutput_nf) == True:
        shutil.rmtree(cloutput_nf)
        print(str(cloutput_nf) + " removed")
    print(" ")
    print("def a_clusteroutput: ")
    # pbar = tqdm(total=len(txt_list), leave = False, position=1, desc = "inner loop")
    pbar = tqdm(total=len(txt_list))

    for filename in txt_list:
        pbar.update()
        path1 = os.path.join(filenamepath, filename)

        with open(path1) as file:
            lines = file.readlines()
            img = cv2.imread(os.path.join(imgpath, filename[:-4] + ".jpg"))
            if os.path.isdir(cloutput_nf) == False:
                os.makedirs(cloutput_nf)
            with open(os.path.join(cloutput_nf, filename), 'a') as f:

                if bool(lines) == False:
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
                    continue

                # filter out boxes that are too big
                xmask = thearray[:, 2] < a_2
                thearray = thearray[xmask]
                ymask = thearray[:, 3] < a_2
                thearray = thearray[ymask]
                lines = lines[xmask]
                lines = lines[ymask]

                if lines.size == 0:
                    continue

                getangle = getang(lines)
                boxmat = xywhn2xyxy(thearray)  # get xyxy coordinates
                listocenter = centercoords(boxmat)
                distmat = distancemat(listocenter)
                listocluster = []

                # arbritrary distance 40
                for index, distline in enumerate(distmat):
                    flatlist = sum(listocluster, [])  # flatten the clusterlist so can use count
                    if flatlist.count(index) == False:
                        clustindices = np.where(distline < a_3)
                        clustindices = list(clustindices[0])
                        currcluster = clustindices

                        count = 0
                        newindices = 1

                        while type(newindices) != list:  # recursively add index until there is no point/indices that belongs to the cluster

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

                listocluster = splitcluster2(listocluster,
                                             boxmat)  # split cluster that has different ratio and size (27th March 2022)
                listocluster = splitcluster(listocluster, boxmat,
                                            listocenter)  # split cluster that does not intersect (26th March 2022)
                # (31st March 2022) remove boxes that has redundant iou
                listocluster = rmvcluster(listocluster, boxmat)

                for locind, sgcluster in enumerate(listocluster):  # listocluster indices, singlecluster
                    if len(sgcluster) >= 3:
                        relangle = getangle[sgcluster]  # check relevant angle
                        std = np.std(relangle)

                        # check if 80% of points within the cluster has similar angle value
                        sortedangle = np.argsort(relangle.flatten())  # sortedangle are indices of cluster
                        eightylen = round(a_5 * len(relangle))  # length of eighty percent
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

                            # from here we start to try and predict where the fish is according to cluster
                            # top eighty distance matrix
                            maxno = np.argmax(eighty_dm)
                            maxvalue = np.max(eighty_dm)
                            xmax = math.floor(
                                maxno / len(eighty_dm))  # listoclusters coordinate's indices for the current cluster
                            ymax = maxno % len(eighty_dm)

                            # get the coordinates of the largest distance in the current cluster
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

                            forrbboxmat = boxmat[sgcluster]
                            w = np.mean(forrbboxmat[:, 2] - forrbboxmat[:, 0])
                            h = np.mean(forrbboxmat[:, 3] - forrbboxmat[:, 1])
                            rbxtop = rbcent[0] + (w / 2)  # resultbox xytop and xybottom
                            rbxbottom = rbcent[0] - (w / 2)
                            rbytop = rbcent[1] + (h / 2)
                            rbybottom = rbcent[1] - (h / 2)

                            convert2xywhn = np.expand_dims(np.array([rbxtop, rbytop, rbxbottom, rbybottom]), axis=0)
                            towritecoords = xyxy2xywhn(convert2xywhn)
                            towriteline = "0" + " " + str(towritecoords[0][0]) + " " + str(
                                towritecoords[0][1]) + " " + str(
                                towritecoords[0][2]) + " " + str(towritecoords[0][3]) + " " + str(0.5) + "\n"
                            f.write(towriteline)
    # pbar.close()
# def a_clusteroutput_noangle(a_1, a_2, a_3, a_4, a_5, a_6, filenamepath, cloutput_nf, imgpath):
def sortthelines(towritelines):
    tosortconf = getconf(towritelines)
    confsortedind = np.argsort(-tosortconf, axis=0)  # negate an array to sort descending
    sortedwritelines = [towritelines[i] for i in confsortedind.flatten()]
    return sortedwritelines
def automatictroubleshoot(bestMAPconf, cf_0001_path_folder, cloutput_nf, gtpath, imgpath, outpath, IOU_cutoff, conf_cutoff, writefile = True ):
    # bloopy1

    """
    process output from a_clusteroutput and fuse it with low_conf YOLORGB output

    :param bestMAPconf:
    :param cf_0001_path_folder: YOLORGB output
    :param cloutput_nf: output from a_clusteroutput
    :return: afterfilterTP, afterfilterFP (AKA extraTP, extraFP)
    """

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
    # txt_list = ["gt_220_frame(13).txt", "gt_220_frame(8).txt", "gt_220_frame(9).txt",
    # "gt_220_frame(10).txt", "gt_220_frame(11).txt", "gt_220_frame(12).txt" ]
    for file in os.listdir(cf_0001_path_folder):
        if file.endswith(".txt"):
            txt_list.append(file)

    # create folder
    if os.path.isdir(outpath) == True:
        shutil.rmtree(outpath)
        print(str(outpath) + " removed")
    if os.path.isdir(outpath) == False:
        os.makedirs(outpath)
    print(" ")
    print("automatic troubleshoot: ")
    pbar = tqdm(total = len(txt_list))
    for fi, filename in enumerate(txt_list):
        pbar.update()
        with open(os.path.join(outpath, filename), 'a') as f:

            # print(bcolors.HEADER + "FILENAME: " + str(filename) + bcolors.ENDC)
            cf_0001_path = os.path.join(cf_0001_path_folder, filename)

            # sometimes, there is no cloutput file, so need to create one, that way FN and annotation calculation can continue
            if os.path.exists(os.path.join(cloutput_nf, filename)) == False:
                # create empty missing cl file
                with open(os.path.join(cloutput_nf, filename), 'a') as fcep:
                    pass

            cl_output_path = os.path.join(cloutput_nf, filename)
            gt_filepath = os.path.join(gtpath, filename)

            # process gt_lines
            with open(gt_filepath) as gt_file:
                gt_lines = gt_file.readlines()
                if len(gt_lines) == 0:
                    #print("no gt lines in " + filename)
                    continue
                gtcount += len(gt_lines)
                thearray_gt = pro2nparr(gt_lines)
                boxmat_gt = xywhn2xyxy(thearray_gt)

            # process cf_0001_lines
            with open(cf_0001_path) as cf_file:
                cf_0001_lines = cf_file.readlines()
                if len(cf_0001_lines) == 0:
                    #print("no cf 0001 lines in " + filename)

                    ''' '''
                    # 1st October: code block related to FN no. 1
                    # increment gtFNcount since there's no cf
                    gtFNcount += len(gt_lines)
                    FNsofardict[filename] = np.array([])
                    # print("")
                    # print("FN test--------------")
                    # print(len(gt_lines))
                    # print(" ")
                    ''' '''
                    continue
                thearray_cf0001 = pro2nparr(cf_0001_lines)
                boxmat_cf0001 = xywhn2xyxy(thearray_cf0001)
                conf_cf0001 = getconf(cf_0001_lines)

            # make sure cl_output_lines isn't empty
            with open(cl_output_path) as cl_file:
                cl_output_lines = cl_file.readlines()
            if len(cl_output_lines) == 0:
                for lines_up in cf_0001_lines:
                    f.write(lines_up)


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

                # if there is no low cf then below code is not needed
                if len(filtered_cf0001_ind) == 0:
                    # print("no low cf lines")
                    continue

                filteredcount += len(filtered_cf0001_ind_inv)
                filt_boxmat_cf0001 = boxmat_cf0001[filtered_cf0001_ind]  # boxmat where conf < bestMAPconf
                filt_boxmat_cf0001_inv = boxmat_cf0001[filtered_cf0001_ind_inv]  # boxmat where conf > bestMAPconf

                # [17th Dec 2022] # commented 749 -> 949 [DONE 20TH DEC 2022]
                #  cancel low cf that has big-ish overlap with high cf in general
                #  AKA: get low cf that does not overlap with high cf (np.where(np.max(IOUmat == 0)))
                #  (# this index of (low cf that overlaps with gt > IOU 0.5 [filt_boxmat_cf0001[cfind_IOU05]) that does not overlaps with a high cf
                #             overlaphi_low_ind = np.where(np.max(IOUmat_cfhicflo, axis=1) == 0)[0])

                # 18th Dec 2022
                lohicf_iou = def_ioumat(filt_boxmat_cf0001_inv, filt_boxmat_cf0001)

                # (var locfxiouhi) index of low cf without any overlap with highcf
                # to solve np.max problem
                if len(filt_boxmat_cf0001_inv) == 0: # if there is no high cf, assign ALL low cf to locfxiouhi
                    locfxiouhi = filtered_cf0001_ind # index in relation to original cf line
                if len(filt_boxmat_cf0001_inv) != 0:
                    locfxiouhi = filtered_cf0001_ind[np.where(np.max(lohicf_iou, axis=0) == 0)[0]] # index in relation to original cf line


                #  [17th Dec 2022] [DONE] [20TH DEC 2022]
                #   GET IOU overlap between (cl) and (low cf that does not overlap with high cf) at least 0.1 IOU >= 0.1
                #   remove low cf that share the same cl, higher conf is chosen() == support_box_v1
                #   there are cases where multiple cl share the same cf, just ignore it, np.unique low cf that's it

                locfcl_iouMat = def_ioumat(boxmat_cf0001[locfxiouhi], boxmat_cl) # iou betewen cl box and [locf boxes that doesn't overlap with hicf]

                # now need to get the ind of locf that overlaps with cl > 0.1, len() should be line of cf with iou > 0.1
                ioumat01 = np.where(locfcl_iouMat > IOU_cutoff) # which 2d index has > 0.1
                locfind = []
                for clind in np.unique(ioumat01[1]):
                    h = locfxiouhi[ioumat01[0][np.where(ioumat01[1] == clind)[0]]]
                    if len(h) > 1:
                        locfind.append(h[np.argmax(conf_cf0001[h])])

                    if len(h) == 1:
                        # print(filename)
                        locfind.append(h[0])

                fin_lowcf = np.unique(np.array(locfind))  # low cf ind in relation to original cf0001

                if len(fin_lowcf) == 0:
                    # print("NOTHING EXTRA COMING OUT OF HERE, therefore print as usual")
                    for lines_up in cf_0001_lines:
                        f.write(lines_up)
                    continue
                if len(fin_lowcf) != 0:
                    fin_lowcf_confcutoff = fin_lowcf[
                        np.where(conf_cf0001[fin_lowcf] > conf_cutoff)[0]]  # final lo cf ind in relation to orifinal cf0001

                    # TPFP calculation compared with gt
                    if len(fin_lowcf_confcutoff) == 0:
                        # print("NO CONF HIGHER THAN SET CONF NOTHING EXTRA COMING OUT!")
                        continue
                    if len(fin_lowcf_confcutoff) !=0:

                        # write to output file [31st Dec 2022]
                        cf_0001_lines_up = copy.deepcopy(cf_0001_lines)
                        for updi in fin_lowcf_confcutoff:
                            bl = thearray_cf0001[updi] # just a shorter variable name for thearray_cf0001[updi]
                            cf_0001_lines_up[updi] = "0 " + str(bl[0]) + " " + str(bl[1]) + " " + str(bl[2]) + " " + str(bl[3]) + " 0.9\n"
                            # print(cf_0001_lines_up[updi])

                        # sort the lines to write
                        sortedwritelines = sortthelines(cf_0001_lines_up)
                        for lines_up in sortedwritelines:
                            f.write(lines_up)

                        gtxtracf_iou = np.max(def_ioumat(boxmat_gt, boxmat_cf0001[fin_lowcf_confcutoff]),
                                              axis=0)  # gt and extra inference iou mat
                        afterfilterTP += len(np.where(gtxtracf_iou > 0.5)[0])
                        afterfilterFP += len(np.where(gtxtracf_iou < 0.5)[0])
    pbar.close()

    # bloopy1 [27th Dec 2022]


    # with open(os.path.join(cloutput_nf, filename), 'a') as f:

    # towritecoords = xyxy2xywhn(convert2xywhn)
    # towriteline = "0" + " " + str(towritecoords[0][0]) + " " + str(
    #     towritecoords[0][1]) + " " + str(
    #     towritecoords[0][2]) + " " + str(towritecoords[0][3]) + " " + str(0.5) + "\n"
    # f.write(towriteline)

    return afterfilterTP, afterfilterFP
def automaticstatistics_maker(cf_0001_path_folder, bestMAPconf, gtpath):
    """
    :param cf_0001_path_folder:
    :param bestMAPconf:
    :param gtpath:
    :return: TP, FP, FN
    """
    txt_list = []
    # txt_list = ["gt_220_frame(13).txt", "gt_220_frame(8).txt", "gt_220_frame(9).txt",
    #             "gt_220_frame(10).txt", "gt_220_frame(11).txt", "gt_220_frame(12).txt"]
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

    print(" ")
    print("automatic statistic maker: ")
    for fi, filename in tqdm(enumerate(txt_list)):
        #print(bcolors.HEADER + "FILENAME: " + str(filename) + bcolors.ENDC)
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
                # print("no cf 0001 lines in " + filename)

                ''' '''
                # 1st October: code block related to FN no. 1
                # increment gtFNcount since there's no cf
                FN += len(gt_lines)
                # print("")
                # print("FN test--------------")
                # print(len(gt_lines))
                # print(" ")
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
    return TP, FP, FN
def F1_maker(extra_TP, TP, extra_FP, FP, FN):
    """

    :param extra_TP:
    :param TP:
    :param extra_FP:
    :param FP:
    :param FN:
    :return: loss, F1, F1_extra_improvement
    """
    total_TP = extra_TP + TP
    total_FP = extra_FP + FP
    total_FN = FN - extra_TP
    precision = (total_TP) / (total_TP + total_FP)  # TP/(TP + FP)
    recall = (total_TP) / (total_TP + total_FN)  # TP/(TP + FN)

    F1 = 2 * (precision * recall) / (precision + recall)  # 2 * (Precision * Recall) / (Precision + Recall)
    original = 0.6711  # original RGB
    F1_extra_improvement = F1 - original  # improvement from original
    loss = -F1  # hyperopt is a minimization function so the smaller the F1 the better therefore the negative symbol
    return loss, F1, F1_extra_improvement
def F1_maker_clstats(clTP,clFP, TP, FP):
    clFN = (TP + FP) - clTP
    precision = clTP / (clTP + clFP)
    recall = clTP / (clTP + clFN)
    F1 = 2 * (precision * recall) / (precision + recall)
    return F1
def get_hyperoptgraphs(trials_result, figsavepath = './hyperopt_3f.jpg'):
    keys = trials_result[0].keys()
    iter = len(trials_result)
    keylen = len(keys)
    minind = np.argmin([sub['loss'] for sub in trials_result])
    plt.figure(figsize=(25, 12))
    plt.subplots_adjust(hspace=0.4)
    for i, k in enumerate(keys):
        keylistn = [subkey[k] for subkey in trials_result]
        roundup = math.ceil(keylen / 4)
        plt.subplot(roundup, 4, i + 1, label=k, aspect="auto")
        plt.title(k)
        plt.xlabel("iter")
        plt.scatter(range(iter), keylistn, s=1)
        plt.scatter(minind, keylistn[minind], s=1, c='r')

        # check folder exist, save graph and txt file
        if os.path.isdir(figsavepath) == True:
            shutil.rmtree(figsavepath)
        if os.path.isdir(figsavepath) == False:
            os.mkdir(figsavepath)
        with open(os.path.join(figsavepath, "trials_full.txt"), 'a') as f:
            for lines in trials_result:
                f.write(str(lines) + "\n")
        with open(os.path.join(figsavepath, "best.txt"), 'a') as b_f:
            b_f.write(str(trials_result[minind]) + "\n")
            b_f.write("minimum loss reacher at {} iteration \n".format(minind))
            b_f.write("best loss: " + str(np.min([sub['loss'] for sub in trials_result])))

        plt.savefig(os.path.join(figsavepath, "hyperopt.jpg"), dpi=199)
    print("figure saved to {}".format(figsavepath))
    plt.show()

# 22nd Dec 2022 to be run by ablation.py
# added return values in automatictroubleshoot and run_manager
# since hyperopt minimizes function, output of automatictroubleshoot does that too
def optimize_parameter(opt):
    trials = Trials()
    cp = clusterParams()
    cp.imgpath = opt.imgpath
    cp.cloutput_nf = opt.cloutput
    cp.cf_0001_path_folder = opt.cf0001output
    cp.repre005_path_folder = opt.repre005
    cp.gtpath = opt.gtpath
    cp.bestMAPconf = opt.bestMAPconf
    cp.fusionoutput = opt.fusion_outputpath
    ho_outpath = opt.output_path  # [27th December 2022 still not sure what to output here]
    max_eval = opt.maxeval
    def optim_function(d):
        a_clusteroutput(d['a_1'], d['a_2'], d['a_3'], d['a_4'], d['a_5'], d['a_6'],
                        cp.repre005_path_folder, cp.cloutput_nf,cp.imgpath)
        if opt.fusion_module == True:   # if fusion module is used
            print("fusion is used")
            extraTP, extraFP = automatictroubleshoot(cp.bestMAPconf, cp.cf_0001_path_folder, cp.cloutput_nf, cp.gtpath,
                                                     cp.imgpath, cp.fusionoutput, d['IOU_cutoff'], d['conf_cutoff'])
        else: # if no fusion module is needed, calculate cloutput box stats
            extraTP, extraFP, _ = automaticstatistics_maker(cp.cloutput_nf, cp.bestMAPconf, cp.gtpath)


        TP, FP, FN = automaticstatistics_maker(cp.cf_0001_path_folder, cp.bestMAPconf, cp.gtpath)
        loss, F1, F1_extraimprovement = F1_maker(extraTP, TP, extraFP, FP, FN)

        if opt.fusion_module == True:
            return {"loss": loss, "status": STATUS_OK, "F1": F1, "F1_extra improvement": F1_extraimprovement,
                    "a_1" : d['a_1'], "a_2" : d['a_2'],
                    "a_3": d['a_3'], "a_4" : d['a_4'], "a_5" : d['a_5'], "a_6" : d['a_6'],
                    "IOU_cutoff" : d['IOU_cutoff'], "conf_cutoff" : d['conf_cutoff']}
        if opt.fusion_module == False:
            return {"loss": loss, "status": STATUS_OK, "F1": F1, "F1_extra improvement": F1_extraimprovement,
                    "a_1": d['a_1'], "a_2": d['a_2'],
                    "a_3": d['a_3'], "a_4": d['a_4'], "a_5": d['a_5'], "a_6": d['a_6']}

    # space definition
    if opt.fusion_module == True:
        space = {'a_1': hp.uniform('a_1', 0, 1), 'a_2': hp.uniform('a_2', 0, 1), 'a_3': hp.uniform('a_3', 0, 200),
                 'a_4': hp.uniform('a_4', 200, 3000), 'a_5': hp.uniform('a_5', 0.1, 0.4),
                 'a_6': hp.uniform('a_6', 0, 0.4),
                 'IOU_cutoff': hp.uniform('IOU_cutoff', 0, 1), 'conf_cutoff': hp.uniform('conf_cutoff', 0, 0.5)}
    if opt.fusion_module == False:
        space = {'a_1': hp.uniform('a_1', 0, 1), 'a_2': hp.uniform('a_2', 0, 1), 'a_3': hp.uniform('a_3', 0, 200),
                 'a_4': hp.uniform('a_4', 200, 3000), 'a_5': hp.uniform('a_5', 0.1, 0.4),
                 'a_6': hp.uniform('a_6', 0, 0.4)}
    best = fmin(fn=optim_function,
                space=space,
                algo=tpe.suggest,
                max_evals= max_eval,
                trials=trials,
                show_progressbar=False)

    print(best)
    trials_result = trials.results
    get_hyperoptgraphs(trials_result, figsavepath=ho_outpath)  #save graphs and txt files

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='optimize_param.py')
    parser.add_argument('--fusion-module', default= True, action='store_true', help='add fusion module to run')
    parser.add_argument('--bestMAPconf', type=int, default=0.541,
                        help='best mAP conf from training result')
    parser.add_argument('--cloutput', type=str, default = '../Data/Runs/Test/test_0/cloutput', help = 'path to cl output')
    parser.add_argument('--cf0001output', type=str, default= '../Data/Runs/Test/test_0/RGB_labels/labels', help='path to cf0001 output')
    parser.add_argument('--gtpath', type=str, default='D:/f4kBIG/fishclef_2015_withval_v3/labels/f4kBIG_test_v3',
                        help='path to ground truth')
    parser.add_argument('--imgpath', type=str, default='D:/f4kBIG/fishclef_2015_withval_v3/images/f4kBIG_test_v3',
                        help='path to images')
    parser.add_argument('--repre005', type=str, default='../Data/Runs/Test/test_0/Repre_labels_3f/labels/',
                        help='pathwhere representation labels are stored')
    parser.add_argument('--fusion_outputpath', type=str, default="../Data/Runs/Test/test_1/finalresult",
                        help='pathwhere fusionoutput are stored to be sent to Linux mapCalc.py')
    parser.add_argument('--maxeval', type=int, default=2,
                        help='maximum number of evaluation')
    parser.add_argument('--output_path', type=str, default='../Data/Runs/Test/test_1/opt_param_3f/',
                        help='pathwhereTrialsoutput+graph is stored')

    opt = parser.parse_args()
    optimize_parameter(opt)

# p = clusterParams()
# p.cloutput_nf = "../Data/Runs/Test/test_0/cloutput"
# p.fusionoutput = "../Data/Runs/Test/test_1/fusionoutput_5f"
# extraTP, extraFP = automatictroubleshoot(p.bestMAPconf, p.cf_0001_path_folder, p.cloutput_nf, p.gtpath, p.imgpath, p.fusionoutput, p.IOU_cutoff, p.conf_cutoff)

# p.repre005_path_folder = "../Data/Runs/Test/test_0/Repre_labels_5f/labels"
# p.cloutput_nf = "../Data/Runs/Test/test_0/cloutput_5f"
# # #
# a_clusteroutput(p.a_1, p.a_2, p.a_3, p.a_4, p.a_5, p.a_6, p.repre005_path_folder, p.cloutput_nf,
#                p.imgpath)
# clTP, clFP, clFN = automaticstatistics_maker(p.cloutput_nf, 0.4, p.gtpath)
# extraTP, extraFP = automatictroubleshoot(p.bestMAPconf, p.cf_0001_path_folder, p.cloutput_nf, p.gtpath, p.imgpath, p.fusionoutput, p.IOU_cutoff, p.conf_cutoff)
# TP, FP, FN = automaticstatistics_maker(p.cf_0001_path_folder, p.bestMAPconf, p.gtpath)
# print(" ")
# print("original RGB only stats: ")
# print("TP: " + str(TP))
# print("FP: " + str(FP))
# print("FN: " + str(FN))
# print("directly after cluster_module stats: ")
# print("cl TP: " + str(clTP))
# print("cl FP: " + str(clFP))
# print("cl F1: " + str(F1_maker_clstats(clTP, clFP, TP, FP)))
# print("after fusion, extra TP FP:")
# print("extra TP: " + str(extraTP))
# print("extra FP: " + str(extraFP))
# loss, F1, F1_extra_improvement = F1_maker(extraTP, TP, extraFP, FP, FN)
# print("F1: " + str(F1))
# print("F1 extra improvement: " + str(F1_extra_improvement))




# 4th Dec 2022
# draft: manager_2 is the testing part
# todo: dynamic naming for after clusteroutput [DONE]
# todo: cancel the need of gtpath other than to test mAP [DONE]
    # todo: make sure modified automatic troubleshoot has the same stats as the old one
    # todo: (a tiny piece of code to test it at the bottom)
# todo: write labels immediately instead of going through whole code when there are no low cf nor cl labels
#  (~ln 698 [no cl ] and ln 716 [no low cf])

# todo: writing to YOLOtxt file to be tested with mAP module not available yet
# todo: when running test need to create an instruction history file to remember
#  which parameters were used (exp: weights etc etc..., saved in test run test(n) folders)
# todo: bestMAPconf, IOU needs to be read from training folder as well
# todo: do something about the make 5 feature "feature" in automatic troubleshoot
# todo: writing text file have also been commented out in automatic troubleshoot
# todo: visualization: take care of which folder will store it

