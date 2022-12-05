import numpy as np
import os
import cv2
def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

# def get_iou(pred_box, gt_box):
#     """
#     pred_box : the coordinate for predict bounding box
#     gt_box :   the coordinate for ground truth bounding box
#     return :   the iou score
#     the  left-down coordinate of  pred_box:(pred_box[0], pred_box[1])
#     the  right-up coordinate of  pred_box:(pred_box[2], pred_box[3])
#     """
#
#     # 1.get the coordinate of inters
#     pxmax, pxmin = min(pred_box[0], pred_box[2]), max(pred_box[2], pred_box[0])
#     gxmax, gxmin = min(gt_box[0], gt_box[2]), max(gt_box[0],gt_box[2])
#     pymax, pymin = min(pred_box[3],pred_box[1]), max(pred_box[3],pred_box[1])
#     gymax, gymin = min(gt_box[1], gt_box[3]), max(gt_box[3], gt_box[1])
#
#     ixmin = max(pxmax, gxmax)
#     ixmax = min(pxmin, gxmin)
#     iymin = max(pymax, gymax)
#     iymax = min(pymin, gymin)
#
#     iw = np.maximum(ixmax-ixmin+1., 0.)
#     ih = np.maximum(iymax-iymin+1., 0.)
#
#     # 2. calculate the area of inters
#     inters = iw*ih
#
#     # 3. calculate the area of union
#     uni = ((pxmax-pxmin+1.) * (pymax-pymin+1.) +
#            (gxmax - gxmin + 1.) * (gymax - gymin + 1.) -
#            inters)
#
#     # 4. calculate the overlaps between pred_box and gt_box
#     iou = inters / uni
#     return iou

def get_iou(box1, box2):
    '''Compute the intersection over union of two set of boxes, each box is [x1,y1,x2,y2].
    Args:
      box1: (tensor) bounding boxes, sized (4,).
      box2: (tensor) bounding boxes, sized (4,).
    Return:
      (tensor) iou.
    '''
    lt = np.zeros(2)
    rb = np.zeros(2)    # get inter-area left_top/right_bottom
    # fix some box inconsistency issue
    # if np.sum(box1[0:2]) > np.sum(box1[2:4]):
    #     box1 = box1[[2,3,0,1]]
    # if np.sum(box2[0:2]) > np.sum(box2[2:4]):
    #     box2 = box2[[2, 3, 0, 1]]
    box1 = np.array([min(box1[0], box1[2]), min(box1[1], box1[3]), max(box1[0], box1[2]), max(box1[1], box1[3])])
    box2 = np.array([min(box2[0], box2[2]), min(box2[1], box2[3]), max(box2[0], box2[2]), max(box2[1], box2[3])])
    for i in range(2):
        if box1[i] > box2[i]:
            lt[i] = box1[i]
        else:
            lt[i] = box2[i]
        if box1[i+2] < box2[i+2]:
            rb[i] = box1[i+2]
        else:
            rb[i] = box2[i+2]
    wh = rb-lt
    wh[wh<0] = 0    # if no overlapping
    inter = wh[0] * wh[1]
    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    iou = inter / (area1 + area2 - inter)
    return iou
def xywhn2xyxy(x, w=320, h=240, padw=0, padh=0):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw  # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh  # top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw  # bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh  # bottom right y
    return y
def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y
def getang(filelines):
    line2 = [x.strip() for x in filelines]
    line3 = [x.split(' ') for x in line2]
    line4 = np.array(line3, float)
    return line4[:, 6:7]
def getconf(filelines):
    line2 = [x.strip() for x in filelines]
    line3 = [x.split(' ') for x in line2]
    line4 = np.array(line3, float)
    return line4[:, 5:6]
def pro2nparr(filelines):  # string line into xywh
    line2 = [x.strip() for x in filelines]
    line3 = [x.split(' ') for x in line2]
    line4 = np.array(line3, float)
    return line4[:, 1:5]
def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y
def xyxy2xywhn(x, w = 320, h = 240):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)

    y[:, 0] = ((x[:, 0] + x[:, 2]) / 2)/w  # x center
    y[:, 1] = ((x[:, 1] + x[:, 3]) / 2)/h  # y center
    y[:, 2] = (x[:, 2] - x[:, 0])/w  # width
    y[:, 3] = (x[:, 3] - x[:, 1])/h # height

    return y
def flatten(t):
    return [item for sublist in t for item in sublist]
def getminmax_xy(bco):
    bco = np.array([round(i) for i in bco])
    minx = np.min(bco[::2].clip(min=0))
    maxx = np.max(bco[::2].clip(max=320))
    miny = np.min(bco[1::2].clip(min=0))
    maxy = np.max(bco[1::2].clip(max=240))

    return minx, maxx, miny, maxy
def def_ioumat(boxmat_co,boxmat_cf):   # get an iou mat

    ioumat = np.zeros(((len(boxmat_co), len(boxmat_cf))))

    for index, bco in enumerate(boxmat_co):
        for index2, bcf in enumerate(boxmat_cf):
            ioumat[index][index2] = get_iou(bco, bcf)

    return ioumat
def oneoversixsmaller(modelbox):
    minx, maxx, miny, maxy = getminmax_xy(modelbox)
    mdb = np.array([minx, miny, maxx, maxy])
    width = maxx - minx
    height = maxy - miny
    x_adj = width / 6  # the amount to adjust is 1/6 of the original width and height
    y_adj = height / 6
    array_adj = [x_adj, y_adj, -x_adj, -y_adj]  # array to add to modelbox later
    modelbox_v3 = mdb + array_adj
    return modelbox_v3, array_adj
def oneoverthreebigger(modelbox):
    minx, maxx, miny, maxy = getminmax_xy(modelbox)
    mdb = np.array([minx, miny, maxx, maxy])
    width = maxx - minx
    height = maxy - miny
    x_adj = width / 3  # the amount to adjust is 1/6 of the original width and height
    y_adj = height / 3
    array_adj = [x_adj, y_adj, -x_adj, -y_adj]  # array to add to modelbox later
    modelbox_v3 = mdb - array_adj
    return modelbox_v3, array_adj
def oneoversixbigger(modelbox):
    minx, maxx, miny, maxy = getminmax_xy(modelbox)
    mdb = np.array([minx, miny, maxx, maxy])
    width = maxx - minx
    height = maxy - miny
    x_adj = width / 6  # the amount to adjust is 1/6 of the original width and height
    y_adj = height / 6
    array_adj = [x_adj, y_adj, -x_adj, -y_adj]  # array to add to modelbox later
    modelbox_v3 = mdb - array_adj
    return modelbox_v3, array_adj
def getroi(targetbox, ac, cur_img, imgprevnext_img):
    minx, maxx, miny, maxy = getminmax_xy(targetbox)
    minx_ac, maxx_ac, miny_ac, maxy_ac = getminmax_xy(ac)
    imgcf_t_roi = cur_img[miny: maxy, minx: maxx, :]  # target imagecf roi
    imgback_roi = imgprevnext_img[miny_ac: maxy_ac, minx_ac: maxx_ac, :]

    return imgcf_t_roi, imgback_roi
def drawstuff_forcf(verify, ind, IOUarray, filename, tempimg, color):
    verify_file = os.path.join(verify, filename)
    with open(verify_file) as file:
        verlines = file.readlines()
        if verlines == []:
            verlines = ['0 0 0 0 0 0\n']

    confarray = getconf(verlines)
    ver_array = pro2nparr(verlines)
    ver_boxmat = xywhn2xyxy(ver_array)
    #tempimg = cv2.imread(os.path.join(imgpath, filename[:-4] + '.jpg'))

    # filtered ind for confarray and verboxmat
    confarray = confarray[ind]
    ver_boxmat = ver_boxmat[ind]

    # 1st September: make sure putText is within frame
    for cind, fin in enumerate(ver_boxmat):
        cv2.rectangle(tempimg, (round(fin[0]), round(fin[1])), (round(fin[2]), round(fin[3])), color, 1)

        # make sure text is not out of frame
        txtx = round(fin[0])
        txty = round(fin[1] - 3)
        if fin[0] - 5 < 0:
            txtx = round(fin[2] - 15)
        if fin[2] + 30 > tempimg.shape[0]:
            txtx = round(fin[0])
        if fin[1] - 20 < 0:
            txty = round(fin[3])
        if fin[3] + 30 > tempimg.shape[1]:
            txty = round(fin[1] - 3)
        textcoord = ((txtx), (txty))
        textcoord2 = ((txtx), (txty + 10))
        #color2 = tuple(int (i) for i in abs(np.array(color) - 200))
        color2 = (0, 0, 128)

        # putText conf
        cv2.putText(tempimg, "cf:" + str(np.round(confarray[cind][0], 3)), (textcoord), cv2.FONT_HERSHEY_DUPLEX, 0.3, color, 1)
        # putText cfcl IOU
        cv2.putText(tempimg, "IOU:" + str(np.round(IOUarray[cind], 3)), (textcoord2), cv2.FONT_HERSHEY_DUPLEX, 0.3, color2, 1)

    return tempimg
def drawstuff_forcl(verify, ind, filename, tempimg, color):
    verify_file = os.path.join(verify, filename)
    with open(verify_file) as file:
        verlines = file.readlines()
        if verlines == []:
            verlines = ['0 0 0 0 0 0\n']

    ver_array = pro2nparr(verlines)
    ver_boxmat = xywhn2xyxy(ver_array)
    #tempimg = cv2.imread(os.path.join(imgpath, filename[:-4] + '.jpg'))

    # filtered ind for confarray and verboxmat
    ver_boxmat = ver_boxmat[ind]

    # 2nd September: IOU between cf cl
    for cind, fin in enumerate(ver_boxmat):
        cv2.rectangle(tempimg, (round(fin[0]), round(fin[1])), (round(fin[2]), round(fin[3])), color, 1)


    return tempimg
def drawstuff_forgt(verify, filename, imgpath, color):
    verify_file = os.path.join(verify, filename)
    with open(verify_file) as file:
        verlines = file.readlines()
        if verlines == []:
            verlines = ['0 0 0 0 0 0\n']

    ver_array = pro2nparr(verlines)
    ver_boxmat = xywhn2xyxy(ver_array)

    tempimg = cv2.imread(os.path.join(imgpath, filename[:-4] + '.jpg'))

    for cind, fin in enumerate(ver_boxmat):
        cv2.rectangle(tempimg, (round(fin[0]), round(fin[1])), (round(fin[2]), round(fin[3])), color, 1)

    return tempimg
def remove_duplicl(a, conf, outputlogs = False):
    # sometimes a single cl may overlap with multiple cf, this function choose the highest cf for that cl
    # this function returns indices to remove for the final cf and conf and cl and IOU array (basically everything)
    # a = cl indices
    # conf to compare which to remove
    g = np.unique(a, return_counts= True)
    newarr = np.array([])

    if np.max(g[1]) > 1:
        ind = np.where(g[1] > 1)
        repeating_val = g[0][ind]
        for rv in repeating_val:
            arr = np.where(a == rv) # arr = index of repeating value for conf/IOUarray etc
            if outputlogs == True:
                print(" ")
                print("repeating value")
                print(rv)
                print("index of repeating value")
                print(arr)
                print("conf[arr]")
                print(conf[arr])
                print("argmax.conf[arr]")
                print(np.argmax(conf[arr]))
                print("index of original array to delete, so the highest value conf can be preserved")
                print(np.delete(arr, np.argmax(conf[arr])))
            temp = np.array(np.delete(arr, np.argmax(conf[arr])))
            newarr = np.concatenate((newarr, temp), axis = 0)

        # indices to remove
        toremoveind = newarr.ravel().astype('int')
        # new cl ind after removing indices
        newcl = np.delete(a, toremoveind)

        # end result: indices to remove, and a new cl indice array
        return toremoveind, newcl
    else:
        # no need to remove any ind, return original cl array
        return [], a
def make_5features(cfcl_IOU_array_FP, conf_cf0001, cf0001ind_towardori_FP, thearray_cf0001):
    """
    # can you make sure all these is from original?
    22nd November 2022

    :param cfcl_IOU_array_FP: FP IOU array itself
    :param conf_cf0001: full confidence (including high and low conf)
    :param cf0001ind_towardori_FP: (index of relevant FP towards original array)
    :param thearray_cf0001: (lines read from txt file)
    :return: 5 features concatenate array
    """

    # get the 5 features (last one is ratio) and reshape them
    lenFP = len(cfcl_IOU_array_FP)
    # (1) IOU
    FPIOUreshaped = np.reshape(cfcl_IOU_array_FP, (-1, 1))
    # (2) cfcl_IOU_array_FP is IOU
    cffromind_FP = conf_cf0001[cf0001ind_towardori_FP]
    cff_FPreshaped = np.reshape(cffromind_FP, (lenFP, 1))
    # (3 & 4) width and height
    xywh_FP = thearray_cf0001[cf0001ind_towardori_FP]
    w_FP = np.reshape(xywh_FP[:, 2], (-1, 1))
    h_FP = np.reshape(xywh_FP[:, 3], (-1, 1))
    # (5) ratio
    wh_pair_FP = np.concatenate((w_FP, h_FP), axis=1)
    ratioarr_FP = np.reshape(np.min(wh_pair_FP, axis=1) / np.max(wh_pair_FP, axis=1), (-1, 1))

    # time to concatenate all 5 features
    # IOU, conf, width, height, ratio
    five_features = np.concatenate((FPIOUreshaped, cff_FPreshaped, w_FP, h_FP, ratioarr_FP), axis=1)

    return five_features