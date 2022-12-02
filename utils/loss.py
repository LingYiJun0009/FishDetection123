# Loss functions

import torch
import torch.nn as nn
import numpy as np
import math

from utils.general import bbox_iou
from utils.torch_utils import is_parallel


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super(BCEBlurWithLogitsLoss, self).__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(QFocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    # Compute losses
    def __init__(self, model, autobalance=False):
        super(ComputeLoss, self).__init__()
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))
        #BCEag = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['angleweight']], device=device)) # added angle loss  # 22nd Feb 2022
        MSEag = nn.MSELoss(reduction = 'mean') # 28th Feb 2022 change to MSE loss
        
        linear = nn.Linear

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=0.0)

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj, MSEag = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g), FocalLoss(MSEag, g)

        det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
        self.balance = {3: [3.67, 1.0, 0.43], 4: [3.78, 1.0, 0.39, 0.22], 5: [3.88, 1.0, 0.37, 0.17, 0.10]}[det.nl]
        #---------------------------------------------------------------------JeditPIlAttemptToSolveLoss---------------------------------------------
        #print("HELLO")
        #print(det.stride)
        # self.balance = [1.0] * det.nl   <--------already commented out originally
        #print(det.stride==8)
        self.ssi = (det.stride == 16).nonzero(as_tuple=False).item()  # stride 16 index --------------------Jedit 16 to 8--------------------------
        #self.ssi = 1                                                  #Jedit straight away assign self.ssi as 1--------- 
        #print("LOOK HERE SELF SSI" + str(self.ssi))
        #Jedit PIlAttempt to SolveLoss Ends------------------------------------------------------------------------------------------------------------
        self.BCEcls, self.BCEobj, self.MSEag, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, MSEag, model.gr, h, autobalance
        for k in 'na', 'nc', 'nl', 'anchors':
            setattr(self, k, getattr(det, k))

    def __call__(self, p, targets, angle, whichclass):  # predictions, targets, model
#        print("")
#        print("p and targets size--------------------------------")
#        print(len(p))
#        print(targets[0])
#        print(targets)
#        print("")
        
        device = targets.device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        if bool(angle):
          angloss = torch.zeros(1, device=device)
        # 22nd Feb 02:08 stopped here
        
        
        tcls, tbox, indices, anchors, *angle_mt  = self.build_targets(p, targets, *angle)  # targets (last variable = angle_mt represent angle match targets)

        #print("---------ALL DEM LEN---------------")
#        print("len(tbox): " + str(len(tbox)) + "\n")
#        print("len(indices): " + str(len(indices)) + "\n")
#        print("len(anchors): " + str(len(anchors)) + "\n")
        
#        print("---------ALL DEM SHAPES------------")
#        print("")
#        print("tbox[0].shape: " + str(tbox[0].shape) + "\n")
#        print("len(indices[0]): " + str(len(indices[0])) + "\n")
#        print("anchors[0].shape: " + str(anchors[0].shape) + "\n")
        
#        print("")
#        print("------------------TBOX AND ITS SHAPE-------------")
#        print(tbox[0].shape)
#        print(tbox[1].shape)
#        print(tbox[2].shape)
        
        if bool(angle):
          angle_mt = angle_mt[0]
#          print("-----------------ANGLE_MT AND ITS SHAPE----------")          
#          print(angle_mt[0].shape)
#          print(angle_mt[1].shape)
#          print(angle_mt[2].shape)
        #print("")
        #print("-----------------INDICES----------")
        #print(indices)
        #print("-----------------ANCHORS----------")
        #print(anchors)
#        print("")
#        print("---------what's p again?---------")
        #print(len(p)) 
        #print(p[0].shape)
       # print(p[0])
        
        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            if bool(angle):
              pred_angle = pi[:,:,:,:,-1]
              angdiff = torch.zeros(1)
#            print("")
#            print("pred_angle here") # torch.Size([2, 3, 20, 20, 1])
#            print(pred_angle.shape)
#            print("")
            pi = pi[:,:,:,:,0:-1]
            if type(whichclass) == int:
              clsch = 5+whichclass # pay attention to only the specified classes
              plus =pi[:,:,:,:,clsch:clsch+1]
              pi = pi[:,:,:,:,0:5]
              pi = torch.cat((pi, plus), axis = -1)
                           
#            print("")
#            print("trying to seperate angle from the rest") # torch.Size([2, 3, 20, 20, 6])
#            print(pi.shape)
#            print("")
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj
#            print(pi[...,0].shape)
#            print(b.shape)
#            print("------------b---------")
#            print(b)

            n = b.shape[0]  # number of targets
            if n:
#                print("")
#                print("GJ")
#                print(gj)
#                print("GI")
#                print(gi)
#                print("")
                
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets
                
                if bool(angle):
                  pred_angle_s = pred_angle[b,a,gj,gi] # prediction angle subset corresponding to targets
#                  print("pred_angle_s")
#                  print(pred_angle_s.shape)
#                  print("")
#                print("ps")
#                print(ps)
#                print("")
#                print("ps.shape")
#                print(ps.shape)
                
                # Regression
#                print(" ")
#                print("REGRESSION")
                #print(pi)
                
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i] # times whatever anchor that matches it

#                pxy = ps[:, :2].tanh() * 2. - 0.5
#                pwh = (ps[:, 2:4].tanh() * 2) ** 2 * anchors[i] # times whatever anchor that matches it

            
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                
                
                if bool(angle):  # 22nd Feb 2022
                   #filter out target and pred angle to not include those that is more than 90 or -90 in target angles

                  bool4ag = ((angle_mt[i] < 90.) & (angle_mt[i] > -90.)) # create boolean filter
                  tolossangle = angle_mt[i][bool4ag]
                  tolossangle = tolossangle[:,None] # to match shape of tolosspredangle
                  tolosspredangle = pred_angle_s[bool4ag]
#                  print("")
#                  print("does angle pred and angle target match?")
#                  print(tolossangle.shape)
#                  print(tolosspredangle.shape)
#                  print("")
#                  print("bool4ag[0] and shape")
#                  print(bool4ag)
#                  print(bool4ag.shape)
#                  print("")                 
                  # normalize target angles
#                  print("")
#                  print("tolossangle before normalize")
#                  print(tolossangle)
                  tolossangle =  (tolossangle+math.pi)/(math.pi*2)  # normalize 0-1
#                  print("tolossangle")
#                  print(tolossangle)
#                  print(tolossangle.shape)
#                  print("")
                   # then, tanh pred angles
                  tolosspredangle = torch.sigmoid(tolosspredangle)
                  
                  # check why my output angle looks weird # 5th March 2022
#                  print("")
#                  print("prediction angle")
#                  print(tolosspredangle[:6])
#                  print("target angle")
#                  print(tolossangle[:6])
#                  print("")
#
                  #tolosspredangle = torch.tanh(tolosspredangle)  # 28th Feb 2022
#                  print("after normalize target angles")
#                  print(tolossangle[0:3])
#                  print("")
#                  print("after tanh prediction angle")
#                  print(tolosspredangle[0:3])
#                  print("")
      
                  angdiff = torch.sum(torch.abs(tolosspredangle-tolossangle.flatten().to(device)))/len(tolosspredangle)
                  loss = self.MSEag(tolosspredangle, tolossangle.flatten().to(device)) # loss function here 
 
                  if torch.isnan(loss): # just so it won't become nan
                    loss = 0      
                                  
                  angloss += loss

                       
                           
                
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                #print("")
                #print("--------------IOU--------------")
                #print(iou)
                #print("")
                #print("--------------pbox-------------")
#                print(pbox)
#                print(pbox.shape)
                #print("")
                #print("--------------tbox-------------")
#                print(tbox[i])
#                print(tbox[i].shape)
                #print("")
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio
                
                # Classification
                if self.nc > 1 and whichclass == False:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(ps[:, 5:], t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            
             
            obji = self.BCEobj(pi[..., 4], tobj) # 328
            #print("--------------difference between pi and pi[...,4] ------------------")
#            print("pi")
#            print(pi.shape)    # torch.Size([2, 3, 20, 20, 6])
#            print(pi)
#            print("pi[...,4]")
#            print(pi[...,4].shape)      # torch.Size([2, 3, 20, 20])  # objectness score for each "pixel"
#            print(pi[...,4])
#            print(pi[...,4].max())
           # print("---------------tobj------------------")
#            print(tobj.shape)            # torch.Size([2, 3, 20, 20])
#            print(tobj)
#            print(tobj.max())
            
            lobj += obji * self.balance[i]  # obj loss
            #print("")
            #print("-----------OBJECT LOSS TROUBLESHOOT------------------")
#            print("")
#            print("obji")
#            print(obji)
#            print("")
#            print("lobj")
#            print(lobj)
#            print("")
#            print("self.balance[i]")
#            print(self.balance[i])
           # print("---------------------------------------------------")
            #print("")
            
            
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.99 + 0.0001 / obji.detach().item()
        
        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        if bool(angle):
          angloss *= self.hyp['anglegain']
        bs = tobj.shape[0]  # batch size
#        print("why can't concantenate?")
#        print(angloss)
#        print(lcls)

        if bool(angle) ==  False:
          loss = lbox + lobj + lcls
          return loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach()
        if bool(angle):
          loss = lbox + lobj + lcls + angloss
#          print("")
#          print("HOW COME 2ND ROUND ISN'T ")
#          print(loss)
#          print("")
#          print("")
#          print("angle_loss")
#          print(angloss)
#          print("")
          return loss * bs, torch.cat((lbox, lobj, lcls, loss, angloss)).detach(), angdiff
          

    def build_targets(self, p, targets, *angle):
        # ag, angle, aglist
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
#        print("")
#        print("IN BUILLD TARGETS")
#        print(targets.shape)
#        print(angle[0])
#        print(len(angle[0]))
#        print("")
        
        
        
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch, aglist = [], [], [], [], []
        gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        #print("what's ai?-----------------------------")
        #print(ai)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices
        if bool(angle):
          angletorch = torch.from_numpy(np.expand_dims(angle[0], axis = 0).repeat(3, axis = 0))

        #print("what's targets?------------------------")
        #print(targets)
        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        for i in range(self.nl):
            anchors = self.anchors[i]
            
            if bool(angle):
              ag = torch.unsqueeze(angletorch[i], dim = 0)

#            print("")
#            print("AnChoRS{}".format(i)) # 4th February 2022
#            print(anchors)
#            print("")
#            print("p[i].sHaPe")
#            print(p[i].shape)
#            print("")
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain
#            print("")
#            print("GaIN:")
#            print(gain)
#            print("")
#            print("targets")
#            print(targets)
#            print("")
            # Match targets to anchors
            
            t = targets * gain
#            print("")
#            print("just t")
#            print(t)
#            print("")
#            print("t.shape")
#            print(t.shape)
#            print("angle repeat?")
            if bool(angle):
                ag = torch.repeat_interleave(ag, 3, dim=0)
#            print(ag.shape)
#            print("")

            if nt: # if there are targets
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
#                print("")
#                print("----------t[:,:,4:6]-------------")
#                print(t[:,:,4:6])
#                print("")
#                print("---------anchors[:, None]--------")
#                print(anchors[:, None])
#                print("")
#                print("---------r-----------")
#                print(r)
#                print(r.shape)
#                print("")
                
                # all the anchors that belong to the layer and whose size ratio to the target (w and h resp.) is less than 4
                # (yolov3: only one anchor with max iou with the target across three layers (#) plus neighboring 2 pixels
                j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # compare 
#                print("")
#                print("-------torch.max(r,1./r)")  # 
#                print(torch.max(r, 1. / r))
#                print("")
#                print("-----------torch.max(r, 1. / r).max(2)")
#                print(torch.max(r, 1. / r).max(2))
#                print("")
#                print("----------------------j----------------------")
#                print(j)
#                print(j.shape)
#                print("")
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter
#                print("")
                #print("---------moving on t--------------------")
#                print(t)
#                print("t.shape")
#                print(t.shape)
#                print("")
                #print("---------moving on angle--------------------")
                if bool(angle):
                  ag = ag[j]
#                  print(ag)
#                  print(ag.shape)
        
                # Offsets
                gxy = t[:, 2:4]  # grid xy
#                print("")
                #print("--------TIME FOR OFFSETS-----------")
                #print("------gxy-------")
                #print(gxy)
#                print("")
#                print("------gain[[2,3]]------")
#                print(gain[[2,3]])                                            
                gxi = gain[[2, 3]] - gxy  # inverse
#                print("")
#                print("------------gxi------------")
#                print(gxi)
#                print("")
#                print("----------gxy%1---------------")
#                print(gxy%1)
#                print("")
                #print("----------gxy%1 < g---------------")
#                print(gxy%1  < g)
#                print("")
#                print("----------gxy >  1---------------")
#                print(gxy > 1)
#                print("")
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T    # "&" operator performs bitwise operation
#                print("---------bottom j----------------")
#                print(j)
#                print("")
#                print("-------bottom k---------------")
#                print(k)
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T    
                j = torch.stack((torch.ones_like(j), j, k, l, m))
#                print("before repeat what's j")
#                print(j)
#                print("")
#                print("before t repeat")
#                print(t)
#                print(t.shape)

                t = t.repeat((5, 1, 1))[j]
                
                if bool(angle):
                  ag = ag.repeat((5,1))[j]
#                print("after t repeat")
#                print(t)
#                print(t.shape)  
#                print(ag.shape)
#                print("after repeat angle")
#                print(ag)   
#                print("")
                
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
#                print("offset")
#                print(offsets.shape)
#                print("")
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices
#            print("GRIDXY INDICES")
#            print(gi)
#            print(gj)
#            print("-------------------t-----------------")
#            print(t)
#            print("-------------------gxy---------------")
#            print(gxy)
#            print("-------------------gij---------------")
#            print(gij)
#            print("")
#            print("------------offsets-------------")
#            print(offsets)
#            print("")

            # Append
            a = t[:, 6].long()  # anchor indices
#            print("a")
#            print(a.shape)
#            print(a)
#            print("")
#            print("-------indices-----------")
#            print((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))
#            print("")
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
#            print("-------------------tbox---------------")
#            print(torch.cat((gxy - gij, gwh), 1))
#            print("")
#            print("-----------------anchors------------")
#            print(anchors[a])
#            print("") 
#            print("")
#            print("--i haven't check the difference here")
#            print(torch.cat((gxy - gij, gwh), 1).shape)
#            print(ag.shape)
#            print(ag)
#            print("")
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class
            if bool(angle):
              aglist.append(ag)
        if bool(angle):
          #print("")
          #print(" difference between aglist and tbox")
#          print(aglist)
#          print(tbox)
#          print("")
#          print("tcls")
#          print(tcls[0].shape)
#          print("tbox")
#          print(tbox)
#          print("indices")
#          print(indices)
#          print("aglist")
#          print(aglist)
#          print("")
          return tcls, tbox, indices, anch, aglist
        else:
          return tcls, tbox, indices, anch
