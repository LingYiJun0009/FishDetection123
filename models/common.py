# This file contains modules common to various models

import math

import numpy as np
import requests
import torch
import torch.nn as nn
from PIL import Image, ImageDraw

from utils.datasets import letterbox
from utils.general import non_max_suppression, make_divisible, scale_coords, xyxy2xywh
from utils.plots import color_list
from matplotlib import pyplot as plt

def bloop4(array,layername,evalstats): # 1st Dec just tryna troubleshoot eval()

  array = array.detach().cpu().numpy()  
  np.save('{layername}{evalstats}evalvalues.npy'.format(layername = layername, evalstats = evalstats ),array)

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


def DWConv(c1, c2, k=1, s=1, act=True):
    # Depthwise convolution
    return Conv(c1, c2, k, s, g=math.gcd(c1, c2), act=act)

#class Conv(nn.Module):      # Conv edit to use current batch statistics to solve eval() problem 5th Dec JEdit for UperNET
#    # Standard convolution 
#    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
#        super(Conv, self).__init__()
#        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
#        self.bn = nn.BatchNorm2d(c2)
#        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
#        self.flatten = nn.Flatten(start_dim=2)
#
#    def forward(self, x):
#        
#        issitfocus = x.shape[1]
#        x = self.conv(x)
#        
#        if self.training == False:
#          state_dict_0 = self.bn.state_dict()
##          print(state_dict_0['running_mean'][0])
##          print(state_dict_0['running_var'][0])
#          flatten = self.flatten(x)
#          mean_init = torch.mean(flatten, dim = 2)[0]
#          var_init = torch.var(flatten, dim = 2)[0]
#          #print("thisvariance?[0]: " + str(np.var(x[0][0].detach().cpu().numpy())))
#          #print("thismean?[0]: " + str(np.mean(x[0][0].detach().cpu().numpy())))
# 
#          state_dict_0['running_mean'].copy_(mean_init)
#          state_dict_0['running_var'].copy_(var_init)
#        
#        x = self.bn(x)
#        x = self.act(x)
#        return x       
#
#    def fuseforward(self, x):
#        return self.act(self.conv(x))

class Conv(nn.Module):      # original Conv 1st Dec
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):

        return self.act(self.bn(self.conv(x)))
        

    def fuseforward(self, x):
        return self.act(self.conv(x))
        
#class Conv(nn.Module): # 1st Dec eval troubleshoot conv 
#    # Standard convolution
#    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
#        super(Conv, self).__init__()
#        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
#        self.bn = nn.BatchNorm2d(c2)
#        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
#
#    def forward(self, x):       
#        
##        evalstat = 'with'
#        #issitfocus = x.shape[3]  # 2nd Dec 2021 JEdits (better judge of layer no. is no. of channel)
##        issitfocus = x.shape[1]
#        #print(x.shape)
#        #After knowing I saved the wrong array (2nd time Conv instead of 1st time conv)
#        #if issitfocus == 12:
#          #bloop4(x, "FOCConvbeforeConv ", evalstat)
#        #if issitfocus == 272:
#          #bloop4(x, "CatbeforeFoc4", evalstat)
#          #bloop4(x, "SameornotBConv", evalstat)
#        x = self.conv(x)       
##        if issitfocus == 12:
##          print("After Conv: " + str(x[0][0][0][0]))
##          print("thisvariance?[0]: " + str(np.var(x[0][0].detach().cpu().numpy())))
##          print("thismean?[0]: " + str(np.mean(x[0][0].detach().cpu().numpy())))
##          #bloop4(x, "FOCConvafterConv ", evalstat)
##          #BN0 = torch.nn.BatchNorm2d(64, affine=True)
##          print("")
##          weights, bias = self.bn.parameters()
##          weight_init = torch.ones(48)
##          bias_init = torch.zeros(48)
##          mean_init = torch.rand(48) * 100
##          var_init = torch.rand(48) * 100
#          
#          state_dict_0 = self.bn.state_dict()
##          state_dict_0['weight'].copy_(weight_init)
##          state_dict_0['bias'].copy_(bias_init)
##          state_dict_0['running_mean'].copy_(mean_init)
##          state_dict_0['running_var'].copy_(var_init)
#          
##          state_dict_0['weight'].copy_(weights)
##          state_dict_0['bias'].copy_(bias)
##          state_dict_0['running_mean'].copy_(self.bn.running_mean)
##          state_dict_0['running_var'].copy_(self.bn.running_var)
##
##          print("weights[0]: " + str(weights[0]))
##          print("statedicweights: " + str(state_dict_0['weight'][0]))
##          print("bias[0]: " + str(bias[0]))
##          print("statedictbias: " + str(state_dict_0['bias'][0]))
##          print("mean[0]: " + str(self.bn.running_mean[0]))
##          print("statedictmean: " + str(state_dict_0['running_mean'][0]))
##          print(self.bn.running_mean[0])
##          print("variance[0]: " + str(self.bn.running_var[0]))
##          print("statedictvar: " + str(state_dict_0['running_var'][0]))
##          print(self.bn.track_running_stats)
#
##          bloop4(weights, "BNWeights", evalstat)
##          bloop4(bias,"BNBias", evalstat)
##          bloop4(self.bn.running_mean,"BNMean", evalstat)
##          bloop4(self.bn.running_var,"BNVar", evalstat)
#        
##        if issitfocus == 512:
##          bloop4(x, "NOBNfocusConv4", evalstat)
#        
#        x = self.bn(x)
##        if issitfocus==512:
##          bloop4(x, "NOBNfocusBN", evalstat)
#
#
#        
##        if issitfocus == 12:
##          #bloop4(x, "FOCConvafterBN ", evalstat)
##          print("")
##          print("")
##          print("After BN: " + str(x[0][0][0][0]))
#          
#
#          
#        x = self.act(x)
##        if issitfocus==512:
##          bloop4(x, "NOBNfocusAct", evalstat)
#        #if issitfocus == 12:
#          #bloop4(x, "FOCConvafterACT ", evalstat)
#        
#        return x
#        #return self.act(self.bn(self.conv(x)))
#        
#    def fuseforward(self, x):
#        return self.act(self.conv(x))

class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSP, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))

#----------------------------------------------------------------------------------Original C3 module--------------------------------------------
#class C3(nn.Module):
#    # CSP Bottleneck with 3 convolutions
#    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
#        super(C3, self).__init__()
#        c_ = int(c2 * e)  # hidden channels
#        self.cv1 = Conv(c1, c_, 1, 1)
#        self.cv2 = Conv(c1, c_, 1, 1)
#        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
#        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
#        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])
#
#    def forward(self, x):
#        
#        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1)) 
        
class C3(nn.Module):             #-----------------------------------------------------JEdit C3 module------------------------------------------------
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(C3, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x):
        
#        blah = self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1)) 
#        print("THIS IS A C3 MODULE")
#        print(blah.shape)
#        return blah
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))        

        



#class SPP(nn.Module):                                                                                #JunEdit_BLURPOOLSPP_V1<<<<----------------------
#    # Spatial pyramid pooling layer used in YOLOv3-SPP
#    def __init__(self, c1, c2, k=(5, 9, 13)):
#        super(SPP, self).__init__()
#        c_ = c1 // 2  # hidden channels
#        self.cv1 = Conv(c1, c_, 1, 1)
#        self.mp = nn.MaxPool2d(kernel_size=3, stride=1, padding = 3//2)                              #JunEdit here -------------------------
#        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1) 
#        #print("this is where c2 and c_... is")
#        #print(c_ * (len(k) + 1))
#        #print(c2)   
#        #print(c_)    
#        self.m = nn.ModuleList([BlurPool(channels = c_, filt_size=x, stride=1, pad_off=x // 2) for x in k])
#        #self.n = nn.ModuleList()                                                                              #JunEdit here-----------------
#
#    def forward(self, x):  
#        #print("COME HERE?")                                                                    
#        x = self.cv1(x)
#        #print("comhere 1")
#        #print(x.shape) 
#        x = self.mp(x)   
#        #print("comhere 2")
#        #print(x.shape)                                                              
#        #return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))                                  #JunEdit here------------------          
#        output = self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))                                 #JEdit--------------------
#        #print(output.shape)                                                                           #JEdit--------------------
#        return output
 
 
 
class SPP(nn.Module):                                                                                  #ORIGINAL SPP<<<<<<----------------------------
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))
        
#class SPP(nn.Module):                                                                                  #BlurPool SPP_V2<<<<<<--------------------------
#    # Spatial pyramid pooling layer used in YOLOv3-SPP
#    def __init__(self, c1, c2, k=(5, 9, 13)):
#        super(SPP, self).__init__()
#        c_ = c1 // 2  # hidden channels
#        self.cv1 = Conv(c1, c_, 1, 1)
#        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
#        self.b = nn.ModuleList([BlurPool(channels = c_, filt_size=x, stride=1, pad_off=x // 2) for x in k])
#        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
#        self.k = k
#
#    def forward(self, x):
#        x = self.cv1(x)
#        #return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))
#        x = self.cv2(torch.cat([x] + [self.b[i](self.m[i](x)) for i in range(len(self.k))], 1))
#        return x
        
               
#class SPP(nn.Module):
#    # Spatial pyramid pooling layer used in YOLOv3-SPP
#    def __init__(self, c1, c2, k=(5, 9, 13)):
#        super(SPP, self).__init__()
#        c_ = c1 // 2  # hidden channels
#        self.cv1 = Conv(c1, c_, 1, 1)
#        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
#        
#        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
#        #self.n = nn.ModuleList()                                                                              #JunEdit here-----------------
#
#    def forward(self, x):   
#        #sprint("original")
#        #print(x.shape)                                                                   
#        x = self.cv1(x)  
#        #print(" x self.cv1(x)")
#        #print(x.shape)                                                                
#        #return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))                                  #JunEdit here------------------          
#        output = self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))   
#        print("OUTPUT")                              #JEdit--------------------
#        print(output.shape)                                                                           #JEdit--------------------
#        return output
        
#class SPP(nn.Module):
    ## Spatial pyramid pooling layer used in YOLOv3-SPP
    #def __init__(self, c1, c2, k=(5, 9, 13)):
        #super(SPP, self).__init__()
        #c_ = c1 // 2  # hidden channels
        #self.cv1 = Conv(c1, c_, 1, 1)
        #self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        
        #self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        ##self.n = nn.ModuleList()                                                                              #JunEdit here-----------------

    #def forward(self, x):                                                                      
        #x = self.cv1(x)                                                                  
        ##return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))                                  #JunEdit here------------------          
        #output = self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))                                 #JEdit--------------------
        #print(output.shape)                                                                           #JEdit--------------------
        #return output
        
class BlurPool(nn.Module):
    def __init__(self, channels, pad_type='zero', filt_size=2, stride=2, pad_off=0):
        super(BlurPool, self).__init__()
        self.filt_size = filt_size
        self.pad_sizes = pad_off
        self.stride = stride
        self.off = int((self.stride-1)/2.)
        self.channels = channels

        if(self.filt_size==1):
            a = np.array([1.,])
        elif(self.filt_size==2):
            a = np.array([1., 1.])
        elif(self.filt_size==3):
            a = np.array([1., 2., 1.])
        elif(self.filt_size==4):    
            a = np.array([1., 3., 3., 1.])
        elif(self.filt_size==5):    
            a = np.array([1., 4., 6., 4., 1.])
        elif(self.filt_size==6):    
            a = np.array([1., 5., 10., 10., 5., 1.])
        elif(self.filt_size==7):    
            a = np.array([1., 6., 15., 20., 15., 6., 1.])
        elif(self.filt_size==9):
            a = np.array([1., 8., 28., 56., 70., 56., 28., 8., 1.])
        elif(self.filt_size==13):
            a = np.array([1., 12., 66., 220., 495., 792., 924., 792., 495., 220., 66., 12., 1.])
            

        filt = torch.Tensor(a[:,None]*a[None,:])
        filt = filt/torch.sum(filt)
        self.register_buffer('filt', filt[None,None,:,:].repeat((self.channels,1,1,1)))

        self.pad = get_pad_layer(pad_type)(self.pad_sizes)
        #print("pad size")
        #print(self.pad_sizes)
        #print("supposed pad size")
        #print(5//2)
        #print(9//2)
        #print(13//2)

    def forward(self, inp):
        if(self.filt_size==1):
            if(self.pad_off==0):
                return inp[:,:,::self.stride,::self.stride]    
            else:
                return self.pad(inp)[:,:,::self.stride,::self.stride]
        else:
            return nn.functional.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])

def get_pad_layer(pad_type):
    if(pad_type in ['refl','reflect']):
        PadLayer = nn.ReflectionPad2d
    elif(pad_type in ['repl','replicate']):
        PadLayer = nn.ReplicationPad2d
    elif(pad_type=='zero'):
        PadLayer = nn.ZeroPad2d
    else:
        print('Pad type [%s] not recognized'%pad_type)
    return PadLayer

class Focus(nn.Module):   # original Focus 1st Dec 2021
     #Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus, self).__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)
         #self.contract = Contract(gain=2)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        x = self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))   
#        print("X SHAPE--------------------")
#        print(x.shape)
        for i in range(10):     # feature map visualization Jan 2022
          bloop = x[0][i].detach().cpu().numpy()
          plt.imsave('visualization/featuremapvis_repre/firstlayer_nomosaic/detect_1440_{channel}.png'.format(channel = i), bloop)
           
        return x   
      
    
class DilateFocus(nn.Module):   # dilated Focus for representation training 13th January 2021
     #Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(DilateFocus, self).__init__()
        #self.conv = Conv(c1 * 4, c2, k, s=1, padding = 2, dilation = 2, g, act)
        self.conv = nn.Conv2d(c1*4, c2, k, s, padding = 15, dilation = 15, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        #self.contract = Contract(gain=2)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
           
        return self.act(self.bn(self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))))
         #return self.conv(self.contract(x))

#class Focus(nn.Module):   # edited but forgot reason
#    # Focus wh information into c-space
#    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
#        super(Focus, self).__init__()
#        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)
#        # self.contract = Contract(gain=2)
#
#    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
#           
#        #bloop4(x,"FocusBeforeCAT","without")   
#        x = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
#        #bloop4(x,"FocusAfterCAT","without") 
#        #bloop4(x, "SameornotBConvatFocus", "without")
#        #print(x.shape)
#        x = self.conv(x)
#        #bloop4(x,"FocusAfterConv","with")
#        #bloop4(x,"FocusAfterConvv2","without")
#        return x
#        
#        #return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))
#  
#        # return self.conv(self.contract(x))


class Contract(nn.Module):
    # Contract width-height into channels, i.e. x(1,64,80,80) to x(1,256,40,40)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        N, C, H, W = x.size()  # assert (H / s == 0) and (W / s == 0), 'Indivisible gain'
        s = self.gain
        x = x.view(N, C, H // s, s, W // s, s)  # x(1,64,40,2,40,2)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # x(1,2,2,64,40,40)
        return x.view(N, C * s * s, H // s, W // s)  # x(1,256,40,40)


class Expand(nn.Module):
    # Expand channels into width-height, i.e. x(1,64,80,80) to x(1,16,160,160)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        N, C, H, W = x.size()  # assert C / s ** 2 == 0, 'Indivisible gain'
        s = self.gain
        x = x.view(N, s, s, C // s ** 2, H, W)  # x(1,2,2,16,80,80)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # x(1,16,80,2,80,2)
        return x.view(N, C // s ** 2, H * s, W * s)  # x(1,16,160,160)


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)
        
class ConcatvWeights(nn.Module):                                                       #Jun edit (Copy the whole Concat function & call it ConcatPixel)
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super(ConcatvWeights, self).__init__()
        self.d = dimension
        self.conv = nn.Conv1d(1, 1, 2, padding=1, stride = 2)        
        self.relu = nn.LeakyReLU(0.1)
        self.hardsig = nn.Hardsigmoid()

    def forward(self, x):
        xmean = []
        for i in range(len(x)):
          xmean.append(x[i].mean())
        xmean =  torch.tensor(xmean).reshape(1,1,1,2)
        if x[i].is_cuda:
          xmean = xmean.cuda()
        xconv = self.conv(xmean[0])
        weight = self.conv.weight.data.cpu().numpy()
        
        #print("weight")
        #print(weight)
        xrelu = self.relu(xconv)
        xweight = self.hardsig(xrelu)
        #print(xweight.shape)
        #print(xweight)
        for i in range(len(x)):
          x[i] = torch.multiply(x[i],xweight[0][0][i])
          
#        print("")
#        print("COMMON CONCAT V WEIGHTS SIZE")
#        print(x[0].size())
#        print(x[1].size())
#        print(self.d)
#        #print(x.size)
#        #print(self.d.size)
#        print("")
        
        return torch.cat(x, self.d)   
            

class ConcatPixel(nn.Module):                                                          #Jun edit (Copy the whole Concat function & call it ConcatPixel)
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super(ConcatPixel, self).__init__()
        self.d = dimension

    def forward(self, x):
        
        return torch.cat(x, self.d)

class NMS(nn.Module):
    # Non-Maximum Suppression (NMS) module
    conf = 0.25  # confidence threshold
    iou = 0.45  # IoU threshold
    classes = None  # (optional list) filter by class

    def __init__(self):
        super(NMS, self).__init__()

    def forward(self, x):
        return non_max_suppression(x[0], conf_thres=self.conf, iou_thres=self.iou, classes=self.classes)


class autoShape(nn.Module):
    # input-robust model wrapper for passing cv2/np/PIL/torch inputs. Includes preprocessing, inference and NMS
    img_size = 640  # inference size (pixels)
    conf = 0.25  # NMS confidence threshold
    iou = 0.45  # NMS IoU threshold
    classes = None  # (optional list) filter by class

    def __init__(self, model):
        super(autoShape, self).__init__()
        self.model = model.eval()

    def autoshape(self):
        print('autoShape already enabled, skipping... ')  # model already converted to model.autoshape()
        return self

    def forward(self, imgs, size=640, augment=False, profile=False):
        # Inference from various sources. For height=720, width=1280, RGB images example inputs are:
        #   filename:   imgs = 'data/samples/zidane.jpg'
        #   URI:             = 'https://github.com/ultralytics/yolov5/releases/download/v1.0/zidane.jpg'
        #   OpenCV:          = cv2.imread('image.jpg')[:,:,::-1]  # HWC BGR to RGB x(720,1280,3)
        #   PIL:             = Image.open('image.jpg')  # HWC x(720,1280,3)
        #   numpy:           = np.zeros((720,1280,3))  # HWC
        #   torch:           = torch.zeros(16,3,720,1280)  # BCHW
        #   multiple:        = [Image.open('image1.jpg'), Image.open('image2.jpg'), ...]  # list of images

        p = next(self.model.parameters())  # for device and type
        if isinstance(imgs, torch.Tensor):  # torch
            return self.model(imgs.to(p.device).type_as(p), augment, profile)  # inference

        # Pre-process
        n, imgs = (len(imgs), imgs) if isinstance(imgs, list) else (1, [imgs])  # number of images, list of images
        shape0, shape1 = [], []  # image and inference shapes
        for i, im in enumerate(imgs):
            if isinstance(im, str):  # filename or uri
                im = Image.open(requests.get(im, stream=True).raw if im.startswith('http') else im)  # open
            im = np.array(im)  # to numpy
            if im.shape[0] < 5:  # image in CHW
                im = im.transpose((1, 2, 0))  # reverse dataloader .transpose(2, 0, 1)
            im = im[:, :, :3] if im.ndim == 3 else np.tile(im[:, :, None], 3)  # enforce 3ch input
            s = im.shape[:2]  # HWC
            shape0.append(s)  # image shape
            g = (size / max(s))  # gain
            shape1.append([y * g for y in s])
            imgs[i] = im  # update
        shape1 = [make_divisible(x, int(self.stride.max())) for x in np.stack(shape1, 0).max(0)]  # inference shape
        x = [letterbox(im, new_shape=shape1, auto=False)[0] for im in imgs]  # pad
        x = np.stack(x, 0) if n > 1 else x[0][None]  # stack
        x = np.ascontiguousarray(x.transpose((0, 3, 1, 2)))  # BHWC to BCHW
        x = torch.from_numpy(x).to(p.device).type_as(p) / 255.  # uint8 to fp16/32

        # Inference
        with torch.no_grad():
            y = self.model(x, augment, profile)[0]  # forward
        y = non_max_suppression(y, conf_thres=self.conf, iou_thres=self.iou, classes=self.classes)  # NMS

        # Post-process
        for i in range(n):
            scale_coords(shape1, y[i][:, :4], shape0[i])

        return Detections(imgs, y, self.names)


class Detections:
    # detections class for YOLOv5 inference results
    def __init__(self, imgs, pred, names=None):
        super(Detections, self).__init__()
        d = pred[0].device  # device
        gn = [torch.tensor([*[im.shape[i] for i in [1, 0, 1, 0]], 1., 1.], device=d) for im in imgs]  # normalizations
        self.imgs = imgs  # list of images as numpy arrays
        self.pred = pred  # list of tensors pred[0] = (xyxy, conf, cls)
        self.names = names  # class names
        self.xyxy = pred  # xyxy pixels
        self.xywh = [xyxy2xywh(x) for x in pred]  # xywh pixels
        self.xyxyn = [x / g for x, g in zip(self.xyxy, gn)]  # xyxy normalized
        self.xywhn = [x / g for x, g in zip(self.xywh, gn)]  # xywh normalized
        self.n = len(self.pred)

    def display(self, pprint=False, show=False, save=False, render=False):
        colors = color_list()
        for i, (img, pred) in enumerate(zip(self.imgs, self.pred)):
            str = f'image {i + 1}/{len(self.pred)}: {img.shape[0]}x{img.shape[1]} '
            if pred is not None:
                for c in pred[:, -1].unique():
                    n = (pred[:, -1] == c).sum()  # detections per class
                    str += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                if show or save or render:
                    img = Image.fromarray(img.astype(np.uint8)) if isinstance(img, np.ndarray) else img  # from np
                    for *box, conf, cls in pred:  # xyxy, confidence, class
                        # str += '%s %.2f, ' % (names[int(cls)], conf)  # label
                        ImageDraw.Draw(img).rectangle(box, width=4, outline=colors[int(cls) % 10])  # plot
            if pprint:
                print(str.rstrip(', '))
            if show:
                img.show(f'image {i}')  # show
            if save:
                f = f'results{i}.jpg'
                img.save(f)  # save
                print(f"{'Saving' * (i == 0)} {f},", end='' if i < self.n - 1 else ' done.\n')
            if render:
                self.imgs[i] = np.asarray(img)

    def print(self):
        self.display(pprint=True)  # print results

    def show(self):
        self.display(show=True)  # show results

    def save(self):
        self.display(save=True)  # save results

    def render(self):
        self.display(render=True)  # render results
        return self.imgs

    def __len__(self):
        return self.n

    def tolist(self):
        # return a list of Detections objects, i.e. 'for result in results.tolist():'
        x = [Detections([self.imgs[i]], [self.pred[i]], self.names) for i in range(self.n)]
        for d in x:
            for k in ['imgs', 'pred', 'xyxy', 'xyxyn', 'xywh', 'xywhn']:
                setattr(d, k, getattr(d, k)[0])  # pop out of list
        return x


class Classify(nn.Module):
    # Classification head, i.e. x(b,c1,20,20) to x(b,c2)
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Classify, self).__init__()
        self.aap = nn.AdaptiveAvgPool2d(1)  # to x(b,c1,1,1)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g)  # to x(b,c2,1,1)
        self.flat = nn.Flatten()

    def forward(self, x):
        z = torch.cat([self.aap(y) for y in (x if isinstance(x, list) else [x])], 1)  # cat if list
        return self.flat(self.conv(z))  # flatten to x(b,c2)
