#yolo somewhat the original with attempt to return inference result (but these have been commentedd out so it's all cool)
import argparse
import logging
import sys
from copy import deepcopy
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

sys.path.append('./')  # to run '$ python *.py' files in subdirectories
logger = logging.getLogger(__name__)

from models.common import *
from models.experimental import *
from utils.autoanchor import check_anchor_order
from utils.general import make_divisible, check_file, set_logging
from utils.torch_utils import time_synchronized, fuse_conv_and_bn, model_info, scale_img, initialize_weights, \
    select_device, copy_attr

try:
    import thop  # for FLOPS computation
except ImportError:
    thop = None


class Detect(nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx export

    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
        super(Detect, self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        #-------------------------------------------------------------------JEdit starts-------------------------------
        #self.up = nn.Upsample(scale_factor=2, mode='nearest')
        #self.down = nn.Upsample(scale_factor=0.5, mode='nearest')
        #self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        #self.down2 = nn.Upsample(scale_factor=0.5, mode='nearest')
        #self.conv = nn.Conv2d(1, 1, 1)        
        #self.relu = nn.LeakyReLU(0.1)
        #self.hardsig = nn.Hardsigmoid()
        #------------------------------------------------------------------JEdit ends------------------------------------




    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        
        print("x and SHAPE")
 
        print(x[0].shape)
        print(x[1].shape)
        print(x[2].shape)
        #print(x[0][0])
        
        
        #for hum in range(10):
          #plt.imsave('feature_mapVisualization/2ndlvloriTemp_{channel}.png'.format(channel = hum), x[1][0][hum].detach().cpu().numpy())
          
        #xmean = []
        #for i in range(len(x)):
          #xmean.append(x[i].mean())
          

        #xmean =  torch.tensor(xmean).reshape(1,1,1,3)    #after this line it is in cuda
        
        #if x[i].is_cuda:
          #xmean = xmean.cuda()

#        print("")
#        print("XMEAN")
#        print(xmean)
        #print("problem here?")
        #print(xmean.is_cuda)
        #xsum = xsum.reshape(1,1,1,3)        
        #xconv = self.conv(xmean)
#        print("convweight")
        #weight = self.conv.weight.data.cpu().numpy()
#        print(weight)       
#        print("XCONV")
#        print(xconv)       
        #xrelu = self.relu(xconv)
#        print("XRELU")
#        print(xrelu)
        #xweight = self.hardsig(xrelu)
#        print("XWEIGHT")
#        print(xweight)
        
        #for i in range(len(x)):
#          print("X[{}].SHAPE".format(i))
#          print(x[i].shape)
#          print("XWEIGHT[{}].SHAPE".format(i))
#          print(xweight[0][0][0][i])
          #x[i] = torch.multiply(x[i],xweight[0][0][0][i])
          #print(xweight[0][0][0][i])
          #print("final")
          #print(x[i].shape)
        
        
          
          
        
        
        
        
#------------------------------------------------------ FAILEDJEdit Pi L attempt--------        
#        x[0] = self.down(x[0])
#        x[2] = self.up(x[2])
##        print(x[0].shape)
##        print(x[1].shape)
##        print(x[2].shape)
#        
#        reshape = torch.cat(x, dim = 1)
#        reshape2 = torch.transpose(reshape, 1,2)
#        print("conv")
#        #print(ch)
#        print(reshape2.shape)
#        print(reshape2.shape[1])
#        reshape3 = self.conv(reshape2, reshape)
#        #print(reshape3.shape)
#        relued = self.relu(reshape3)
#        #print(relued.shape)
#        hSig = self.hardsig(relued)
#        #print(hSig.shape)
#        #reshaping back on how it should be
#        reshape5 = torch.transpose(hSig, 1,2)
#        #print(reshape5.shape)
#        x = torch.split(reshape5, [192, 384, 768], dim = 1)
#        x = list(x)
#        
#        print(len(x))
#        print(x[0].shape)
#        print(x[1].shape)
#        print(x[2].shape)
#        x[0] = self.up2(x[0])
#        x[2] = self.down2(x[2])
#        print(x[0].shape)
#        print(x[1].shape)
#        print(x[2].shape)



#----------------------------------------------------- FAILED JEdit Pi L attempt ends
        
#        print(x[0].shape)
#        print(x[1].shape)
#        print(x[2].shape)

        
#featuremapVisualization (AFTER PI_L)---------------------------------------------------------------------------------------Jedit
#        print(x[0].shape)
#        print(x[2].shape)
        #print("YOLO")
        #print(x[0][0][1].shape)
        #for hum in range(10):
          #plt = x[0][0][hum]
          #plt.imshow(x[0][0][hum].detach().cpu().numpy())
          #plt.imsave('feature_mapVisualization/2ndlvltemp_piL{channel}.png'.format(channel = hum), x[1][0][hum].detach().cpu().numpy())
          #image = x[0][0][hum].detach().cpu().numpy()
          #print(image.shape)
          #im = Image.fromarray(x[0][0][hum].detach().cpu().numpy())
          #im.save('feature_mapVisualization/temp{channel}.png'.format(channel = hum)
#featuremapVisualization---------------ENDS
        

        
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
#            print("convweight second stage {}".format(i))
#            weight = self.m[i].weight.data.cpu().numpy()
#            print(weight[0][0])
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            print("NY, NX")
            print(ny)
            print(nx)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            print("shape haih")
            print(x[i].shape)

            if not self.training:  # inference               #JEDIT 14th October 2021 ---------------------------------------this is the original btw
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy

                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh                
                z.append(y.view(bs, -1, self.no))    #-----------------------------------------------14th October 2021-------------up till here

#            if len(x) !=0:  # inference               #JEDIT 14th October 2021 ---------------------------------------this is the edited version
#                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
#                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)
#
#                self.stride = torch.tensor([ 8., 16., 32.])
#                y = x[i].sigmoid()
#                print("IS IT THE stride?")
#                print(len(stride))
#                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
#
#                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh                
#                z.append(y.view(bs, -1, self.no))    #-----------------------------------------------14th October 2021-------------edit up till here



        print("SELF.ANCHORGRID")
        print(self.anchor_grid)
        #Jedit---------------12thOct2021----------------------------------------------------------------------------------------
        #quicktest = (torch.cat(z, 1), x)
#        print("len(quicktest):")
#        print(len(quicktest))
#        print("quicktest[0].shape:")
#        print(quicktest[0].shape)
#        print("len(quicktest[1])")
#        print(len(quicktest[1]))
#        print("quicktest[1][0].shape")
#        print(quicktest[1][0].shape)
#        print("Anchor test")
#        print("nl:")
#        print(self.nl)
#        print("self.anchor_grid:")
#        print(self.anchor_grid)
#        print("TYPE anchors:")
#        print(type(self.anchors))
        #Jedit ENDS---------------12thOct2021----------------------------------------------------------------------------------------
        
        

        return x if self.training else (torch.cat(z, 1), x) #by the way this is the original
          
#        if len(z)!=0:
#          print("FINALLY CAN USE PRED!")
#          return (torch.cat(z, 1), x)


    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


class Model(nn.Module):
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None):  # model, input channels, number of classes
        super(Model, self).__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.load(f, Loader=yaml.SafeLoader)  # model dict

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:
            logger.info('Overriding model.yaml nc=%g with nc=%g' % (self.yaml['nc'], nc))
            self.yaml['nc'] = nc  # override yaml value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        # print([x.shape for x in self.forward(torch.zeros(1, ch, 64, 64))])

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            s = 256  # 2x min stride
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride
            self._initialize_biases()  # only run once
            # print('Strides: %s' % m.stride.tolist())

        # Init weights, biases
        initialize_weights(self)
        self.info()
        logger.info('')

    def forward(self, x, augment=False, profile=False):
        if augment:
            img_size = x.shape[-2:]  # height, width
            s = [1, 0.83, 0.67]  # scales
            f = [None, 3, None]  # flips (2-ud, 3-lr)
            y = []  # outputs
            for si, fi in zip(s, f):
                xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
                yi = self.forward_once(xi)[0]  # forward
                # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
                yi[..., :4] /= si  # de-scale
                if fi == 2:
                    yi[..., 1] = img_size[0] - yi[..., 1]  # de-flip ud
                elif fi == 3:
                    yi[..., 0] = img_size[1] - yi[..., 0]  # de-flip lr
                y.append(yi)
            return torch.cat(y, 1), None  # augmented inference, train
        else:
            return self.forward_once(x, profile)  # single-scale inference, train

    def forward_once(self, x, profile=False):
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

            if profile:
                o = thop.profile(m, inputs=(x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPS
                t = time_synchronized()
                for _ in range(10):
                    _ = m(x)
                dt.append((time_synchronized() - t) * 100)
                print('%10.1f%10.0f%10.1fms %-40s' % (o, m.np, dt[-1], m.type))

            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output

        if profile:
            print('%.1fms total' % sum(dt))
        return x

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            print(('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    # def _print_weights(self):
    #     for m in self.model.modules():
    #         if type(m) is Bottleneck:
    #             print('%10.3g' % (m.w.detach().sigmoid() * 2))  # shortcut weights

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        print('Fusing layers... ')
        for m in self.model.modules():
            if type(m) is Conv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.fuseforward  # update forward
        self.info()
        return self

    def nms(self, mode=True):  # add or remove NMS module
        present = type(self.model[-1]) is NMS  # last layer is NMS
        if mode and not present:
            print('Adding NMS... ')
            m = NMS()  # module
            m.f = -1  # from
            m.i = self.model[-1].i + 1  # index
            self.model.add_module(name='%s' % m.i, module=m)  # add
            self.eval()
        elif not mode and present:
            print('Removing NMS... ')
            self.model = self.model[:-1]  # remove
        return self

    def autoshape(self):  # add autoShape module
        print('Adding autoShape... ')
        m = autoShape(self)  # wrap model
        copy_attr(m, self, include=('yaml', 'nc', 'hyp', 'names', 'stride'), exclude=())  # copy attributes
        return m

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)


def parse_model(d, ch):  # model_dict, input_channels(3)
    logger.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, DWConv, MixConv2d, Focus, CrossConv, BottleneckCSP,
                 C3]:
            c1, c2 = ch[f], args[0]

            # Normal
            # if i > 0 and args[0] != no:  # channel expansion factor
            #     ex = 1.75  # exponential (default 2.0)
            #     e = math.log(c2 / ch[1]) / math.log(2)
            #     c2 = int(ch[1] * ex ** e)
            # if m != Focus:

            c2 = make_divisible(c2 * gw, 8) if c2 != no else c2

            # Experimental
            # if i > 0 and args[0] != no:  # channel expansion factor
            #     ex = 1 + gw  # exponential (default 2.0)
            #     ch1 = 32  # ch[1]
            #     e = math.log(c2 / ch1) / math.log(2)  # level 1-n
            #     c2 = int(ch1 * ex ** e)
            # if m != Focus:
            #     c2 = make_divisible(c2, 8) if c2 != no else c2

            args = [c1, c2, *args[1:]]
            if m in [BottleneckCSP, C3]:
                args.insert(2, n)
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum([ch[x if x < 0 else x + 1] for x in f])
        elif m is ConcatPixel:                                                                      #Jun add ConcatPixel
            c2 = sum((int(ch[f[0]]/4), ch[f[1]]))
            #print(f[1])
        elif m is ConcatvWeights:                                                                   #Jun add ConcatvWeights
            c2 = sum([ch[x if x < 0 else x + 1] for x in f])
        elif m is Detect:
            args.append([ch[x + 1] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is Contract:
            c2 = ch[f if f < 0 else f + 1] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f if f < 0 else f + 1] // args[0] ** 2
        else:
            c2 = ch[f if f < 0 else f + 1]

        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum([x.numel() for x in m_.parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        logger.info('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n, np, t, args))  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        #print("LAYERS LENGTH") #------------------------------------------------------------------Jedit--------------------------------
        #print(len(layers))
        
#        if i == 12:
#          print(layers[0])
#          print("rest")
#          print(layers[12])
        #------------------------------------------------------------------Jedit ENDS--------------------------------
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov5s.yaml', help='model.yaml')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
    set_logging()
    device = select_device(opt.device)

    # Create model
    model = Model(opt.cfg).to(device)
    model.train()

    # Profile
    # img = torch.rand(8 if torch.cuda.is_available() else 1, 3, 640, 640).to(device)
    # y = model(img, profile=True)

    # Tensorboard
    # from torch.utils.tensorboard import SummaryWriter
    # tb_writer = SummaryWriter()
    # print("Run 'tensorboard --logdir=models/runs' to view tensorboard at http://localhost:6006/")
    # tb_writer.add_graph(model.model, img)  # add model to tensorboard
    # tb_writer.add_image('test', img[0], dataformats='CWH')  # add model to tensorboard
