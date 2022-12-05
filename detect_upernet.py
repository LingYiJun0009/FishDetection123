import argparse
import time
import os
from pathlib import Path

import cv2
#upernet attempt_load edit JEDIT 10th November 2021

import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from numpy import random
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, non_max_suppression, apply_classifier, scale_coords, \
    xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
from mit_semseg.dataset import TestDataset
from mit_semseg.config import cfg
from mit_semseg.lib.nn import UserScatteredDataParallel, user_scattered_collate, patch_replication_callback
from mit_semseg.utils import colorEncode, find_recursive, setup_logger

#JEdit 10th November Upernet use the attempt load here...-------------------------------
        
        
        
def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    
    print("")
    print("SAVE DIR")
    print(save_dir)
    print("")

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True

    else:
        save_img = True
        #dataset = LoadImages(source, img_size=imgsz, stride=stride)
        

        
        #path, dirs, files = next(os.walk(source))
        #file_count = len(files)
#        print("")
#        print("FILE COUNT")
#        print(file_count)
#        print("path")
#        print(path)
#        print("dirs")
#        print(dirs)
#        print("files")
#        print(files)
#        print("")


        if os.path.isdir(source):
          imgs = find_recursive(source)
        else:
          imgs = [source]
        assert len(imgs), "imgs should be a path to image (.jpg) or directory."

        file_count = len(imgs)

        cfg.list_test = [{'fpath_img': x} for x in imgs] #JEdit 8th November 2021
        dataset_test = TestDataset( cfg.list_test, cfg.DATASET)

        loader_test = torch.utils.data.DataLoader( dataset_test, cfg.TRAIN.batch_size_per_gpu,  # we have modified data_parallel
        shuffle=False,  # we do not use this param collate_fn=user_scattered_collate,
        num_workers=cfg.TRAIN.workers,
        drop_last=True,
        pin_memory=True)
        
        iterator_test = iter(loader_test)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()

    #for path, img, im0s, vid_cap in dataset:
    pbar = tqdm(range(0, file_count))
    for i in pbar: 
        imgname = os.path.basename(os.path.normpath(imgs[i]))
#        print("")
#        print("IMG NAME???")
#        print(imgname)
#        print("")
        
        batchdata = next(iterator_test)
        img = batchdata['img_data'][0]
        print("")
        print("IMG LOAD SUCCESFUL?")
        print(img.shape)
        print("")

      
        img = img.to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        

        # Inference
        t1 = time_synchronized()
        
        print("img shape why so weird")
        print(img.shape)
        
        
#        if self.use_softmax:  # is True during inference
#            x = nn.functional.interpolate(
#                x, size=segSize, mode='bilinear', align_corners=False)
#            x = nn.functional.softmax(x, dim=1)
#            return x
        
        segsizex = img.shape[3]
        segsizey = img.shape[4]
        
        pred = model(img[0])

        
        pred = nn.functional.interpolate(pred, size = (segsizex, segsizey), mode = 'bilinear', align_corners = False)
        #pred = nn.functional.softmax(pred, dim=1)
        
        print("")
        print("PRED RESULTS before torch.max")
        print(pred.shape)
        print("")
        _, vispred = torch.max(pred, dim = 1)
        save_path = str(os.path.join(save_dir,str(imgname)))  # img.jpg
        vispred = vispred.detach().cpu().numpy()
        print("VISPRED")
        print(vispred.shape)
        print(np.unique(vispred))
        #break

#                         print("")
#                 print("PRED SIZE AFTER TORCH MAX")
#                 print(vispred.shape)
#                 print(imgs.shape)
#                 print(targets.shape)
##                svimgs =  torch.tensor(imgs).reshape(1,1,1,3)
##                svtargets = torch.tensor(targets).reshape(1,1,1,3)
#                 plt.imsave('visualization/dataloadervis/image.png', imgs[0][0].detach().cpu().numpy())
#                 plt.imsave('visualization/dataloadervis/labels.png', targets[0][0], cmap = 'jet')
        
        
        print("")
        print("original image path from:" + " " + str(imgs[i]))
        print("save directory at: ")
        print(save_dir)
        
        
        plt.imsave(save_path, vispred[0], cmap = 'brg')
        #txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
        
  


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)
    check_requirements()

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
