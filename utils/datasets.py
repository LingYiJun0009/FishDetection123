# Dataset utils and dataloaders
# IMPORTANT note for if self.angle is activated, only fliplr has an effect on angle since default yolov5 augmentation 
# doesn't use other perspective augmentation. (ps. translation doesn't affect angle whatsoever)
# random perspective apparantly doesn't include fliplr or up down, therefore, angles are not affected


import glob
import logging
import math
import os
import random
import shutil
import time
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
from threading import Thread

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ExifTags
from torch.utils.data import Dataset
from tqdm import tqdm
import tifffile

from utils.general import xyxy2xywh, xywh2xyxy, xywhn2xyxy, clean_str
from utils.torch_utils import torch_distributed_zero_first

# Parameters
help_url = 'https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data'
img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng']  # acceptable image suffixes
vid_formats = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes
logger = logging.getLogger(__name__)

# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break


def get_hash(files):
    # Returns a single hash value of a list of files
    return sum(os.path.getsize(f) for f in files if os.path.isfile(f))


def exif_size(img):
    # Returns exif-corrected PIL size
    s = img.size  # (width, height)
    try:
        rotation = dict(img._getexif().items())[orientation]
        if rotation == 6:  # rotation 270
            s = (s[1], s[0])
        elif rotation == 8:  # rotation 90
            s = (s[1], s[0])
    except:
        pass

    return s


def create_dataloader(path, imgsz, batch_size, stride, opt, hyp=None, augment=False, cache=False, pad=0.0, rect=False,
                      rank=-1, world_size=1, workers=8, image_weights=False, quad=False, prefix='', schannel = False, angle = False, whichclass = False):
    # Make sure only the first process in DDP process the dataset first, and the following others can use the cache
    with torch_distributed_zero_first(rank):
        dataset = LoadImagesAndLabels(path, imgsz, batch_size,
                                      augment=augment,  # augment images
                                      hyp=hyp,  # augmentation hyperparameters
                                      rect=rect,  # rectangular training
                                      cache_images=cache,
                                      single_cls=opt.single_cls,
                                      stride = int(stride),
                                      pad = pad,
                                      image_weights=image_weights,
                                      prefix=prefix,
                                      schannel = schannel,
                                      angle = angle,
                                      whichclass = whichclass)
                                      
#    print("INSIDE CREATE DATALOADER DATASET.SHAPES---------------------------------")
#    print(dataset.shapes)
#    print("-------------------------------------------------------------------------")

    batch_size = min(batch_size, len(dataset))
    nw = min([os.cpu_count() // world_size, batch_size if batch_size > 1 else 0, workers])  # number of workers
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if rank != -1 else None
    loader = torch.utils.data.DataLoader if image_weights or angle else InfiniteDataLoader
    # Use torch.utils.data.DataLoader() if dataset.properties will update during training else InfiniteDataLoader()
    dataloader = loader(dataset,
                        batch_size=batch_size,
                        num_workers=nw,
                        sampler=sampler,
                        pin_memory=True,
                        collate_fn=LoadImagesAndLabels.collate_fn4 if quad else LoadImagesAndLabels.collate_fn_a if angle else LoadImagesAndLabels.collate_fn)
    
#    print("IMGSZTEST--------------------IMFSZTEST-------------------------------------") #---------------------------------------JTROUBLESHOOT
#    print(imgsz)
#    print(stride)
#    print("IMGSZENDS--------------------IMGSIZEENDS-----------------------------------")  #JTROUBLESHOOT ENDS----------------------------------

    return dataloader, dataset


class InfiniteDataLoader(torch.utils.data.dataloader.DataLoader):
    """ Dataloader that reuses workers

    Uses same syntax as vanilla DataLoader
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class LoadImages:  # for inference
    def __init__(self, path, img_size=640, stride=32, schannel=False):
        p = str(Path(path))  # os-agnostic
        p = os.path.abspath(p)  # absolute path
        if '*' in p:
            files = sorted(glob.glob(p, recursive=True))  # glob
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
        elif os.path.isfile(p):
            files = [p]  # files
        else:
            raise Exception(f'ERROR: {p} does not exist')

        images = [x for x in files if x.split('.')[-1].lower() in img_formats]
        videos = [x for x in files if x.split('.')[-1].lower() in vid_formats]
        ni, nv = len(images), len(videos)

        self.img_size = img_size
        self.stride = stride
        self.files = images + videos
        self.nf = ni + nv  # number of files
        self.video_flag = [False] * ni + [True] * nv
        self.mode = 'image'
        self.schannel = schannel
        if any(videos):
            self.new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nf > 0, f'No images or videos found in {p}. ' \
                            f'Supported formats are:\nimages: {img_formats}\nvideos: {vid_formats}'

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = 'video'
            ret_val, img0 = self.cap.read()
            if not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:  # last video
                    raise StopIteration
                else:
                    path = self.files[self.count]
                    self.new_video(path)
                    ret_val, img0 = self.cap.read()

            self.frame += 1
            print(f'video {self.count + 1}/{self.nf} ({self.frame}/{self.nframes}) {path}: ', end='')

        else:
            # Read image
            self.count += 1
            if self.schannel == False:
             img0 = cv2.imread(path)  # BGR
             assert img0 is not None, 'Image Not Found ' + path
             print(f'image {self.count}/{self.nf} {path}: ', end='')
            if self.schannel == True:
              img0 = tifffile.imread(path) # BGRBGR
              assert img0 is not None, 'Image Not Found ' + path
              print(f'image {self.count}/{self.nf} {path}: ', end='')
              

        # Padded resize
        img = letterbox(img0, self.img_size, stride=self.stride)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        return path, img, img0, self.cap

    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.nframes = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.nf  # number of files


class LoadWebcam:  # for inference
    def __init__(self, pipe='0', img_size=640, stride=32):
        self.img_size = img_size
        self.stride = stride

        if pipe.isnumeric():
            pipe = eval(pipe)  # local camera
        # pipe = 'rtsp://192.168.1.64/1'  # IP camera
        # pipe = 'rtsp://username:password@192.168.1.64/1'  # IP camera with login
        # pipe = 'http://wmccpinetop.axiscam.net/mjpg/video.mjpg'  # IP golf camera

        self.pipe = pipe
        self.cap = cv2.VideoCapture(pipe)  # video capture object
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # set buffer size

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if cv2.waitKey(1) == ord('q'):  # q to quit
            self.cap.release()
            cv2.destroyAllWindows()
            raise StopIteration

        # Read frame
        if self.pipe == 0:  # local camera
            ret_val, img0 = self.cap.read()
            img0 = cv2.flip(img0, 1)  # flip left-right
        else:  # IP camera
            n = 0
            while True:
                n += 1
                self.cap.grab()
                if n % 30 == 0:  # skip frames
                    ret_val, img0 = self.cap.retrieve()
                    if ret_val:
                        break

        # Print
        assert ret_val, f'Camera Error {self.pipe}'
        img_path = 'webcam.jpg'
        print(f'webcam {self.count}: ', end='')

        # Padded resize
        img = letterbox(img0, self.img_size, stride=self.stride)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        return img_path, img, img0, None

    def __len__(self):
        return 0


class LoadStreams:  # multiple IP or RTSP cameras
    def __init__(self, sources='streams.txt', img_size=640, stride=32):
        self.mode = 'stream'
        self.img_size = img_size
        self.stride = stride

        if os.path.isfile(sources):
            with open(sources, 'r') as f:
                sources = [x.strip() for x in f.read().strip().splitlines() if len(x.strip())]
        else:
            sources = [sources]

        n = len(sources)
        self.imgs = [None] * n
        self.sources = [clean_str(x) for x in sources]  # clean source names for later
        for i, s in enumerate(sources):
            # Start the thread to read frames from the video stream
            print(f'{i + 1}/{n}: {s}... ', end='')
            cap = cv2.VideoCapture(eval(s) if s.isnumeric() else s)
            assert cap.isOpened(), f'Failed to open {s}'
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) % 100
            _, self.imgs[i] = cap.read()  # guarantee first frame
            thread = Thread(target=self.update, args=([i, cap]), daemon=True)
            print(f' success ({w}x{h} at {fps:.2f} FPS).')
            thread.start()
        print('')  # newline

        # check for common shapes
        s = np.stack([letterbox(x, self.img_size, stride=self.stride)[0].shape for x in self.imgs], 0)  # shapes
        self.rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
        if not self.rect:
            print('WARNING: Different stream shapes detected. For optimal performance supply similarly-shaped streams.')

    def update(self, index, cap):
        # Read next stream frame in a daemon thread
        n = 0
        while cap.isOpened():
            n += 1
            # _, self.imgs[index] = cap.read()
            cap.grab()
            if n == 4:  # read every 4th frame
                _, self.imgs[index] = cap.retrieve()
                n = 0
            time.sleep(0.01)  # wait time

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        img0 = self.imgs.copy()
        if cv2.waitKey(1) == ord('q'):  # q to quit
            cv2.destroyAllWindows()
            raise StopIteration

        # Letterbox
        img = [letterbox(x, self.img_size, auto=self.rect, stride=self.stride)[0] for x in img0]

        # Stack
        img = np.stack(img, 0)

        # Convert
        img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB, to bsx3x416x416
        img = np.ascontiguousarray(img)

        return self.sources, img, img0, None

    def __len__(self):
        return 0  # 1E12 frames = 32 streams at 30 FPS for 30 years


def img2label_paths(img_paths):
    # Define label paths as a function of image paths
    sa, sb = os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep  # /images/, /labels/ substrings
    return [x.replace(sa, sb, 1).replace('.' + x.split('.')[-1], '.txt') for x in img_paths]
    
def lbl2angle_paths(lbl_paths): # 17th February try to read from angle file # need to create anglelabels folder
    return lbl_paths.replace("labels", "anglelabels")
      
class LoadImagesAndLabels(Dataset):  # for training/testing # bloopy2
    def __init__(self, path, img_size=640, batch_size=16, augment=False, hyp=None, rect=False, image_weights=False,
                 cache_images=False, single_cls=False, stride=32, pad=0.0, prefix='', schannel = False, angle = False, whichclass = False):
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp
        self.image_weights = image_weights
        self.rect = False if image_weights else rect
        self.mosaic = self.augment and not self.rect  # load 4 images at a time into a mosaic (only during training)
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.stride = stride
        self.path = path
        self.schannel = schannel
        self.angle = angle
        self.whichclass = whichclass
        
        try:
            f = []  # image files
            for p in path if isinstance(path, list) else [path]:
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    f += glob.glob(str(p / '**' / '*.*'), recursive=True)
                elif p.is_file():  # file
                    with open(p, 'r') as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [x.replace('./', parent) if x.startswith('./') else x for x in t]  # local to global path
                else:
                    raise Exception(f'{prefix}{p} does not exist')
            self.img_files = sorted([x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in img_formats]) 
            assert self.img_files, f'{prefix}No images found'
        except Exception as e:
            raise Exception(f'{prefix}Error loading data from {path}: {e}\nSee {help_url}')

        # Check cache # bloopy3
        self.label_files = img2label_paths(self.img_files)  # labels
        
#        if self.angle:
#          self.anglelabel_files = img2angle_paths(self.img_files)
          
#        print("")
#        print("ln394 label_files---------------- (15th amnesia)")
#        print(self.label_files)
#        print("")
        cache_path = Path(self.label_files[0]).parent.with_suffix('.cache')  # cached labels
        if cache_path.is_file():
            cache = torch.load(cache_path)  # load
            if cache['hash'] != get_hash(self.label_files + self.img_files) or 'results' not in cache:  # changed
                cache = self.cache_labels(cache_path, prefix)  # re-cache
        else:
            cache = self.cache_labels(cache_path, prefix)  # cache

#        print("")
#        print("cache before any popping")
#        print(cache)
#        print("")
        
        # Display cache
        [nf, nm, ne, nc, n] = cache.pop('results')  # found, missing, empty, corrupted, total
        desc = f"Scanning '{cache_path}' for images and labels... {nf} found, {nm} missing, {ne} empty, {nc} corrupted"
        tqdm(None, desc=prefix + desc, total=n, initial=n)
        assert nf > 0 or not augment, f'{prefix}No labels in {cache_path}. Can not train without labels. See {help_url}'

        # Read cache
        cache.pop('hash')  # remove hash
#        print("BEFORE ARRVIING")
#        print(self.angle)
        if self.angle == False:
#          print("")
#          print("YOU HAVE ARRIVED AT ANGLE FALSE")
#          print("")
          labels, shapes = zip(*cache.values())
          
        if self.angle:
#          print("")
#          print("YOU HAVE ARRIVED AT ANGLE TRUE")
#          print("")
          labels, shapes, anglelist = zip(*cache.values())
          self.anglelist = list(anglelist)
#          print("")
#          print("datasets.py cache")
#          print(cache)
#          print("")
#        print("")
#        print("datasets.py ln413 labels, shapes from zip(*cache.values())-------------------HALOOOOOOO") # 14th February
#        print("labels")
#        print(labels)
#        print("shapes")
#        print(shapes)
#        print("")
        self.labels = list(labels) 
          
#        print("")
#        print("list(labels-------)")
#        print(list(labels))
#        print("")
        
        self.shapes = np.array(shapes, dtype=np.float64)
#        print("please pay attention here--------------------------------------------")
#        print(self.shapes)
        # schannel = 320, 240
        # no schannel 320, 240
        self.img_files = list(cache.keys())  # update
#        print("img_files-------------------- (15th amnesia)")
#        print(self.img_files)
#        print("")
#        print("")
#        print("cache.keys()-------------------")
#        print(cache.keys())
#        print("")
#        print("list(cache.keys())---------------------")
#        print(list(cache.keys()))
#        print("")
        
        self.label_files = img2label_paths(cache.keys())  # update the order of image if im not mistaken
#        print("ln451 label_files no.2 ----------------- (15th amnesia)")  # my point is to check how they make sure the list is the same
#        print(self.label_files)
#        print("")
        if angle:    # 14th February # 17th February
          print("ANGLE ACTIVATED!!-----------")
          #self.anglelabel_files = img2angle_paths(cache.keys())   # just following the update
          

        
        if single_cls:
            for x in self.labels:
                x[:, 0] = 0
        

        n = len(shapes)  # number of images
        bi = np.floor(np.arange(n) / batch_size).astype(np.int)  # batch index
        nb = bi[-1] + 1  # number of batches
        self.batch = bi  # batch index of image
        self.n = n
        self.indices = range(n) 

        # Rectangular Training
        if self.rect:
            # Sort by aspect ratio
            s = self.shapes  # wh
#            print("TIME FOR S")    # 8TH January*************
#            print(s)
            ar = s[:, 1] / s[:, 0]  # aspect ratio
            irect = ar.argsort()
            self.img_files = [self.img_files[i] for i in irect]
            self.label_files = [self.label_files[i] for i in irect]
            self.labels = [self.labels[i] for i in irect]
            
            if self.angle:  
              self.anglelist = [self.anglelist[i] for i in irect]
            self.shapes = s[irect]  # wh
#            print("SHEPS------------SHEPS--------------SHEPS-----------")        #----------------------------------JTROUBLESHOOT(concat mismatch)
#            print(self.shapes[0])  #<-----------------1920 1088----------------
#            print("------------------SHEPS ENDS------------------------")        #----------------------------------JTROUBLESHOOT_ENDS-----------
            ar = ar[irect]
 
            # Set training image shapes
            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]

#--------------------------------------------------------------------------------------------JEDIT change stride to solve concat mismatch problem
#            print("RECT PROGRESS SO FAR-------------------------------------------------")
            stride =32
#            print(stride)
#            
#            #print(self.batch_shapes)
#            print("RECT PROGRESS SO FAR-------------------------------------------------")
#---------------------------------------------------------------------------------------------JEDIT ends---------------------------------------
            
            self.batch_shapes = np.ceil(np.array(shapes) * img_size / stride + pad).astype(np.int) * stride


        # Cache images into memory for faster training (WARNING: large datasets may exceed system RAM)
        self.imgs = [None] * n
        if cache_images:
            gb = 0  # Gigabytes of cached images
            self.img_hw0, self.img_hw = [None] * n, [None] * n
            results = ThreadPool(8).imap(lambda x: load_image(*x, schannel), zip(repeat(self), range(n)))  # 8 threads
            pbar = tqdm(enumerate(results), total=n)
            for i, x in pbar:
                self.imgs[i], self.img_hw0[i], self.img_hw[i] = x  # img, hw_original, hw_resized = load_image(self, i)
                gb += self.imgs[i].nbytes
                pbar.desc = f'{prefix}Caching images ({gb / 1E9:.1f}GB)'

    def cache_labels(self, path=Path('./labels.cache'), prefix=''):
        # Cache dataset labels, check images and read shapes
        x = {}  # dict
        nm, nf, ne, nc = 0, 0, 0, 0  # number missing, found, empty, duplicate
        pbar = tqdm(zip(self.img_files, self.label_files), desc='Scanning images', total=len(self.img_files))
          
        
        for i, (im_file, lb_file) in enumerate(pbar):
            try:
                # verify images
                im = Image.open(im_file)
                if im_file.endswith(".tiff"):
                  im = tifffile.imread(im_file)                                  
                  im = im.swapaxes(0,1)
                  preshape = im.shape[0:2]
                  shape = preshape
                  #shape = [preshape[1],preshape[0]]
                  #shape = [120, 320]
                                                    
                if im_file.endswith(".tiff") == False:
                  im.verify()  # PIL verify
                  shape = exif_size(im)  # image size
#                print("CACHE LABELS SHAPE----------------")
                #print(shape)
                assert (shape[0] > 9) & (shape[1] > 9), f'image size {shape} <10 pixels'
                
                if im_file.endswith(".tiff") == False:
                  assert im.format.lower() in img_formats, f'invalid image format {im.format}'
                  
                #../VOC3/labels/testangle_106/gt_106_frame(1008).txt
                

                # verify labels
                if os.path.isfile(lb_file):
                    nf += 1  # label found
                    with open(lb_file, 'r') as f:
                        l = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)  # labels
#                        print("")
#                        print("WHAT DOES L LOOK LIKE AGAIN?--------")
#                        print(l)
#                        print("")
#                        
                    # angle labels
                    if self.angle: # 18th February 2022
                      #print("YEP IT DEFINITELY WENT HERE")
                      ag_file = lbl2angle_paths(lb_file)
#                      print(ag_file)
#                      print(os.path.isfile(ag_file))
#                      print("")
                      with open(ag_file, 'r') as k:
                        #print("YEP IT DEFINITELY WENT HERE#2")
                        a = np.array(k.read().strip().splitlines(), dtype=np.float32)
#                        print("")
#                        print("WHAT DOES A LOOK LIKE THEN?---------")
#                        print(a)
#                        print("")
#                        print("HOW DOES IT EVEN AFFECT L?")
#                        print(l)
#                        print("")
#                      
                    
                    if len(l):
                        assert l.shape[1] == 5, 'labels require 5 columns each'
                        assert (l >= 0).all(), 'negative labels'
                        assert (l[:, 1:] <= 1).all(), 'non-normalized or out of bounds coordinate labels'
                        assert np.unique(l, axis=0).shape[0] == l.shape[0], 'duplicate labels'
                    else:
                        ne += 1  # label empty
                        l = np.zeros((0, 5), dtype=np.float32)
                        if angle:
                          a = np.array([99])
                else:
                    nm += 1  # label missing
                    l = np.zeros((0, 5), dtype=np.float32)
                    if angle:
                          a = np.array([99])
                
                
#                print("")
#                print("WHAT ABOUT HERE?")
#                print(self.angle)
#                print("")
                if self.angle == False:
#                  print("")
#                  print("ANGLE IS FALSE IN THIS ONE")
#                  print("")
                  x[im_file] = [l, shape]
                
                if self.angle:
#                  print("")
#                  print("ANGLE IS TRUE IN THIS ONE")
#                  print("")
                  
                  x[im_file] = [l,shape,a]
#                  print("")
#                  print("SAMPLE X[IM_FILE]")
#                  print(x[im_file])
#                  print("")
                            
                
#                print("DOES IT INCLUDE FOLDER NAME?") # 17th  Feb 2022
#                print(lb_file)
#                print("")
#                print("what happens if I do this----------------------")
#                ag_file = lbl2angle_paths(lb_file)
#                print(ag_file)
#                print("")
                
            except Exception as e:
                nc += 1
                print(f'{prefix}WARNING: Ignoring corrupted image and/or label {im_file}: {e}')
                
                
#                if self.angle == True: # 17th February bloopy4
#                  if os.path
                

            pbar.desc = f"{prefix}Scanning '{path.parent / path.stem}' for images and labels... " \
                        f"{nf} found, {nm} missing, {ne} empty, {nc} corrupted"

        if nf == 0:
            print(f'{prefix}WARNING: No labels found in {path}. See {help_url}')

        x['hash'] = get_hash(self.label_files + self.img_files)
        x['results'] = [nf, nm, ne, nc, i + 1]
#        print("")
#        print("datasets.py ln543 def cache_labels")
#        print("x[hash]")      # 20th Febr 2022
#        print(x['hash'])
#        print("x['results']")
#        print(x['results'])
#        print("")
#        print("X[IM_FILE]")
#        print(len(x[im_file]))
#        print(x[im_file][0])
#        print(x[im_file][0].shape)
#        print(x[im_file][1])
#        print(x[im_file][2])
#        print(angle)
#        print("")
        
        torch.save(x, path)  # save for next time
        logging.info(f'{prefix}New cache created: {path}')
        return x
        
        

    def __len__(self):
        return len(self.img_files)

    # def __iter__(self):
    #     self.count = -1
    #     print('ran dataset iter')
    #     #self.shuffled_vector = np.random.permutation(self.nF) if self.augment else np.arange(self.nF)
    #     return self
        
    def __getitem__(self, index):
    
#        print("") # 18th Feb 2022
#        print("INDEX in _GETITEM_BEFORE")
#        print(index)
#        
        index = self.indices[index]  # linear, shuffled, or image_weights
        
#        print("") # 18th Feb 2022
#        print("INDEX in _GETITEM_AFTER")
#        print(index)
#        print("SELF . RECT")
#        print(self.rect)
#        print("")
          
          
          
        hyp = self.hyp
        mosaic = self.mosaic and random.random() < hyp['mosaic']
        schannel = self.schannel
        if mosaic:
            # Load mosaic bloopy44
            if self.angle == False:
              # print("IT SEEMS LIKE IT CAME HERE INSTEEEEEEEAD 22ND FEB")
              
              img, labels = load_mosaic(self, index, schannel, angle = self.angle)
            
            
            if self.angle:
#              print("")
#             print("AYYYYY IT CAME HEEEEEEERE 22ND FEB")
#              print(index)
#              print(schannel)
#              print(self.angle)
#              print("")
#              print("datasets.py angle + schannel troubleshoot 13th March 2022")
#              print("labels.shape")
#              print(labels.shape)
#              print("angle_s.shape")
#              print(angle_s.shape)
#              print("")
              img, labels, angle_s = load_mosaic(self, index, schannel, angle = self.angle) # self includes self.anglelist where angle array is

            shapes = None         

            # MixUp https://arxiv.org/pdf/1710.09412.pdf
            if random.random() < hyp['mixup']:
                img2, labels2 = load_mosaic(self, random.randint(0, self.n - 1))
                r = np.random.beta(8.0, 8.0)  # mixup ratio, alpha=beta=8.0
                img = (img * r + img2 * (1 - r)).astype(np.uint8)
                labels = np.concatenate((labels, labels2), 0)

        else: # 23rd Feb 2022 bloopy222
            # Load image
        
            img, (h0, w0), (h, w) = load_image(self, index, schannel)

            # Letterbox
            shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size  # final letterboxed shape
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment) #bloopy3
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

            labels = self.labels[index].copy()
            
            if self.angle:
              angle_s = self.anglelist[index].copy()
              
#              print("")
#              print("self.img files")     # 27th Feb 2022 confirm ok
#              print(self.img_files[index])
#              print("self.label files")
#              print(self.label_files[index])
#              print("angle_s")
#              print(angle_s)
#              print("")
              
            if labels.size:  # normalized xywh to pixel xyxy format
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])
                
            

        if self.augment:
            # Augment imagespace
            if not mosaic:
                if self.angle == False:
                 img, labels = random_perspective(img, labels,
                                                  degrees=hyp['degrees'],
                                                  translate=hyp['translate'],
                                                  scale=hyp['scale'],
                                                  shear=hyp['shear'],
                                                  perspective=hyp['perspective'])
                 if self.angle:
                  img, labels, angle_s = random_perspective(img, labels,  
                                      degrees=self.hyp['degrees'],        # some labels (because of scale)
                                      translate=self.hyp['translate'],
                                      scale=self.hyp['scale'],
                                      shear=self.hyp['shear'],
                                      perspective=self.hyp['perspective'],
                                      border=self.mosaic_border,
                                      angle = angle,
                                      anglelist = angle_s)  # border to remove

            # Augment colorspace
            if not schannel:
              augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])

            # Apply cutouts
            # if random.random() < 0.9:
            #     labels = cutout(img, labels)

        nL = len(labels)  # number of labels
#        print("")
#        print("is everything alright?")
#        print("angle length")
#        print(angle_s.shape)
#        print("label shape")
#        print(labels.shape)
#        print("")
        
        if nL:
            labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])  # convert xyxy to xywh
            labels[:, [2, 4]] /= img.shape[0]  # normalized height 0-1
            labels[:, [1, 3]] /= img.shape[1]  # normalized width 0-1
            if self.angle:
              # return a list of True or False representing is it positive or negative angle
              angleboolt = []
              # angleboolf = []
              
              for a in range(nL):
                if angle_s[a] >= 0:
                  angleboolt.append(True)    # ~ln785
                  #angleboolf.append(False)
                else:
                  angleboolt.append(False)
                  #angleboolf.append(True)
        
#        print("")
#        print("ANGLE_S BEFORE ANY PROCESSING")
#        print(angle_s)
#        print("")
        
        if self.augment:
            # flip up-down
            if random.random() < hyp['flipud']:
                img = np.flipud(img)
                if nL:
                    labels[:, 2] = 1 - labels[:, 2]                   
                    if self.angle:
                      angle_s = angle_s * (-1)
                      angleboolt = [not elem for elem in angleboolt]
                      
#                      for a in range(nL): # different type of processing for positive and negative radius 20th Feb 2022
#                        angle_s[a] = (angle_s[a]) * (-1)                 

#            print("")
#            print("ANGLE_S AFTER FLIPUD")
#            print(angle_s)
#            print("")
            

            # flip left-right
            if random.random() < hyp['fliplr']:
                img = np.fliplr(img)
                if nL:
                    labels[:, 1] = 1 - labels[:, 1]
                    if self.angle:
                      for a in range(nL):
#                        print("")
#                        print("is this where it gets messed up? datasets.py ")
#                        print("before")
#                        print(angle_s[a])
#                        print("after")
                        angle_s[a] = math.pi - (angle_s[a])
                        if angle_s[a] > math.pi:
                          angle_s[a] = angle_s[a] - (math.pi*2)
#                        print("when flipping is involved")
#                        print(angle_s[a])
#                        print("")
#                        if angleboolt[a] == True: # means it's in positive radian 20th Feb 2022  # ln 855 -> ln 859 (bogus formula)
#                          angle_s[a] = math.pi - angle_s[a]
#                          
#                        else:    # means negative radian
#                          angle_s[a] = (math.pi*(-1)) - angle_s[a]

      
                            
#            print("")
#            print("ANGLE_S AFTER FLIPLR")
#            print(angle_s)
#            print("")

        labels_out = torch.zeros((nL, 6))
        if self.angle:
          angle_out = torch.tensor([])
#        print("")
#        print("nL")
#        print(labels_out)
#        print(nL)
#        print(self.angle)
#        print("")
#        if nL == 0:
#          print("")
#          print("nL")
#          print(nL)
#          print("")
        if nL:       
            labels_out[:, 1:] = torch.from_numpy(labels)
            if self.angle: 
              #print("CAME HERE")
              angle_out = torch.from_numpy(angle_s)
              #print(angle_out)

        # Convert
        #print("---------------------CONVERT-------------------------")
        #print(img.shape)
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        #print(img.shape)
        img = np.ascontiguousarray(img)
        #print(img.shape)
        #print("BLIOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOP")
        
        # load images and labels returned here 
        
        if self.angle == False:
          if self.whichclass != False:
            labels_out = labels_out[labels_out[:,1] == int(whichclass)]
          return torch.from_numpy(img), labels_out, self.img_files[index], shapes

        if self.angle:
#          print("")
#          print("dataset.py")
#          print("angle shape")
#          print(angle_out.shape)
#          print("labels_out shape")
#          print(labels_out.shape)
#          print("")

#          print("")
#          print("MOSAIC")
#          print(mosaic)
#          print("self.img_files")
#          print(self.img_files[index])
#          print("angle_out")
#          print(angle_out.shape)  # 27th Feb 2022 everything seems to be accurate here too
#          print("")

          if type(self.whichclass) == int:  # if specific class only is specified
            angle_out = angle_out[labels_out[:,1]== int(self.whichclass)]
            labels_out = labels_out[labels_out[:,1] == int(self.whichclass)]
            

          return torch.from_numpy(img), labels_out, self.img_files[index], shapes, angle_out

    @staticmethod
    def collate_fn(batch):
        #print("AYY IT MANAGE TO COM HERE")
        img, label, path, shapes = zip(*batch)  # transposed

        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes
        
        
    @staticmethod
    def collate_fn_a(batch):
#       print("AYY IT MANAGE TO COME HERE")    
       img, label, path, shapes, angle_z = zip(*batch)  # transposed
       
#       print("what's batch?")
#       print(batch)
#       
#       print("")
#       print("fresh from batch")
#       print(angle_z)
#       print("")
       
       for i, l in enumerate(label):
           l[:, 0] = i  # add target image index for build_targets()
           
#       print("")
#       print("dataset.py solve selfcollate issue")
#       print("angle_z")
#       print(angle_z)
#       print("torch.cat(angle_z,0))")
#       print(torch.cat(angle_z,0))      
#       print("what about labels?")
#       print(label) 
#       print(torch.cat(label, 0))
#       print("torch.cat angle_z")
#       print(torch.cat(angle_z,0).shape)
#       print("")       
       
       return torch.stack(img, 0), torch.cat(label, 0), path, shapes, torch.cat(angle_z,0)
        

    @staticmethod
    def collate_fn4(batch):
        img, label, path, shapes = zip(*batch)  # transposed
        n = len(shapes) // 4
        img4, label4, path4, shapes4 = [], [], path[:n], shapes[:n]

        ho = torch.tensor([[0., 0, 0, 1, 0, 0]])
        wo = torch.tensor([[0., 0, 1, 0, 0, 0]])
        s = torch.tensor([[1, 1, .5, .5, .5, .5]])  # scale
        for i in range(n):  # zidane torch.zeros(16,3,720,1280)  # BCHW
            i *= 4
            if random.random() < 0.5:
                im = F.interpolate(img[i].unsqueeze(0).float(), scale_factor=2., mode='bilinear', align_corners=False)[
                    0].type(img[i].type())
                l = label[i]
            else:
                im = torch.cat((torch.cat((img[i], img[i + 1]), 1), torch.cat((img[i + 2], img[i + 3]), 1)), 2)
                l = torch.cat((label[i], label[i + 1] + ho, label[i + 2] + wo, label[i + 3] + ho + wo), 0) * s
            img4.append(im)
            label4.append(l)

        for i, l in enumerate(label4):
            l[:, 0] = i  # add target image index for build_targets()
            
        

        return torch.stack(img4, 0), torch.cat(label4, 0), path4, shapes4


# Ancillary functions --------------------------------------------------------------------------------------------------
def load_image(self, index, schannel = False):
    # loads 1 image from dataset, returns img, original hw, resized hw
    img = self.imgs[index]
    #print("at least I reached here")
#    print(self.img_files[index])
#    print(cv2.imread(self.img_files[index]) )
    if not schannel:
     if img is None:  # not cached

         path = self.img_files[index]    
         img = cv2.imread(path)  # BGR
         assert img is not None, 'Image Not Found ' + path
         h0, w0 = img.shape[:2]  # orig hw
         r = self.img_size / max(h0, w0)  # resize image to img_size
         if r != 1:  # always resize down, only resize up if training with augmentation
             interp = cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR
             img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)            
             
#             if img is not None:
#               print("THERE'S  HERE!---------------------------------")
#               print(h0)
#               print(w0)
#               print(img.shape[:2])

         return img, (h0, w0), img.shape[:2]  # img, hw_original, hw_resized
     else:
     
         return self.imgs[index], self.img_hw0[index], self.img_hw[index]  # img, hw_original, hw_resized
         
    if schannel == True: # bloopy1 31st December 2021
      #if img is None:  # not cached
    
      path = self.img_files[index]
      img = tifffile.imread(path)  # BGR
      assert img is not None, 'Image Not Found ' + path
      h0, w0 = img.shape[:2]  # orig hw
      r = self.img_size / max(h0, w0)  # resize image to img_size
      if r != 1:  # always resize down, only resize up if training with augmentation
          interp = cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR
          img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
          #if img is not None:
#            print("THERE'S  HERE!---------------------------------")
#            print(h0)
#            print(w0)
#            print(img.shape[:2])
          
      return img, (h0, w0), img.shape[:2]  # img, hw_original, hw_resized
#      else:
#          return self.imgs[index], self.img_hw0[index], self.img_hw[index]  # img, hw_original, hw_resized


def augment_hsv(img, hgain=0.5, sgain=0.5, vgain=0.5):
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed


def hist_equalize(img, clahe=True, bgr=False):
    # Equalize histogram on BGR image 'img' with img.shape(n,m,3) and range 0-255
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV if bgr else cv2.COLOR_RGB2YUV)
    if clahe:
        c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        yuv[:, :, 0] = c.apply(yuv[:, :, 0])
    else:
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])  # equalize Y channel histogram
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR if bgr else cv2.COLOR_YUV2RGB)  # convert YUV image to RGB

# bloopy24 18th February 2022 
def load_mosaic(self, index, schannel = False, angle = False):
    # loads images in a 4-mosaic
    
    labels4 = []
    anglelist4 = []
    s = self.img_size
    yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border]  # mosaic center x, y
    indices = [index] + [self.indices[random.randint(0, self.n - 1)] for _ in range(3)]  # 3 additional image indices #mosaic element 20th Feb 2022

    for i, index in enumerate(indices):
        # Load image
 

        img, _, (h, w) = load_image(self, index, schannel)

        # place img in img4
        if i == 0:  # top left
            img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        
        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        padw = x1a - x1b
        padh = y1a - y1b

        # Labels        
#        print("OK WE AT DEF LOAD MOSAIC NOW")
#        print("s")
#        print(s)       
#       
#        print("")
#        print("INDEX NOW")
#        print(index)
        labels = self.labels[index].copy()
#        print("")
#        print("troubleshoot #2")
#        print("labels.shape")
#        print(labels.shape)
#        if i == 3:
#          print("datasets.py ln 1162")
#          print(self.labels[index].shape)
#          print(self.anglelist[index].shape)
#          print(self.img_files[index])

        if self.angle:
#          print("")
#          print("anglelist at index({}): ".format(index))          
          anglelist = self.anglelist[index].copy()
#          print("len(anglelist)")
#          print(len(anglelist))
#          print("")
          anglelist4.append(anglelist)
          
        if labels.size:
            labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padw, padh)  # normalized xywh to pixel xyxy format

        labels4.append(labels)
        
        
        
#        print("")
#        print("LABELS4 BEFORE CLIPPING")
#        print(labels4)
#        print("")
    
    
    # Concat/clip labels
#    print("dataset.py ln1196")
#    print(labels4[0].shape)
#    print(anglelist4[0].shape)
#    print(labels4[1].shape)
#    print(anglelist4[1].shape)
#    print(labels4[2].shape)
#    print(anglelist4[2].shape)
#    print(labels4[3].shape)
#    print(anglelist4[3].shape)
    if len(labels4):
        labels4 = np.concatenate(labels4, 0)
        np.clip(labels4[:, 1:], 0, 2 * s, out=labels4[:, 1:])  # use with random_perspective
        
        if self.angle:
          anglelist4 = np.concatenate(anglelist4, 0)
        if self.angle == False:
          anglelist4 = []
        # img4, labels4 = replicate(img4, labels4)  # replicate
#    print("")
#    print("LABELS4 and anglist4.SHAPE BEFORE PERESPECTIVE")
#    print(labels4.shape)
#    print(anglelist4)
#    print("")
    # Augment
    if self.angle == False:
          img4, labels4 = random_perspective(img4, labels4,
                                             degrees=self.hyp['degrees'],
                                             translate=self.hyp['translate'],
                                             scale=self.hyp['scale'],
                                             shear=self.hyp['shear'],
                                             perspective=self.hyp['perspective'],
                                             border=self.mosaic_border,
                                             angle = angle,
                                             anglelist = anglelist4)  # border to remove
    
    if self.angle:
          
          img4, labels4, anglelist4 = random_perspective(img4, labels4,    # we need anglelist here because for some reason random_perspective removes
                                       degrees=self.hyp['degrees'],        # some labels (because of scale)
                                       translate=self.hyp['translate'],
                                       scale=self.hyp['scale'],
                                       shear=self.hyp['shear'],
                                       perspective=self.hyp['perspective'],
                                       border=self.mosaic_border,
                                       angle = angle,
                                       anglelist = anglelist4)  # border to remove

    if angle == False:
      return img4, labels4
      
    if angle:
#      print("")
#      print("SOMETHING SHOULD BE HERE ANGLELIST4")
#      print(anglelist4)
#      print("")
      return img4, labels4, anglelist4


def load_mosaic9(self, index):
    # loads images in a 9-mosaic

    labels9 = []
    s = self.img_size
    indices = [index] + [self.indices[random.randint(0, self.n - 1)] for _ in range(8)]  # 8 additional image indices
    for i, index in enumerate(indices):
        # Load image
        img, _, (h, w) = load_image(self, index)

        # place img in img9
        if i == 0:  # center
            img9 = np.full((s * 3, s * 3, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
            h0, w0 = h, w
            c = s, s, s + w, s + h  # xmin, ymin, xmax, ymax (base) coordinates
        elif i == 1:  # top
            c = s, s - h, s + w, s
        elif i == 2:  # top right
            c = s + wp, s - h, s + wp + w, s
        elif i == 3:  # right
            c = s + w0, s, s + w0 + w, s + h
        elif i == 4:  # bottom right
            c = s + w0, s + hp, s + w0 + w, s + hp + h
        elif i == 5:  # bottom
            c = s + w0 - w, s + h0, s + w0, s + h0 + h
        elif i == 6:  # bottom left
            c = s + w0 - wp - w, s + h0, s + w0 - wp, s + h0 + h
        elif i == 7:  # left
            c = s - w, s + h0 - h, s, s + h0
        elif i == 8:  # top left
            c = s - w, s + h0 - hp - h, s, s + h0 - hp

        padx, pady = c[:2]
        x1, y1, x2, y2 = [max(x, 0) for x in c]  # allocate coords

        # Labels
        labels = self.labels[index].copy()
        if labels.size:
            labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padx, pady)  # normalized xywh to pixel xyxy format
        labels9.append(labels)

        # Image
        img9[y1:y2, x1:x2] = img[y1 - pady:, x1 - padx:]  # img9[ymin:ymax, xmin:xmax]
        hp, wp = h, w  # height, width previous

    # Offset
    yc, xc = [int(random.uniform(0, s)) for x in self.mosaic_border]  # mosaic center x, y
    img9 = img9[yc:yc + 2 * s, xc:xc + 2 * s]

    # Concat/clip labels
    if len(labels9):
        labels9 = np.concatenate(labels9, 0)
        labels9[:, [1, 3]] -= xc
        labels9[:, [2, 4]] -= yc

        np.clip(labels9[:, 1:], 0, 2 * s, out=labels9[:, 1:])  # use with random_perspective
        # img9, labels9 = replicate(img9, labels9)  # replicate

    # Augment
    img9, labels9 = random_perspective(img9, labels9,
                                       degrees=self.hyp['degrees'],
                                       translate=self.hyp['translate'],
                                       scale=self.hyp['scale'],
                                       shear=self.hyp['shear'],
                                       perspective=self.hyp['perspective'],
                                       border=self.mosaic_border)  # border to remove

    return img9, labels9


def replicate(img, labels):
    # Replicate labels
    h, w = img.shape[:2]
    boxes = labels[:, 1:].astype(int)
    x1, y1, x2, y2 = boxes.T
    s = ((x2 - x1) + (y2 - y1)) / 2  # side length (pixels)
    for i in s.argsort()[:round(s.size * 0.5)]:  # smallest indices
        x1b, y1b, x2b, y2b = boxes[i]
        bh, bw = y2b - y1b, x2b - x1b
        yc, xc = int(random.uniform(0, h - bh)), int(random.uniform(0, w - bw))  # offset x, y
        x1a, y1a, x2a, y2a = [xc, yc, xc + bw, yc + bh]
        img[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        labels = np.append(labels, [[labels[i, 0], x1a, y1a, x2a, y2a]], axis=0)

    return img, labels


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    
#    print("LETTERBOX TROUBLESHOOT=========================")
#    print(img.shape)
    if img.shape[2]!=6:
      img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    if img.shape[2] == 6:
      img1 = cv2.copyMakeBorder(img[:,:,:3], top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
      img2 = cv2.copyMakeBorder(img[:,:,3:6], top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
      img = np.dstack((img1, img2))
    #print("its ok-------------------!!!!!!!!!!")
    return img, ratio, (dw, dh)


def random_perspective(img, targets=(), degrees=10, translate=.1, scale=.1, shear=10, perspective=0.0, border=(0, 0), angle = False, anglelist = []):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # targets = [cls, xyxy]

    height = img.shape[0] + border[0] * 2  # shape(h,w,c)
    width = img.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            img = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(114, 114, 114))
        else:  # affine
            img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

    # Visualize
    # import matplotlib.pyplot as plt
    # ax = plt.subplots(1, 2, figsize=(12, 6))[1].ravel()
    # ax[0].imshow(img[:, :, ::-1])  # base
    # ax[1].imshow(img2[:, :, ::-1])  # warped

    # Transform label coordinates
    n = len(targets)
    if n:
        # warp points
        xy = np.ones((n * 4, 3))
        xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ M.T  # transform
        if perspective:
            xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)  # rescale
        else:  # affine
            xy = xy[:, :2].reshape(n, 8)

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # # apply angle-based reduction of bounding boxes
        # radians = a * math.pi / 180
        # reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
        # x = (xy[:, 2] + xy[:, 0]) / 2
        # y = (xy[:, 3] + xy[:, 1]) / 2
        # w = (xy[:, 2] - xy[:, 0]) * reduction
        # h = (xy[:, 3] - xy[:, 1]) * reduction
        # xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).T

        # clip boxes
        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)

        # filter candidates
        i = box_candidates(box1=targets[:, 1:5].T * s, box2=xy.T)
#        print("")
#        print("AT RANDOM PERSPECTIVE NOW")
#        print(i.shape)
#        print(anglelist.shape)
#        print(targets.shape)
#        print("")
#        print("")
#        print("targets before i's interference")
#        print(targets.shape)
        targets = targets[i]
        if angle: # 20th Feb 2022 omit the same index of angle the same with labels
#          # 13th March 2022 troublshoot error when schannel and angle is together
#          print("")
#          print("datasets.py")
#          print(i)
#          print(i.shape)
#          print(targets.shape)
#          print(anglelist.shape)
#          print("")
          anglelist = anglelist[i]
          
          
          
          
        targets[:, 1:5] = xy[i]

    
    if angle == False:
      return img, targets
    if angle:
      return img, targets, anglelist


def box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.1, eps=1e-16):  # box1(4,n), box2(4,n)
    # Compute candidate boxes: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # aspect ratio
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)  # candidates


def cutout(image, labels):
    # Applies image cutout augmentation https://arxiv.org/abs/1708.04552
    h, w = image.shape[:2]

    def bbox_ioa(box1, box2):
        # Returns the intersection over box2 area given box1, box2. box1 is 4, box2 is nx4. boxes are x1y1x2y2
        box2 = box2.transpose()

        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

        # Intersection area
        inter_area = (np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)).clip(0) * \
                     (np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)).clip(0)

        # box2 area
        box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) + 1e-16

        # Intersection over box2 area
        return inter_area / box2_area

    # create random masks
    scales = [0.5] * 1 + [0.25] * 2 + [0.125] * 4 + [0.0625] * 8 + [0.03125] * 16  # image size fraction
    for s in scales:
        mask_h = random.randint(1, int(h * s))
        mask_w = random.randint(1, int(w * s))

        # box
        xmin = max(0, random.randint(0, w) - mask_w // 2)
        ymin = max(0, random.randint(0, h) - mask_h // 2)
        xmax = min(w, xmin + mask_w)
        ymax = min(h, ymin + mask_h)

        # apply random color mask
        image[ymin:ymax, xmin:xmax] = [random.randint(64, 191) for _ in range(3)]

        # return unobscured labels
        if len(labels) and s > 0.03:
            box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
            ioa = bbox_ioa(box, labels[:, 1:5])  # intersection over area
            labels = labels[ioa < 0.60]  # remove >60% obscured labels

    return labels


def create_folder(path='./new'):
    # Create folder
    if os.path.exists(path):
        shutil.rmtree(path)  # delete output folder
    os.makedirs(path)  # make new output folder


def flatten_recursive(path='../coco128'):
    # Flatten a recursive directory by bringing all files to top level
    new_path = Path(path + '_flat')
    create_folder(new_path)
    for file in tqdm(glob.glob(str(Path(path)) + '/**/*.*', recursive=True)):
        shutil.copyfile(file, new_path / Path(file).name)


def extract_boxes(path='../coco128/'):  # from utils.datasets import *; extract_boxes('../coco128')
    # Convert detection dataset into classification dataset, with one directory per class

    path = Path(path)  # images dir
    shutil.rmtree(path / 'classifier') if (path / 'classifier').is_dir() else None  # remove existing
    files = list(path.rglob('*.*'))
    n = len(files)  # number of files
    for im_file in tqdm(files, total=n):
        if im_file.suffix[1:] in img_formats:
            # image
            im = cv2.imread(str(im_file))[..., ::-1]  # BGR to RGB
            h, w = im.shape[:2]

            # labels
            lb_file = Path(img2label_paths([str(im_file)])[0])
            if Path(lb_file).exists():
                with open(lb_file, 'r') as f:
                    lb = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)  # labels

                for j, x in enumerate(lb):
                    c = int(x[0])  # class
                    f = (path / 'classifier') / f'{c}' / f'{path.stem}_{im_file.stem}_{j}.jpg'  # new filename
                    if not f.parent.is_dir():
                        f.parent.mkdir(parents=True)

                    b = x[1:] * [w, h, w, h]  # box
                    # b[2:] = b[2:].max()  # rectangle to square
                    b[2:] = b[2:] * 1.2 + 3  # pad
                    b = xywh2xyxy(b.reshape(-1, 4)).ravel().astype(np.int)

                    b[[0, 2]] = np.clip(b[[0, 2]], 0, w)  # clip boxes outside of image
                    b[[1, 3]] = np.clip(b[[1, 3]], 0, h)
                    assert cv2.imwrite(str(f), im[b[1]:b[3], b[0]:b[2]]), f'box failure in {f}'


def autosplit(path='../coco128', weights=(0.9, 0.1, 0.0)):  # from utils.datasets import *; autosplit('../coco128')
    """ Autosplit a dataset into train/val/test splits and save path/autosplit_*.txt files
    # Arguments
        path:       Path to images directory
        weights:    Train, val, test weights (list)
    """
    path = Path(path)  # images dir
    files = list(path.rglob('*.*'))
    n = len(files)  # number of files
    indices = random.choices([0, 1, 2], weights=weights, k=n)  # assign each image to a split
    txt = ['autosplit_train.txt', 'autosplit_val.txt', 'autosplit_test.txt']  # 3 txt files
    [(path / x).unlink() for x in txt if (path / x).exists()]  # remove existing
    for i, img in tqdm(zip(indices, files), total=n):
        if img.suffix[1:] in img_formats:
            with open(path / txt[i], 'a') as f:
                f.write(str(img) + '\n')  # add image to txt file
