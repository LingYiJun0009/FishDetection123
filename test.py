import argparse
import json
import os
from pathlib import Path
from threading import Thread
from time import sleep

import numpy as np
import torch
import yaml
from tqdm import tqdm

from models.experimental import attempt_load
from utils.datasets import create_dataloader
from utils.general import coco80_to_coco91_class, check_dataset, check_file, check_img_size, check_requirements, \
    box_iou, non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, set_logging, increment_path, colorstr
from utils.metrics import ap_per_class, ConfusionMatrix
from utils.plots import plot_images, output_to_target, plot_study_txt
from utils.torch_utils import select_device, time_synchronized

# 25th September 2022 get intermediate feature maps
activation = {}
def get_activation(name):
  def hook(model, input, output):
    activation[name] = output.detach()
  return hook

def test(data,
         weights=None,
         batch_size=32,
         imgsz=640,
         conf_thres=0.001,
         iou_thres=0.6,  # for NMS
         save_json=False,
         single_cls=False,
         augment=False,
         verbose=False,
         model=None,
         dataloader=None,
         save_dir=Path(''),  # for saving images
         save_txt=True,  # for auto-labelling
         save_hybrid=False,  # for hybrid auto-labelling
         save_conf=True,  # save auto-label confidences
         plots=True,
         log_imgs=0,  # number of logged images
         compute_loss=None,
         angle_t =  False,
         whichclass = False):
    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        
        device = next(model.parameters()).device  # get model device

    else:  # called directly
        set_logging()
        device = select_device(opt.device, batch_size=batch_size)

        # Directories
        save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size


        # Multi-GPU disabled, incompatible with .half() https://github.com/ultralytics/yolov5/issues/99
        # if device.type != 'cpu' and torch.cuda.device_count() > 1:
        #     model = nn.DataParallel(model)

    # Half
    half = device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    # Configure
    model.eval()
    model.train()
    model.eval()
    is_coco = data.endswith('coco.yaml')  # is COCO dataset
    with open(data) as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)  # model dict
    check_dataset(data)  # check
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Logging
    log_imgs, wandb = min(log_imgs, 100), None  # ceil
    try:
        import wandb  # Weights & Biases
    except ImportError:
        log_imgs = 0

    # Dataloader
    if not training:
        # 3rd July 2022----------------
#        print("")
#        print("TEST.PY LN 91 IT WILL COME HERE EVEN DURING VALIDATION") #apparantly not
#        print("")
        
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        path = data['test'] if opt.task == 'test' else data['val']  # path to val/test images
        #print("THE NEXT LETTER IS MODEL STRIDE MAX------------------------------------")
        #print(model.stride.max())
        
        if angle_t == False:
         dataloader = create_dataloader(path, imgsz, batch_size, model.stride.max(), opt, pad=0.5, rect=True,
                                        prefix=colorstr('test: ' if opt.task == 'test' else 'val: '))[0]
        if angle_t:
            dataloader = create_dataloader(path, imgsz, batch_size, model.stride.max(), opt, pad=0.5, rect=True, angle = True,
                                       prefix=colorstr('test: ' if opt.task == 'test' else 'val: '))[0]
          
                                       
        # 3rd July 2022 -----------------create_dataloader for angle test, else the dataloader above doesn't include angle
#        dataloader = create_dataloader(path, imgsz, total_batch_size, gs, opt,  # testloader
#                                       hyp=hyp, cache=opt.cache_images and not opt.notest, rect=True, rank=-1,
#                                       world_size=opt.world_size, workers=opt.workers,
#                                       pad=0.5, prefix=colorstr('val: '), schannel = opt.schannel, angle = opt.angle, whichclass = opt.whichclass)[0] 
        

    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    coco91class = coco80_to_coco91_class()
    # s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@.5', 'mAP@.5:.95') # original 24th Feb 2022
    s = ('%20s' + '%12s' * 7) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@.5', 'mAP@.5:.95', 'anglediff')
    p, r, f1, mp, mr, map50, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(3, device=device)

    jdict, stats, ap, ap_class, wandb_images = [], [], [], [], []
    for batch_i, (img, targets, paths, shapes, *angle) in enumerate(tqdm(dataloader, desc=s)): 

#        print("IMAGE SHAPE ----------------------")#----------------------------------------------------------------------JEDIT-----------
#        print(img.shape)
#        print("IMAGE SHAPE ENDSSSSS.........................................")

#        print("")
#        print("test.py")
#        print("paths")
#        print(paths)
#        print("angle")
#        print(angle)
#        print("")
        img = img.to(device, non_blocking=True)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        nb, _, height, width = img.shape  # batch size, channels, height, width       

        with torch.no_grad():
            # Run model
            t = time_synchronized()
            #print("test.py 25th September 2022")
#            print("----model.named_parameters-----")
#            print(model.named_parameters) # print out layer in model
            #print("----------outputmodel----------")
            #print(model.model[24]) # this will output Detect module have what layer [25th September]
            #x[0] biggest pyramid ..  x[2] smallest pyramid
            #print(model.model[24].m[0])
            # get intermediate layers 25th September 2022 (cv3 is the last layer in C3 module in BottleNeck)
            # register forward hook before running the model below (before making inferences 25th September)
            
            model.model[17].cv3.act.register_forward_hook(get_activation('model[17].cv3.act'))
            model.model[20].cv3.act.register_forward_hook(get_activation('model[20].cv3.act'))
            model.model[23].cv3.act.register_forward_hook(get_activation('model[23].cv3.act'))
            model.model[24].m[2].register_forward_hook(get_activation('model[24].m[2]')) # smallest pyramid: torch.Size([1, 3, 16, 21, 7])

            inf_out, train_out = model(img, augment=augment)  # inference and training outputs
            
#            print("")
#            print("test.py path")
#            print(paths) 
#            print("26th September 2022 inf_out:")
#            print(inf_out.shape)  # should be the one that doesn't make sense
#            # .detach().cpu().numpy()
#            np.save("./scratch26thSept/{}_inf_out.npy".format(os.path.basename(paths[0])[:-4]),inf_out.detach().cpu().numpy())
#            print("train_out")
#            print(train_out[2].shape) # I wonder if they have the same values... probably
            
            # printfeature maps [25th September 2022] 
#            print("")
#            print("ACTIVATION")
#            if len(activation)!=0: 
#              print("")
#              print("forward hook layer shape")
#              print(activation['model[17].cv3.act'].shape)
#              print(activation['model[20].cv3.act'].shape)
#              print(activation['model[23].cv3.act'].shape)
              #print(activation['model[24].m[2]'].shape)
            
            t0 += time_synchronized() - t

            # Compute loss 

            if compute_loss and (angle_t == False): # angle is the array, angle_t is just boolean
                loss += compute_loss([x.float() for x in train_out], targets, angle, whichclass)[1][:3]  # box, obj, cls
               
            if compute_loss and angle_t:
                _,lossi, anglediff = compute_loss([x.float() for x in train_out], targets, angle, whichclass)  # box, obj, cls, ag(angle/anglediff will be removed later)
                
                targets[:,0:2] = torch.round(targets[:,0:2])
                
                targets = torch.cat((torch.round(targets[:,0:2]), targets[:,2:6]),1)
#                print("what about now")
#                print(targets)
#                print("")
                loss = lossi[:3] + loss
#            print("")
#            print("where does the non scalar operation take place")
#            print("before")
#            print(targets)
#            print("after")
#            print(torch.round(targets[:,0:2]))
#            print("")
            # Run NMS
            targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
            lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
            t = time_synchronized() 
            
            if angle_t:
              angle_out = inf_out[:,:,-1]  # seperate prediction angle and the rest
              inf_out = inf_out[:,:,:-1]
              output, ag_output = non_max_suppression(inf_out, conf_thres=conf_thres, iou_thres=iou_thres, labels=lb, angle_out = angle_out)
            
            if angle_t == False:
              inf_out = inf_out[:,:,:-1]
              output = non_max_suppression(inf_out, conf_thres=conf_thres, iou_thres=iou_thres, labels=lb)
              
            t1 += time_synchronized() - t

        # Statistics per image        
        for si, pred in enumerate(output):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            path = Path(paths[si])
            seen += 1

            if len(pred) == 0:    # this piece of code just open a new txt file 
                newpath = save_dir / 'labels'  
                if not os.path.exists(newpath):
                  os.makedirs(newpath)
                with open(save_dir / 'labels' / (path.stem + '.txt'), 'a') as f:
                  pass
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Predictions
            predn = pred.clone()
#            print("")
#            print("attempt to include angle in save_txt")
#            print(pred.shape)
#            print(ag_output[0][si].item())
#            print("")
            scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # native-space pred
            
            # Append to text file
            if save_txt:
                gn = torch.tensor(shapes[si][0])[[1, 0, 1, 0]]  # normalization gain whwh
                newpath = save_dir / 'labels'  
                if not os.path.exists(newpath):
                  os.makedirs(newpath)
                with open(save_dir / 'labels' / (path.stem + '.txt'), 'a') as f:
                
                 for index, (*xyxy, conf, cls) in enumerate(predn.tolist()):                  
                     xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                     line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                     
#                     print("")
#                     print("test.py agoutput")
#                     print(ag_output)
#                     print(index)
#                     
#                     print("")
                     if angle_t:
                       if type(whichclass) != int and conf > 0.001: 
#                         print("ag_output")
#                         print(ag_output)                     
                         try:
                           f.write(('%g ' * len(line)).rstrip() % line + ' ' + str(ag_output[si][index].item()) + '\n')
                         except:
                           print("yeah there's an error, here's ag_output")
                           print(len(ag_output[0]))
                       if type(whichclass) == int:
                         if cls == whichclass and conf > 0.001:                        
                           f.write(('%g ' * len(line)).rstrip() % line + ' ' + str(ag_output[si][index].item()) + '\n')
                     if not angle_t:
                       if type(whichclass) != int and conf > 0.001:                      
                         f.write(('%g ' * len(line)).rstrip() % line + '\n')
                       if type(whichclass) == int:
                         if cls == whichclass and conf > 0.001:                        
                           f.write(('%g ' * len(line)).rstrip() % line + '\n')
                      

            # W&B logging
#            if plots and len(wandb_images) < log_imgs:
#                box_data = [{"position": {"minX": xyxy[0], "minY": xyxy[1], "maxX": xyxy[2], "maxY": xyxy[3]},
#                             "class_id": int(cls),
#                             "box_caption": "%s %.3f" % (names[cls], conf),
#                             "scores": {"class_score": conf},
#                             "domain": "pixel"} for *xyxy, conf, cls in pred.tolist()]
#                boxes = {"predictions": {"box_data": box_data, "class_labels": names}}  # inference-space
##                print("-------------------------TEST FILE--------------------------------")
##                print(img[si].shape)
##                print("------------------------------------------------------------------")
#                wandb_images.append(wandb.Image(img[si][3:6, :, :], boxes=boxes, caption=path.name))

            # Append to pycocotools JSON dictionary
            if save_json:
                # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                image_id = int(path.stem) if path.stem.isnumeric() else path.stem
                box = xyxy2xywh(predn[:, :4])  # xywh
                box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                for p, b in zip(pred.tolist(), box.tolist()):
                    jdict.append({'image_id': image_id,
                                  'category_id': coco91class[int(p[5])] if is_coco else int(p[5]),
                                  'bbox': [round(x, 3) for x in b],
                                  'score': round(p[4], 5)})

            # Assign all predictions as incorrect
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
            if nl:
                detected = []  # target indices
                tcls_tensor = labels[:, 0]

                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5])
                scale_coords(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])  # native-space labels
                if plots:
                    confusion_matrix.process_batch(predn, torch.cat((labels[:, 0:1], tbox), 1))

                # Per target class
                for cls in torch.unique(tcls_tensor):
                    ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # prediction indices
                    pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # target indices
#                    print("")
#                    print("figuring out correc.cpu() test.py")
#                    print(ti)
#                    print(pi)
#                    print(pred[:,5])
#                    print("")
                    # Search for detections
                    if pi.shape[0]:
#                        print("")
#                        print("just checking the values  so i can use box_iou")
#                        print(predn[pi, :4].shape)
#                        print(tbox[ti].shape)
#                        print(predn[pi, :4][:, None, 2:].shape)
#                        print(tbox[ti][:, 2:].shape)
                        # Prediction to target ious
                        ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)  # best ious, indices
                        # Append detections
                        detected_set = set()
#                        print("")
#                        print("iouv")
#                        print(iouv)
#                        print("ious")
#                        print(ious)
#                        print("")
                        #print((ious > iouv[0]).nonzero(as_tuple=False))
                        for j in (ious > iouv[0]).nonzero(as_tuple=False):
                            d = ti[i[j]]  # detected target
                            if d.item() not in detected_set:
                                detected_set.add(d.item())
                                detected.append(d)
                                correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                if len(detected) == nl:  # all targets already located in image
                                    break

            # Append statistics (correct, conf, pcls, tcls)
#            print("")
#            print("correct")
#            print(len(correct))
#            print("pred[:,4]")
#            print(pred[:,4])
#            print("pred[:,5]")
#            print(pred[:,5])
#            print("")

            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

        # Plot images
        if plots and batch_i < 743:
            f = save_dir / f'test_batch{batch_i}_labels.jpg'  # labels
            
            if not angle_t:
              Thread(target=plot_images, args=(img, targets, paths, f, [], names, False, whichclass), daemon=True).start()
            
            if angle_t:
             # print("")
             # print("how does targets differ : test.py")
              targets = targets.detach().cpu()
#              print(targets.shape)
#              print(targets[0])
#              print(targets.type())
#              print("")
#
#              print("")
#              print("test.py")
#              print("paths")
#              print(paths)
#              print("angle")
#              print(angle)
#              print("")
              Thread(target=plot_images, args=(img, targets, paths, f, angle, names, False, whichclass), daemon=True).start()
                      
            f = save_dir / f'test_batch{batch_i}_pred.jpg'  # predictions
            
            if not angle_t:              
              Thread(target=plot_images, args=(img, output_to_target(output), paths, f, [], names, True, whichclass), daemon=True).start()
              
            if angle_t:
#              print("")
#              print("f")
#              print(f)
#              print("")
#              print("")
#              print("--------------look for this-------------")
#              print(torch.cat(ag_output, 0).shape)  # (600,0)
#              print(len(ag_output))
#              print(ag_output[0].shape)   # (300,0)
#              print(ag_output[1].shape)   # (300,0)
#              print("")

              Thread(target=plot_images, args=(img, output_to_target(output), paths, f, [torch.cat(ag_output, 0).flatten()], names, True, whichclass), daemon=True).start()
              
    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
        print("")
        print("ap")
        print(ap[:,0])
        print("stats")
        print(stats[0].shape)
        print("")
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    if angle_t and training:
     pf = '%20s' + '%12.3g' * 7  # print format
     print(pf % ('all', seen, nt.sum(), mp, mr, map50, map, anglediff))
    
     
    if angle_t == False or not training:
     pf = '%20s' + '%12.3g' * 6  # print format
     print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))
      

    # Print results per class
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Print speeds
    t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (imgsz, imgsz, batch_size)  # tuple
    if not training:
        print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)

    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        if wandb and wandb.run:
            val_batches = [wandb.Image(str(f), caption=f.name) for f in sorted(save_dir.glob('test*.jpg'))]
            wandb.log({"Images": wandb_images, "Validation": val_batches}, commit=False)

    # Save JSON
    if save_json and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
        anno_json = '../coco/annotations/instances_val2017.json'  # annotations json
        pred_json = str(save_dir / f"{w}_predictions.json")  # predictions json
        print('\nEvaluating pycocotools mAP... saving %s...' % pred_json)
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            anno = COCO(anno_json)  # init annotations api
            pred = anno.loadRes(pred_json)  # init predictions api
            eval = COCOeval(anno, pred, 'bbox')
            if is_coco:
                eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.img_files]  # image IDs to evaluate
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        except Exception as e:
            print(f'pycocotools unable to run: {e}')

    # Return results
    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")
    model.float()  # for training
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
#        
#    print("")
#    print("test.py map are there multiples there?")
#    print(map)
#    print("*(loss.cpu() / len(dataloader)")
#    print(*(loss.cpu() / len(dataloader)))
    
    if angle_t and training: # add an anglediff in result.txt
      return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist(), anglediff.cpu()), maps, t
      
    else:
      return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='*.data path')
    parser.add_argument('--batch-size', type=int, default=32, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--task', default='val', help="'val', 'test', 'study'")
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--project', default='runs/test', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--angle', action='store_true', help='switch to include angle in training')
    opt = parser.parse_args()
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.data = check_file(opt.data)  # check file
    print(opt)
    check_requirements()

    if opt.task in ['val', 'test']:  # run normally
        test(opt.data,
             opt.weights,
             opt.batch_size,
             opt.img_size,
             opt.conf_thres,
             opt.iou_thres,
             opt.save_json,
             opt.single_cls,
             opt.augment,
             opt.verbose,
             save_txt=opt.save_txt | opt.save_hybrid,
             save_hybrid=opt.save_hybrid,
             save_conf=opt.save_conf,
             angle_t = opt.angle,
             )

    elif opt.task == 'speed':  # speed benchmarks
        for w in opt.weights:
            test(opt.data, w, opt.batch_size, opt.img_size, 0.25, 0.45, save_json=False, plots=False)

    elif opt.task == 'study':  # run over a range of settings and save/plot
        x = list(range(256, 1536 + 128, 128))  # x axis (image sizes)
        for w in opt.weights:
            f = f'study_{Path(opt.data).stem}_{Path(w).stem}.txt'  # filename to save to
            y = []  # y axis
            for i in x:  # img-size
                print(f'\nRunning {f} point {i}...')
                r, _, t = test(opt.data, w, opt.batch_size, i, opt.conf_thres, opt.iou_thres, opt.save_json,
                               plots=False)
                y.append(r + t)  # results and times
            np.savetxt(f, y, fmt='%10.4g')  # save
        os.system('zip -r study.zip study_*.txt')
        plot_study_txt(x=x)  # plot
