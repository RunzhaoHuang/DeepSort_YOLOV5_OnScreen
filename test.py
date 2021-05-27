import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import sys
sys.path.insert(0, './yolov5')
from yolov5.utils.datasets import LoadImages
from yolov5.utils.general import non_max_suppression, scale_coords
from yolov5.utils.torch_utils import select_device
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
import shutil
import cv2
import torch
from grabscreen import grab_screen 
import numpy as np
import time
import pandas as pd
import queue


# convert the xy coordinate of the bounding box to the format like, 
# center_x, center_y, width, height of the bounding box.
def bbox_rel(*xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    
    x_c = (xyxy[0].item() + xyxy[2].item()) /2
    y_c = (xyxy[1].item() + xyxy[3].item()) /2

    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h


# create a Palette
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def draw_boxes(img, bbox, identities=None, offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        color = compute_color_for_labels(id)
        label = '{}{:d}'.format("", id)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, y1), cv2.FONT_HERSHEY_PLAIN, 1, [255, 255, 255], 2)
    return img


# resize the screen capture into certain ratio of width and height
def process_img(original_image, resize_width = 928, resize_height = 512):
    processed_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    processed_img = cv2.resize(processed_img, (resize_width, resize_height))
    return processed_img


def get_center(array):
    x = (array[0] + array[2]) / 2
    y = (array[1] + array[3]) / 2
    
    return [x, y, array[-1]]


def get_distance(l1, l2):
    square = pow((l1[0] - l2[0]), 2) + pow((l1[1] - l2[1]), 2)
    
    return np.sqrt(square)
    







weights = r'yolov5/weights/fish150.pt'
config_deepsort = r"deep_sort_pytorch/configs/deep_sort.yaml"
record_path = r'inference/output'
record_name = r'tracking_record.txt'
conf_thres = 0.4
iou_thres = 0.5

region_x1 = 0
region_x2 = 928
region_y1 = 175
region_y2 = 687

resize_width = 928
resize_height = 512



with torch.no_grad():
    
    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(config_deepsort)
    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, 
                        min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, 
                        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, 
                        n_init=cfg.DEEPSORT.N_INIT, 
                        nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)
    
    # Initialize
    device = select_device()

    # Load model
    model = torch.load(weights, map_location=device)['model'].float()  # load to FP32
    model.to(device).eval() 
    # names = model.module.names if hasattr(model, 'module') else model.names
    # Set Dataloader

    fps_interval = 0
    
    
    loop = 0
    pool_outputs = [0,1,2,3,4]
    pool_id = [0,11,2,3,4]
    thre = 40.0
    
    
    
    while True:
        if fps_interval % 5 == 0:
            t1 = time.time()
        
        printscreen_pil = np.array(grab_screen(region = (region_x1, region_y1, 
                                                         region_x2, region_y2)))
        frame = process_img(printscreen_pil, resize_width, resize_height)
        img = frame.copy()      # img is gpu format
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).to(device)
        img = img.float()       # uint8 to fp32
        img /= 255.0
    
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
    
        pred = model(img, augment=False)[0]
        pred = non_max_suppression(pred, 
                                   conf_thres, 
                                   iou_thres, 
                                   classes=[0], 
                                   agnostic=False)
        
        
        
        
        for i, det in enumerate(pred):

            if det is not None and len(det):
                tmp_id = []
                
                # Rescale boxes from img_size to im0 size
                det[:,:4] = scale_coords(img.shape[2:], det[:, :4], 
                                         printscreen_pil.shape).round()
                # write results
                bbox_xywh = []
                confs = []
                
                for *xyxy, conf, cls in det:
                    x_c, y_c, bbox_w, bbox_h = bbox_rel(*xyxy)
                    obj = [x_c, y_c, bbox_w, bbox_h]
                    bbox_xywh.append(obj)
                    confs.append([conf.item()])
                    # print('%s' % (names[int(cls)]))
                    
                xywhs = torch.Tensor(bbox_xywh)
                confss = torch.Tensor(confs)

                # Pass detections to deepsort
                outputs = deepsort.update(xywhs, confss, printscreen_pil)   
                
                
                
                
                

                
                
                pool_outputs[loop] = outputs
                
                
                for o in range(len(outputs)):
                    tmp_id.append(outputs[o][4])
                
                
                
                
                
                pool_id[loop] = tmp_id
                
                

                
                if type(pool_id[loop-1]) != int and \
                   type(pool_id[loop-2]) != int and \
                   type(pool_id[loop-3]) != int and \
                   type(pool_id[loop-4]) != int and \
                   type(pool_id[loop-5]) != int:
                       
                    # print('before: ', outputs)
                    
                    
                    for j in range(len(outputs)):
                        if outputs[j][4] not in pool_id[loop-1]:
                            
                            
                            outlier = get_center(outputs[j])
                            # print('outlier: ', outlier)
                            
                            
                            
                            p_distance = []
                            for p in pool_outputs[loop-1]:
                                p_distance.append([get_distance(get_center(p), 
                                                                outlier),  p[-1]])
                            sorted_distance = sorted(p_distance, key=(lambda x:x[0]))
                            
                            
                            
                            
                            
                            
                            
                            if len(sorted_distance) > 0 and \
                                sorted_distance[0][0] <= thre and \
                                sorted_distance[0][1] not in pool_id[loop]:
                                
                                outputs[j][4] = sorted_distance[0][1]
                                
                            
                            
                            
                            
                            
                            # print('new id: ', outputs[j][4])
                                
                            # print(sorted_distance)              
                            
    
    
    
    
    
    
  
                
                # print('after: ', outputs)
                # print('\n')
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -1]
                    draw_boxes(printscreen_pil, bbox_xyxy, identities)
                    
                    
                    
                    
                    
                    
                    
                    # save outputs into a text file
                    # p = os.path.join(record_path, record_name)
                    # with open(p, 'a') as f:
                    #     f.write(str(outputs)+'\n\n')

        # calculate the frame rates once every 5 frames
        if fps_interval % 5 == 0:
            t2 = time.time()   
        
        fps = 1 / (t2 - t1)
        cv2.putText(printscreen_pil, 'fps: %.1f' % fps, (5, 15), 
                    cv2.FONT_HERSHEY_PLAIN, 1, [255, 0, 0], 2)
        fps_interval += 1
        
        cv2.imshow('capture', printscreen_pil[:, :, ::-1])
        

    
        loop += 1
        if loop == 5:
            loop = 0
    
    
    
    
        
    
        # press q to exit the window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        
    cv2.destroyWindow(winname='capture')    



    
   

