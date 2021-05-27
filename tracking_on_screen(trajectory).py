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
import argparse
import random
import pandas as pd

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
    np.random.seed(round(label*1.333+2.222))

    random_factor = np.random.randint(0,255,size=[1,1]).reshape(1,).tolist()[0]
    # color = np.random.randint(0,255,size=[1,3]).reshape(3,).tolist()
    # color = [(c*random_factor)%255 for c in color]

    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]

    return tuple(color)



def draw_boxesANDcenter(img, bbox, identities=None, offset=(0, 0), label_text=True):
    colorset = []
    ids = []
    centers = []
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        color = compute_color_for_labels(id)
        # print(color)
        label = '{}{:d}'.format("", id)
        
        center_x = int(np.average([x1,x2]))
        center_y = int(np.average([y1,y2]))
        
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.circle(img,  (center_x, center_y), 2, color, -1)
        # label the corresponding ids for each tracked fish

        cv2.putText(img, label, (x1+1, y1+13), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)
        colorset.append(color)
        ids.append(id)
        centers.append([center_x, center_y])
    
    return img, colorset, ids, centers


# resize the screen capture into certain ratio of width and height
def process_img(original_image, resize_width = 928, resize_height = 512):
    processed_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    processed_img = cv2.resize(processed_img, (resize_width, resize_height))
    return processed_img


def draw_grid(img, width, height, seg):
    for s in range(1, seg):
        cv2.line(img, 
                 (0, int((height/4)*s)), 
                 (width, int((height/4)*s)), 
                 (99, 99, 0), 1, 1)
        cv2.line(img, 
                 (int((width/4)*s), 0), 
                 (int((width/4)*s), height), 
                 (99, 99, 0), 1, 1)
    return img


track_history = {}
record_time = []
fish_number = []
info_dic = {'record_time':[],
           'count': []}



def detect(opt):
    with torch.no_grad():
        global track_history
        # initialize deepsort
        cfg = get_config()
        cfg.merge_from_file(opt.config_deepsort)
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
        device = select_device(opt.device)
    
        # Load model
        model = torch.load(opt.weights, map_location=device)['model'].float()  # load to FP32
        model.to(device).eval() 
        # names = model.module.names if hasattr(model, 'module') else model.names
        # Set Dataloader
    
        fps_interval = 0
        info_time = 0
        while True:
            if fps_interval == 0:
                t1 = time.time()
            
            if info_time == 0:
                info_t1 = time.time()
                
                
            printscreen_pil = np.array(grab_screen(region = (opt.region_x1, opt.region_y1, 
                                                             opt.region_x2, opt.region_y2)))
            frame = process_img(printscreen_pil, opt.resize_width, opt.resize_height)
            img = frame.copy()      # img is gpu format
            img = np.transpose(img, (2, 0, 1))
            img = torch.from_numpy(img).to(device)
            img = img.float()       # uint8 to fp32
            img /= 255.0
        
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
        
            pred = model(img, augment=False)[0]
            pred = non_max_suppression(pred, 
                                       opt.conf_thres, 
                                       opt.iou_thres, 
                                       classes=opt.classes, 
                                       agnostic=False)

            for i, det in enumerate(pred):
                if det is not None and len(det):
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
                        
                        
                    xywhs = torch.Tensor(bbox_xywh)
                    confss = torch.Tensor(confs)
    
                    # Pass detections to deepsort
                    outputs = deepsort.update(xywhs, confss, printscreen_pil)   
                    
                    # draw grids
                    draw_grid(printscreen_pil, opt.resize_width, opt.resize_height, 4)
                    
                    if len(outputs) > 0:          
                        bbox_xyxy = outputs[:, :4]
                        identities = outputs[:, -1]
                      
                        _, colorset, ids, centers = draw_boxesANDcenter(printscreen_pil, 
                                                                        bbox_xyxy, 
                                                                        identities)
                        
                        ### take out those false detections
                        ## capture and save those images with mis-counting bounding box to local dir
                        # if len(outputs) != 5:
                        #     printscreen_pil_not5 = printscreen_pil.copy()
                        #     cv2.putText(printscreen_pil_not5, '%d' % len(outputs), 
                        #                 (450, 300), 
                        #                 cv2.FONT_HERSHEY_PLAIN, 5, [255, 0, 0], 1)
                            
                        #     cv2.imwrite(r'C:\Users\82677\Desktop\bbb\%s.png' % int(time.time()), 
                        #                 printscreen_pil_not5)
                        
                        
                        


                        
                                
                        #######################################################
                        # draw tracking line for detected objects
                        for index in range(len(ids)):
                            tt1 = time.time()
                            
                            if ids[index] not in track_history:
                                track_history[ids[index]] = {'color': colorset[index],
                                                             'center': [centers[index]],
                                                             'lastest_time': tt1,
                                                             'time_span': 0}
                            elif ids[index] in track_history:
                                track_history[ids[index]]['color'] = colorset[index]
                                track_history[ids[index]]['center'].append(centers[index])
                                track_history[ids[index]]['lastest_time'] = tt1
                                                    
                            center_list = track_history[ids[index]]['center']

                            if len(center_list) >= 30:                         # hyperparameters
                                track_history[ids[index]]['center'].pop(0)
                                          
                            if len(center_list) >= 2:
                                for cl in range(0, len(center_list)-1):
                                    cv2.line(printscreen_pil, 
                                             tuple(center_list[cl]), 
                                             tuple(center_list[cl+1]), 
                                             track_history[ids[index]]['color'], 2)         
                        #######################################################              
                    ###########################################################
                    
                    
                        # print(outputs)
                   
                   # pop out those ids that were missing for more that 1.5 seconds
                    pop_track = []
                    for single_track in track_history:
                        track_history[single_track]['time_span'] = \
                            (time.time() - track_history[single_track]['lastest_time'])
                        
                        if track_history[single_track]['time_span'] > 0.75:       # hyperparameters
                            pop_track.append(single_track)
                    
                    
                    for p_t in pop_track:
                        # if p_t in track_history:
                        _ = track_history.pop(p_t)

                        
                    ###########################################################
                else:
                    outputs = []
                    
                    
            
            # # # Record information for tracking
            # if not opt.Record:
            #     p = os.path.join(opt.record_path, opt.record_name)
                
            # ## Notice that for every second, it will generate a text file with size around 2kB
            # ## Thus, if you wanna loop it for a long time,
            # ## do not forget to set a boundary in case insufficient disk capacity
            #     output_str = str(outputs).replace('[', '').replace(']', '')                              
            #     with open(os.path.join(opt.record_path, 'withopen.txt'), 'a') as f:
            #         f.write(output_str+'\n')        
  
            
            if fps_interval == 0:
                t2 = time.time()

                
                
                
            fps = 1 / (t2 - t1)
            cv2.putText(printscreen_pil, '%.1ffps' % fps, (5, 15), 
                        cv2.FONT_HERSHEY_PLAIN, 1, [255, 0, 0], 1)
            
            fps_interval = (fps_interval + 1) % 10    # calculate the frame rates once every 10 frames
            
            info_time += 1
            
            if time.time() - info_t1 > 1:      # record info for every X sec
                record_time.append(time.asctime(time.localtime(time.time())))
                fish_number.append(len(outputs))
                # print(record_time)
                # print(fish_number)
                info_time = 0


            cv2.imshow('capture', printscreen_pil[:, :, ::-1])
            
            # press q to exit the window
            if cv2.waitKey(1) & 0xFF == ord('q'):
                # info_dic['record_time'] = record_time
                # info_dic['count'] = fish_number
                # info_df = pd.DataFrame(info_dic)
                # print(info_df)
                # info_df.to_csv(r'C:\Users\82677\Desktop\info_df.csv',
                #                 index=False,header=True)]

                
                
                break
        
        cv2.destroyWindow(winname='capture')    



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights',type=str,default='yolov5/weights/fish150.pt',help='model.pt path')
    parser.add_argument("--config_deepsort",type=str,default="deep_sort_pytorch/configs/deep_sort.yaml")
    
    parser.add_argument('--Record',action='store_false',help='whether to store the detection record or not')
    parser.add_argument('--record-path',type=str,default='inference/output',help='output folder')  
    parser.add_argument('--record-name',type=str,default='tracking_record.txt',help='output text file name')  
    
    parser.add_argument('--device',default='0',help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--conf-thres',type=float,default=0.6,help='object confidence threshold')
    parser.add_argument('--iou-thres',type=float,default=0.4, help='IOU threshold for NMS')
    parser.add_argument('--classes',nargs='+',type=int,default=[0],help='filter by class')  # class 0 is person
    # parser.add_argument('--classes',nargs='+',type=int,default=[1,2,3,4,5,6,7,8,9,10],help='filter by class')  # class 0 is person
    
    parser.add_argument('--agnostic-nms',action='store_true',help='class-agnostic NMS')
    parser.add_argument('--augment',action='store_true',help='augmented inference')
    
    parser.add_argument('--region-x1',type=int,default=0,help='topleft x coordinate')
    parser.add_argument('--region-y1',type=int,default=100,help='topleft y coordinate')
    parser.add_argument('--region-x2',type=int,default=928,help='rightbottom x coordinate')
    parser.add_argument('--region-y2',type=int,default=612,help='rightbottom y coordinate')
    
    parser.add_argument('--resize-width',type=int,default=928,help='resize width of image for processing')
    parser.add_argument('--resize-height',type=int,default=512,help='resize height of image for processing')

    args = parser.parse_args()

    detect(args)

