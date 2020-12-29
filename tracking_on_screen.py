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



def detect(opt):
    with torch.no_grad():
        
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
        while True:
            if fps_interval % 5 == 0:
                t1 = time.time()
            
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
                        # print('%s' % (names[int(cls)]))
                        
                    xywhs = torch.Tensor(bbox_xywh)
                    confss = torch.Tensor(confs)
    
                    # Pass detections to deepsort
                    outputs = deepsort.update(xywhs, confss, printscreen_pil)   
                    
                    if len(outputs) > 0:
                        bbox_xyxy = outputs[:, :4]
                        identities = outputs[:, -1]
                        draw_boxes(printscreen_pil, bbox_xyxy, identities)
                        
                        if opt.NotRecord:
                            p = os.path.join(opt.record_path, opt.record_name)
                            with open(p, 'a') as f:
                                f.write(str(outputs)+'\n\n')

            # calculate the frame rates once every 5 frames
            if fps_interval % 5 == 0:
                t2 = time.time()   
            
            fps = 1 / (t2 - t1)
            cv2.putText(printscreen_pil, 'fps: %.1f' % fps, (5, 15), 
                        cv2.FONT_HERSHEY_PLAIN, 1, [255, 0, 0], 2)
            fps_interval += 1
            
            cv2.imshow('capture', printscreen_pil[:, :, ::-1])
            
            # press q to exit the window
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyWindow(winname='capture')    



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights',type=str,default='yolov5/weights/yolov5s.pt',help='model.pt path')
    parser.add_argument("--config_deepsort",type=str,default="deep_sort_pytorch/configs/deep_sort.yaml")
    
    parser.add_argument('--NotRecord',action='store_false',help='whether to store the detection record or not')
    parser.add_argument('--record-path',type=str,default='inference/output',help='output folder')  
    parser.add_argument('--record-name',type=str,default='tracking_record.txt',help='output text file name')  
    
    parser.add_argument('--device',default='0',help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--conf-thres',type=float,default=0.4,help='object confidence threshold')
    parser.add_argument('--iou-thres',type=float,default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--classes',nargs='+',type=int,default=[0],help='filter by class')  # class 0 is person
    parser.add_argument('--agnostic-nms',action='store_true',help='class-agnostic NMS')
    parser.add_argument('--augment',action='store_true',help='augmented inference')
    
    parser.add_argument('--region-x1',type=int,default=0,help='topleft x coordinate')
    parser.add_argument('--region-y1',type=int,default=175,help='topleft y coordinate')
    parser.add_argument('--region-x2',type=int,default=928,help='rightbottom x coordinate')
    parser.add_argument('--region-y2',type=int,default=687,help='rightbottom y coordinate')
    
    parser.add_argument('--resize-width',type=int,default=928,help='resize width of image for processing')
    parser.add_argument('--resize-height',type=int,default=512,help='resize height of image for processing')

    args = parser.parse_args()

    detect(args)
