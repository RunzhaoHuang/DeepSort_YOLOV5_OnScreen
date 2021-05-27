# Import packages
import win32api
import sys
sys.path.append(r'C:\Users\82677\Desktop\DeepSort Tracking OnScreen\DeepSort_OnScreen')
import numpy as np
from PIL import ImageGrab
import torch
from IPython.display import Image, clear_output
from matplotlib import pyplot as plt
import os
import platform
import shutil
import time
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.torch_utils import select_device, load_classifier, time_synchronized
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from grabscreen import grab_screen 
clear_output()
print('Setup complete. Using torch %s %s' %(torch.__version__,
                                            torch.cuda.get_device_capability() if torch.cuda.is_available() else 'cpu'))
import os
os.chdir(r'C:\Users\82677\Desktop\DeepSort Tracking OnScreen\DeepSort_OnScreen')



#%% Detect objects on screen
# Initialize
device = select_device()
# Load model
model = attempt_load('taipo100.pt', 
                     map_location=device)     # load FP32 model cuda

# Get names and colors
names = model.module.names if hasattr(model, 'module') else model.names
colors = [[random.randint(0, 225) for _ in range(3)] for _ in range(len(names))]

def process_img(original_image):
    processed_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    processed_img = cv2.resize(processed_img, (928, 512))
    return processed_img

while True:
    #time.sleep(1)
    #win32api.keybd_event(87, 0, 0, 0)
    
    printscreen_pil = np.array(grab_screen(region = (0, 175, 928, 690)))
    #print('printscreen_pil:', np.shape(printscreen_pil))
    frame = process_img(printscreen_pil)
    #print(....
    #
    img = frame.copy()      # img is gpu format
    #print(..
    img = np.transpose(img, (2, 0, 1))

    img = torch.from_numpy(img).to(device)
    img = img.float()       # uint8 to fp32
    img /= 255.0

    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    pred = model(img)[0]
    pred = non_max_suppression(pred, 0.4, 0.5)

    gn = torch.tensor(frame.shape)[[1, 0, 1, 0]]  # normalization gain  get the size of image
    if pred != [None]:
        for i, det in enumerate(pred):
            # Rescale boxes from img_size to im0 size
            det[:,:4] = scale_coords(img.shape[2:], det[:, :4], printscreen_pil.shape).round()
            #write results
            for *xyxy, conf, cls in reversed(det):
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh 
                label = '%s %.2f' % (names[int(cls)], conf)
                plot_one_box(xyxy, printscreen_pil, label = label, color = colors[int(cls)], line_thickness = 1)  # utils.general 

    cv2.imshow('capture', printscreen_pil[:, :, ::-1])
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
#cap.release()
cv2.destroyWindow(winname='capture')


#%%
# Detect object using webcam
import cv2
# Initialize
device = select_device()
# Load model
model = attempt_load('yolov5s.pt', map_location = device) # load FP32 model cuda
# Get names and colors
names = model.module.names if hasattr(model, 'module') else model.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
cap = cv2.VideoCapture(0)
while(1):
    # get a frame
    ret, frame = cap.read()
    # processing
    img = frame.copy()      # img is in gpu format, which cannot be read by conventional methods
    img = np.transpose(img, (2,0,1))    # torch.Size([480, 640, 3])转torch.Size([3, 480, 640])
    img = torch.from_numpy(img).to(device)
    img = img.float()   # uint8 to fp32
    img /= 255.0

    #print(np.shape(img))
    if img.ndimension() == 3:
        img = img.unsqueeze(0)      # Add one dimension to img 
    #
    pred = model(img)[0]
    # Apply NMS
    pred = non_max_suppression(pred, 0.4, 0.5)      # output shreshold > 0.4
    # Drawing
    gn = torch.tensor(frame.shape)[[1, 0, 1, 0]]  # normalization gain 
    if pred != [None]:
        for i, det in enumerate(pred):
            # Rescale boxes from img_size to im0 size
            det[:,:4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
            #write results
            for *xyxy, conf, cls in reversed(det):
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                label = '%s %.2f' % (names[int(cls)], conf)
                plot_one_box(xyxy, frame, label = label, color = colors[int(cls)], line_thickness = 1)  # utils.general
    # show a frame
    cv2.imshow('capture', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# cap.release()
cv2.destroyWindow(winname='capture')




#%%
# Detect object on Screen, and set an alert area, mask, to mark down object number changes

# Initialize
device = select_device()
frame_h = 512
frame_w = 928

obj_count = 0   # object number in alert area
obj_count_old = 0    
take_photo_num = 0  
# Each detection may not capture every object all the timeso we set a buf, fer to take the average value, because we want to avoid the loss of the target in a frame.
obj_count_buf = np.array([0,0,0,0,0])     

# load the model
model = attempt_load('yolov5s.pt', map_location = device) # load FP32 model cuda
# Get names and colors
names = model.module.names if hasattr(model, 'module') else model.names
colors = [[random.randint(0, 225) for _ in range(3)] for _ in range(len(names))]
# imgsz = check_img_size(486, s=model.stride.max())  # check img_size

frame_mask = np.zeros((frame_h, frame_w, 3), dtype = np.uint8)  # Make a mask with the same size
position = [(300, 300), (300, 500), (500, 500), (500, 300)]     # Four points defines a alert area
cv2.fillPoly(frame_mask, [np.array(position)], (0, 0, 255))     # The color inside the alert area is (0, 0, 255)

def process_img(original_image):    # Process the original frame/image
    processed_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    processed_img = cv2.resize(processed_img, (frame_w, frame_h))
    return processed_img

cv2.namedWindow('frame')

# def MouseEvent(a, b, c, d, e):  # Mouse response function
#     if(a==1):   # Click left button to get the coordinates
#         print(b, c)
# cv2.setMouseCallback('frame', MouseEvent)   
while True:
    # get a frame
    start = time.time()
    # frame = np.array(ImageGrab.grab(bbox = (0, 175, 928, 690)))   # slow
    frame = grab_screen(region = (0, 175, 928, 690))   # much faster 
    if np.shape(frame):     
        #processing
        frame = process_img(frame)
        img = frame.copy()  # img is in gpu format, which cannot be read by conventional methods
        img = np.transpose(img, (2,0,1))    # torch.Size([480, 800, 3]) to torch.Size([3, 480, 800])
        img = torch.from_numpy(img).to(device) 
        img = img.float()   # uint8 to fp32
        img /= 255.0   # 0 - 255 to 0.0 - 1.0
        #print()
        if img.ndimension() == 3:
            img = img.unsqueeze(0)   # Add one dimension to img 
        pred = model(img)[0]
        # Apply NMS
        pred = non_max_suppression(pred, 0.4, 0.5)  # Output shreshold > 0.4
        
        # Drawing
        if pred != [None]:
            for i, det in enumerate(pred):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
                # Write results
                for *xyxy, conf, cls in reversed(det):     # But it looks like you can also use det instead of reversed(det)         
                    if cls == 0: # Here, cls=0 means we only care about the class 'person'
                        # label = '%s %.2f' % (names[int(cls)], conf) 
                        label = f'{names[int(cls)]}' 
                        plot_one_box(xyxy, frame, label = label, color = colors[int(cls)], line_thickness = 1) # utils.general
                        xy = torch.tensor(xyxy).tolist()    
                        x, y, x1, y1 = int(xy[0]), int(xy[1]), int(xy[2]), int(xy[3])  # Retrieve the coordinates of the bbox
                        center_xy = (int(np.average([x, x1])), int(np.average([y, y1])))  # calculate the center point
                        if (frame_mask[(center_xy[1], center_xy[0])] == [0, 0, 255]).all():  # If the center of the object lies inside the alert area
                            obj_color = (0, 0, 255)     # Change the color of the object center
                            obj_count +=1
                        else:
                            obj_color = (0, 255, 0)   # Else it reminds normal color
                        cv2.circle(frame, center_xy, 2, obj_color, -1)      # Draw a circle for object center
        obj_count_buf = np.append(obj_count_buf[1:], obj_count)     # Update the buffer
        cbr = int(np.around(np.average(obj_count_buf)))
      
        cv2.putText(frame, 'obj_count:%s  take_photo:%s' % (cbr, take_photo_num), (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        frame = cv2.addWeighted(frame, 1.0, frame_mask, 0.1, 0.0)   # Draw a mask
        if (obj_count_old != cbr):
            take_photo_num += 1
            cv2.imwrite("./photo/%s.jpg" % take_photo_num, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 100])  # Save screencapture
            print('take photo number :%s' % take_photo_num)  # Display the total number of photo taken
            cv2.putText(frame, 'Alert', (250, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)  
        
        obj_count_old = cbr  
        obj_count = 0  # Clear the obj_count in this frame, waiting for another detection for the next frame
        fps = "%.2f fps" % (1 / (time.time() - start))     # Calculate the fps
        cv2.putText(frame, '%s' % (fps), (845, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)        
        # show a frame

        cv2.imshow("frame", frame)
        cv2.imshow("frame_mask", frame_mask[:, :, ::-1])
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    

cv2.destroyWindow(winname='frame')
cv2.destroyWindow(winname='frame_mask')
