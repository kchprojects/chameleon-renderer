from cv2 import CV_32FC1
import imageio
import cv2 
import numpy as np
from sys import argv
from os import makedirs

def gradmag(img):
    X = cv2.Sobel((img/255.0).astype(np.float32),cv2.CV_32F,1,0)
    Y = cv2.Sobel((img/255.0).astype(np.float32),cv2.CV_32F,0,1)
    return (cv2.magnitude(X,Y)*255).astype(np.uint8)
    # return cv2.Canny(img,80,120)

base_folder = argv[1]
original_folder = argv[2]
out_path = argv[3]
try:
    makedirs(out_path)
except:
    pass

x,y,width,height = 532,185,800,664
for i in range(40):
    frame = cv2.imread(f"{base_folder}/{i}.png")
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mask = frame_rgb==0
    # frame = gradmag(frame)
    # cv2.imshow("frame",frame)
    # cv2.waitKey()
    origin = cv2.imread(f"{original_folder}/{i}.png")
    origin_rgb = cv2.cvtColor(origin, cv2.COLOR_BGR2RGB)
    frame_rgb[:,:,1] = 0
    frame_rgb[:,:,2] = 0
    # frame_rgb = np.vstack((frame_rgb,origin_rgb))
    strength = 0.3 
    frame_rgb = cv2.addWeighted(frame_rgb,strength,origin_rgb,1-strength,1)
    frame_rgb[mask] = origin_rgb[mask]
    frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    
    frame_rgb = frame_rgb[y:y+height,x:x+width]
    cv2.imwrite(f"{out_path}/{i}.png",frame_rgb)
    
# Convert to gif using the imageio.mimsave method
