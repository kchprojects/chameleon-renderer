import imageio
import cv2 
import numpy as np
from sys import argv

base_folder = argv[1]
original_folder = argv[2]
out = argv[3]
image_lst = []
 
for i in range(40):
    frame = cv2.imread(f"{base_folder}/{i}.png")
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mask = frame_rgb==0
    
    origin = cv2.imread(f"{original_folder}/{i}.png")
    origin_rgb = cv2.cvtColor(origin, cv2.COLOR_BGR2RGB)
    frame_rgb[:,:,1] = 0
    frame_rgb[:,:,2] = 0
    # frame_rgb = np.vstack((frame_rgb,origin_rgb)) 
    frame_rgb = cv2.addWeighted(frame_rgb,0.5,origin_rgb,0.5,1)
    frame_rgb[mask] = origin_rgb[mask]
    image_lst.append(frame_rgb)
# Convert to gif using the imageio.mimsave method
imageio.mimsave(out, image_lst, fps=10)