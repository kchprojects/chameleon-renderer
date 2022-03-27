import imageio
import cv2 
import numpy as np

base_folder = "../../build/views"
original_folder = "/home/karelch/Diplomka/photogramm_data/led_pcb"
image_lst = []
 
for i in range(50):
    frame = cv2.imread(f"{base_folder}/{i}.bmp")
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mask = frame_rgb==0
    
    origin = cv2.imread(f"{original_folder}/{i}.bmp")
    origin_rgb = cv2.cvtColor(origin, cv2.COLOR_BGR2RGB)
    frame_rgb[:,:,1] = 0
    frame_rgb[:,:,2] = 0
    # frame_rgb = np.vstack((frame_rgb,origin_rgb)) 
    frame_rgb = cv2.addWeighted(frame_rgb,0.5,origin_rgb,0.5,1)
    frame_rgb[mask] = origin_rgb[mask]
    image_lst.append(frame_rgb)
# Convert to gif using the imageio.mimsave method
imageio.mimsave('video.gif', image_lst, fps=10)