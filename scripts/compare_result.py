import cv2 as cv
from cv2 import split
import numpy as np
from sys import argv
from glob import glob

def compate_result(result_path, original_path, output_path):
    for position_folder in glob(f"{result_path}/*"):
        position = position_folder.split("/")[-1]
        orig_pos_folder = f"{original_path}/position_{position}"
        cv.namedWindow("original",cv.WINDOW_NORMAL)
        cv.namedWindow("result",cv.WINDOW_NORMAL)
        for result_view_path in glob(f"{position_folder}/*.png"):
            view = result_view_path.split("/")[-1]
            orig_image_path = f"{orig_pos_folder}/{view}"
            print(result_view_path, "vs", orig_image_path)
            result_image = cv.imread(result_view_path,cv.IMREAD_COLOR)/255.0
            orig_image = cv.imread(orig_image_path,cv.IMREAD_COLOR)
            if orig_image is None:
                continue
            else:
                orig_image = orig_image/255.0
            cv.imshow("original", orig_image)
            cv.imshow("result", result_image)
            cv.waitKey()
            
            

result_path = argv[1]
original_path = argv[2]
output_path = argv[3]

compate_result(result_path,original_path,output_path)
