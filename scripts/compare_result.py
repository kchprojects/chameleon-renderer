import cv2 as cv
import numpy as np
from sys import argv
from os import makedirs
from glob import glob
import json


def compate_result(result_path, original_path, output_path):
    info_data = {"positions":[]}
    pix_sum = 0
    err_sum = 0
    for position_folder in glob(f"{result_path}/*"):
        position = position_folder.split("/")[-1]
        print(position_folder)
        new_position = {"pos_label":position, "lights":[]}
        
        orig_pos_folder = f"{original_path}/position_{position}"
        # cv.namedWindow("original",cv.WINDOW_NORMAL)
        # cv.namedWindow("result",cv.WINDOW_NORMAL)
        # cv.namedWindow("abs_diff",cv.WINDOW_NORMAL)
        mask = None;
        for result_view_path in glob(f"{position_folder}/*.png"):
            result_image = cv.imread(result_view_path,cv.IMREAD_GRAYSCALE)
            if mask is None:
                mask = np.ones((result_image.shape[0],result_image.shape[1],3))
                mask[result_image==0,:] = 0
            else:
                mask[result_image!=0,:] = 1
                
                
        pix_count = cv.countNonZero(mask[:,:,0])
        curr_out_path = f"{output_path}/err_images/{position}/"
        try:
            makedirs(curr_out_path)
        except: 
            pass
        for result_view_path in glob(f"{position_folder}/*.png"):
            pix_sum += pix_count
            view = result_view_path.split("/")[-1]
            orig_image_path = f"{orig_pos_folder}/{view}"
            # print(result_view_path, "vs", orig_image_path)
            result_image = mask * cv.imread(result_view_path,cv.IMREAD_COLOR)/255.0
            orig_image = cv.imread(orig_image_path,cv.IMREAD_COLOR)
            if orig_image is None:
                continue
            else:
                orig_image = mask * orig_image/255.0
                
            abs_diff = np.abs(result_image-orig_image)
            abs_err_sum = np.sum(abs_diff[mask != 0])
            mae = abs_err_sum/pix_count
            err_sum += abs_err_sum
            
            new_position["lights"].append({"light":view, "mae": mae})
            cv.imwrite(f"{curr_out_path}/{view}",result_image*255)
            # cv.imshow("original", orig_image)
            # cv.imshow("result", result_image)
            # cv.imshow("abs_diff", abs_diff)
            # cv.waitKey()
            
        info_data["positions"].append(new_position)
        break
    info_data["mae"] = err_sum/pix_sum
    with open(f"{output_path}/info.json","w") as file:
        json.dump(info_data,file,indent=4)
        
result_path = argv[1]
original_path = argv[2]

forests = ["test_col.bp_forest","test_col.lamb_forest"]
datasets = ["fi_rock_bot_small","coin_bot_small", "pcb_bot_small"]
output_path = argv[3]

for obj in datasets:
    for forest in forests:
        compate_result(f"{result_path}/{obj}/render_views/{forest}",f"{original_path}/{obj}/",f"{output_path}/{obj}/{forest}/")
