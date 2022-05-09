import json
from turtle import position
from scipy.spatial.transform import Rotation
import numpy as np 
from viz.viz_3d import draw_coordinates
from matplotlib import pyplot as plt


def get_intrinsic(cam_info):
    out = None
    if(cam_info[0] == "SIMPLE_RADIAL"):
        out = {
            "type_id": "opencv-matrix",
            "rows": 3,
            "cols": 3,
            "dt": "d",
            "data": [float(cam_info[3]), 0.0, float(cam_info[4]),
                     0.0, float(cam_info[3]), float(cam_info[5]), 0.0,
                     0.0, 1.0]
        }
    return out


def read_cameras(filename):
    cameras = {}
    with open(filename) as file:
        for line in file.readlines():
            if line[0] == "#":
                continue
            splitted = line.split(" ")
            cam_id = int(splitted[0])
            new_cam = {
                "camera_id": cam_id,
                "K": get_intrinsic(splitted[1:]),
                "resolution": {
                    "x": int(splitted[2]),
                    "y": int(splitted[3])
                }
            }
            cameras[cam_id] = new_cam
    return cameras

def make_view_mat(params):
    qw,qx,qy,qz,tx,ty,tz = tuple(params)
    r = Rotation.from_quat([float(qx),float(qy),float(qz),float(qw)])
    rot_mat = np.eye(4, dtype=np.float64)
    rot_mat[:3,:3] = r.as_matrix()
    t_mat = np.eye( 4, dtype=np.float64 )
    t_mat[:3,3] = [float(tx),float(ty),float(tz)]
    correction = np.eye(4,dtype=np.float64)
    correction[0,0] = 1
    correction[1,1] = -1
    correction[2,2] = -1
    
    r_corr = Rotation.from_euler("xyz",[np.pi,0,0])
    rot_mat_corr = np.eye(4, dtype=np.float64)
    rot_mat_corr[:3,:3] = r_corr.as_matrix()
    print(rot_mat_corr)

    
    out = rot_mat_corr@np.transpose(rot_mat)@t_mat#@correction
    # print(out)
    # input()
    draw_coordinates(out,ax,8)
    return out.tolist()
    
def read_positions(filename):
    positions = {}
    
    with open(filename, "r") as file:
        skip = False
        for line in file.readlines():
            if line[0] == "#":
                continue
            if not skip:
                splitted = line.split(" ")
                cam_id = int(splitted[8])
                positions[splitted[9][:-1]] = {
                    "view_mat" : make_view_mat(splitted[1:8]),
                    "img_name" : splitted[9][:-1]
                    }
            skip = not skip
    return positions
      
fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111, projection='3d')

data_folder = "/home/karelch/Diplomka/dataset_v2/mag_box"
positions = read_positions(f"{data_folder}/colmap/images.txt")
cameras = read_cameras(f"{data_folder}/colmap/cameras.txt")

ax.set_xlim([-5,5])
ax.set_ylim([-5,5])
ax.set_zlim([-5,5])
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

plt.draw()
plt.show()

out_cameras = {}
print(cameras)
for cam_key in positions:
    new_cam = cameras[1].copy()
    new_cam["view_mat"] = positions[cam_key]["view_mat"]
    new_cam["img_name"] = positions[cam_key]["img_name"]
    out_cameras[positions[cam_key]["img_name"]] = new_cam
    


with open(f"{data_folder}/cameras.json","w") as file:
    json.dump(out_cameras,file,indent=4)