import bpy
import json
import numpy as np


def load_setup(filename):
    curr_mats = {}
    with open(filename, "r") as file:
        data_json = json.load(file)
        for cam_id in data_json:
            cam_data = data_json[cam_id]
            mat = np.asarray(cam_data["view_mat"])
            mat = mat.reshape((4, 4))
            curr_mats[cam_id] = mat
            
    return curr_mats


def make_camera_object(cam_name,world_mat):
        cam_data = bpy.data.cameras.new(name=cam_name)

        cam_object = bpy.data.objects.new(
            name=cam_name, object_data=cam_data)
        cam_object.matrix_world = world_mat
        cam_object.location = world_mat[:3,3]
        return cam_object
    
def create_setup(setup_name, matrices):
    collection = bpy.data.collections.new(setup_name)
    for mat_id in matrices:
        new_cam = make_camera_object(f"{mat_id}",matrices[mat_id])
        collection.objects.link(new_cam)
    bpy.context.scene.collection.children.link(collection)
    
    
matrices = load_setup("/home/karelch/Diplomka/rendering/chameleon-renderer/examples/validate_colmap/cameras/cameras.json")
create_setup("sexy_cams", matrices)