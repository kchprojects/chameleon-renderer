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

def load_lights(filename):
    lights = {}
    with open(filename, "r") as file:
        lights = json.load(file)
    return lights


def make_light_object(position,name,world_mat):

    light_data = bpy.data.lights.new(name=name, type='POINT')

    light_object = bpy.data.objects.new(
        name=name, object_data=light_data)
    position_4d = np.ones(4,dtype=np.float64)
    position_4d[:3] = position
    
    light_object.location = (world_mat@position_4d)[:3]
    return light_object

def make_camera_object(cam_name,world_mat):
    cam_data = bpy.data.cameras.new(name=cam_name)

    cam_object = bpy.data.objects.new(
        name=cam_name, object_data=cam_data)
    cam_object.matrix_world = world_mat
    cam_object.location = world_mat[:3,3]
    return cam_object
    
def create_setup(setup_name, matrices,lights):
    collection = bpy.data.collections.new(setup_name)
    for mat_id in matrices:
        curr_coll = bpy.data.collections.new("pos")
        new_cam = make_camera_object(f"{mat_id}",matrices[mat_id])
        curr_coll.objects.link(new_cam)
        for l in lights:
            position = np.array([l["position"][x] for x in "xyz"])
            lo = make_light_object(position, str(l["id"]), matrices[mat_id])
            curr_coll.objects.link(lo)
        collection.children.link(curr_coll)
        break
    bpy.context.scene.collection.children.link(collection)
    
lights = load_lights("/home/karelch/Diplomka/rendering/chameleon-renderer/resources/setups/lights.json")
matrices = load_setup("/home/karelch/Diplomka/rendering/chameleon-renderer/examples/validate_colmap/cameras/cameras.json")
create_setup("sexy_cams", matrices,lights)