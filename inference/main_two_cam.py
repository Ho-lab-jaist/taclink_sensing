"""Model inference for TacLink (barrel shape)
Input: two cameras
Created date: June 13th, 2024
Lastest update: June 13th, 2024
"""

import cv2
import torch
import numpy as np
import time
import pandas as pd

from inference import config
from network.tacnet_model import TacNet
from utils.processing import BinaryTacImageProcessor
from utils import image_processing_tools as ip
from utils.visualize import TactileVisualize
from utils.utils import CameraControl
from utils.utils import compute_signed_displacement, create_nodal_radial_vectors


dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(dev)

def load_model(model_name, 
               in_nc,
               num_of_features):
    MODEL_PATH = config.NETWORK_PATH / model_name
    tacnet = TacNet(in_nc = in_nc,
                    num_of_features = num_of_features)
    print('model [TacNet] was created')
    print('loading the model from {0}'.format(MODEL_PATH))
    tacnet.load_state_dict(torch.load(MODEL_PATH))
    print('---------- Tactile Networks initialized -------------')
    tacnet.to(dev)
    tacnet.eval()

    return tacnet

def input_image_process(frame_raw, transform):
    processed_frame = transform(frame_raw)
    return processed_frame


MODEL_NAME = "tacnet_two_cam_model.pt"

tacnet = load_model(model_name=MODEL_NAME, 
                    in_nc=2,
                    num_of_features=585)

DATA_PATH = config.RESOURCE_PATH
INIT_PATH = DATA_PATH / "init_pos.csv"
FULL_INIT_PATH = DATA_PATH / "full_init_pos.csv"
ACTIVE_IDX_PATH = DATA_PATH / "active_node_index.csv"

active_node_idexes = np.genfromtxt(ACTIVE_IDX_PATH, 
                                   delimiter=',', 
                                   dtype = np.int16)
full_init_pos = np.genfromtxt(FULL_INIT_PATH, delimiter=',')
full_estimated_pos = np.array(full_init_pos)
radial_vectors = create_nodal_radial_vectors(full_estimated_pos)

init_pos_csv = pd.read_csv(INIT_PATH)
init_positions = (np.array(init_pos_csv.iloc[0, 1:], dtype=float)
                  .reshape(-1, 3))

SKIN_VTK_PATH = config.RESOURCE_PATH / 'barrel_shaped_skin.vtk'
taclink = TactileVisualize(skin_path = str(SKIN_VTK_PATH),
                           init_positions = full_init_pos,
                           depth_range = [-5, 20])

"""
Test with video
"""
# # Paths to the video file
# video_camA = 'tactile_streams_camA.mp4'
# video_camB = 'tactile_streams_camB.mp4'
# # Open the video file
# camA = cv2.VideoCapture(video_camA)
# camB = cv2.VideoCapture(video_camB)

"""
Setting parameters
- Ubuntu
 exposure_mode: manual
 exposure_value: 60
 brightness: 47
 contrast: 32 (default)

- Windows
 exposure_mode: manual
 exposure_value: -6
 brightness: 0 (default)
 contrast: 32 (default)
"""

camA = CameraControl(cam_id = 0,
                    fps= 120, 
                    mjpg_enabled = True,
                    os = 'ubuntu',
                    exposure_value = 123)
camA.set_brightness(47)
camA.set_contrast(32)

camB = CameraControl(cam_id =4,
                    fps= 120, 
                    mjpg_enabled = True,
                    os = 'ubuntu',
                    exposure_value = 123)
camB.set_brightness(47)
camB.set_contrast(32)

"""Real Image Processing"""
transformA = BinaryTacImageProcessor(
                                threshold = 90,
                                filter_size = 3,
                                cropped_size  = (435, 435),
                                resized_size  = (256, 256),
                                apply_mask = True,
                                mask_radius = 130,
                                apply_block_mask = True,
                                block_mask_radius = 38, # 45
                                block_mask_center = (123, 132))

transformB = BinaryTacImageProcessor(
                                threshold = 90,
                                filter_size = 3,
                                cropped_size  = (435, 435),
                                resized_size  = (256, 256),
                                apply_mask = True,
                                mask_radius = 130,
                                apply_block_mask = True,
                                block_mask_radius = 40, # 47
                                block_mask_center = (125, 130))

# Define the rotation matrix
w = 640
h = 480
center = (w//2, h//2)
angle = 180  # rotation angle in degrees
scale = 1.0  # scale factor
M_rotate  = cv2.getRotationMatrix2D(center, angle, scale)
M_rotate_3x3 = np.vstack([M_rotate, [0, 0, 1]])

# Define the translation matrix
tx = 10  # translation along x-axis
ty = -20  # translation along y-axis
M_translate_3x3 = np.eye(3)
M_translate_3x3[0, 2] = tx
M_translate_3x3[1, 2] = ty

# Combine the rotation and translation matrices
M_combined = np.dot(M_translate_3x3, M_rotate_3x3)


while camA.isOpened() and camB.isOpened():
    frame_rawA = camA.read()
    frame_rawB = camB.read()
    
    frame_rawB = cv2.warpAffine(frame_rawB, M_combined[:2], (w, h))
    
    processed_frameA = input_image_process(frame_rawA, transformA)
    processed_frameB = input_image_process(frame_rawB, transformB)
    

    combined_frame = torch.cat((processed_frameA, processed_frameB), 
                        dim=0)
    tac_img = combined_frame.unsqueeze(0)
    input_img = tac_img.to(dev)

    # forward pass to TacNet
    with torch.no_grad():
        # predict of skin deviation
        estimated_positions = ( tacnet(input_img).cpu().numpy().reshape(-1, 3) 
                                + init_positions )
    
    full_estimated_pos[active_node_idexes] = estimated_positions

    # compute deviation intensity at every nodes on the skin surface    
    signed_est_norm_deviations = compute_signed_displacement(full_estimated_pos - full_init_pos,
                                                            radial_vectors)
    
    contact_depth = np.max(signed_est_norm_deviations)
    print("Maximum contact depth: {}".format(contact_depth))


    # for contact depth visualization
    taclink.skin_est['contact depth (unit:mm)'] = signed_est_norm_deviations  # type: ignore
    taclink.skin_est.points = full_estimated_pos

    # print(rgb_frame.dtype, rgb_frame.shape)
    cv2.imshow("RGB Image A", frame_rawA)
    cv2.imshow("RGB Image B", frame_rawB)
    # cv2.imshow("Original Image", frame_raw)
    cv2.imshow("Binary img A", ip.tensor2img(processed_frameA))
    cv2.imshow("Binary img B", ip.tensor2img(processed_frameB))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
