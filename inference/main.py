"""Model inference for TacLink (barrel shape)"""

import cv2
import torch
import numpy as np
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

tacnet = load_model(model_name='tacnet_one_came_model.pt',
                    in_nc = 1,
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

cam = CameraControl(cam_id = 0,
                    fps= 120, 
                    mjpg_enabled = True,
                    os = 'ubuntu',
                    exposure_value = 60)
cam.set_brightness(47)
cam.set_contrast(32)

"""Real Image Processing"""
transform = BinaryTacImageProcessor(
                                threshold = 75,
                                filter_size = 3,
                                cropped_size  = (435, 435),
                                resized_size  = (256, 256),
                                apply_mask = True,
                                mask_radius = 130,
                                apply_block_mask = True,
                                block_mask_radius = 25,
                                block_mask_center = (123, 132))

while cam.isOpened():
    frame_raw = cam.read()
    processed_frame = transform(frame_raw)
    tac_img = processed_frame.unsqueeze(0)
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

    # for contact depth visualization
    taclink.skin_est['contact depth (unit:mm)'] = signed_est_norm_deviations  # type: ignore
    taclink.skin_est.points = full_estimated_pos
    
    print("Maximum contact depth: {}".format(np.max(signed_est_norm_deviations)))
    cv2.imshow("RGB Image", frame_raw)
    cv2.imshow("Processed Image", ip.tensor2img(processed_frame))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()