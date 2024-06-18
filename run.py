import os
import numpy as np
import matplotlib.pyplot as plt
import torch

from tools import DataLoader, Calibration, MathTools, TorchTools
from disparity import DisparityEstimation
from reconstruction import DGNet
from warp import ImageWarp


def run(scene, gpu=True):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    ms_image_raw, calibration_images, camera_pos_array = DataLoader.load_record(scene)
    if calibration_images is None:
        # outdoor scene
        calibration_matrices = Calibration.run_corners_only(DataLoader.load_calibration_corners(scene))
    else:
        # scene calibrated by checkerboard pattern
        calibration_matrices = Calibration.run(calibration_images, 12, 9)

    if scene == 'outdoor':
        DisparityEstimation.IGEV_SEARCH_RANGE = 64
        DisparityEstimation.IGEV_OFFSET = 32
    elif scene == 'lab':
        DisparityEstimation.IGEV_SEARCH_RANGE = 64
        DisparityEstimation.IGEV_OFFSET = 48
    elif scene == 'office':
        DisparityEstimation.IGEV_SEARCH_RANGE = 128
        DisparityEstimation.IGEV_OFFSET = 64

    ms_image_raw = ms_image_raw.float()
    if gpu:
        ms_image_raw = ms_image_raw.cuda()

    DisparityEstimation.load_model(camera_pos_array, gpu=gpu)
    DGNet.load_model(gpu=gpu)
    
    # Run multispectral snapshot image registration
    disparity = DisparityEstimation.run(ms_image_raw, calibration_matrices, camera_pos_array)
    
    plt.imshow(disparity[0, 0].cpu())
    plt.show()
    
    ms_image_masked, ms_mask = ImageWarp.run(ms_image_raw, disparity, calibration_matrices, camera_pos_array)
    ms_image = DGNet.run(ms_image_masked, ms_mask)
    # Finished multispectral snapshot image registration
    
    ms_image = ms_image.cpu().numpy()

    combs = [[0, 4, 7], [3, 4, 5], [6, 4, 2], [1, 4, 8]]
    for comb in combs:
        rgb_image = np.dstack((ms_image[0, comb[0], :, :], ms_image[0, comb[1], :, :], ms_image[0, comb[2], :, :]))
        plt.imshow(rgb_image)
        plt.show()

if __name__ == '__main__':
    scenes = ['lab', 'outdoor', 'office']
    for scene in scenes:
        run(scene)
