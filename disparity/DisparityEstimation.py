import torch.nn.functional as F
import torchvision.transforms.functional as VF
import torchvision.transforms as transforms
import torch
import math
import pathlib

from tools import MathTools, TorchTools
from . import Inference

DEPTH_ESTIMATION_CAMS = [0, 1, 2, 3, 5, 6, 7, 8]

IGEV_SEARCH_RANGE = 48
IGEV_OFFSET = 0
IGEV_PATCH_PADDING = 96
IGEV_PATCH_SIZE = 1600
IGEV_WEIGHTING_RHO = 0.95
DIV = 32

def load_model(camera_pos_array, path='', gpu=True, verbose=False):
    if path == '':
        path = pathlib.Path(__file__).parent.resolve().__str__() + '/model/model.pth'
    disp_to_center, disp_to_center_offset, baseline_ratios = get_real_baselines(camera_pos_array)
    for i in range(len(DEPTH_ESTIMATION_CAMS)):
        cam_idx = DEPTH_ESTIMATION_CAMS[i]
        search_range = int(math.ceil(disp_to_center[cam_idx]/DIV) * DIV)
        Inference.load_model(path, search_range, gpu=gpu, verbose=True)

def run(ms_image_raw, calib_matrices, camera_pos_array, verbose=False):
    if verbose:
        print("Calibrate cameras...")
    ms_image_calibrated = calibrate_images(ms_image_raw, calib_matrices)

    if verbose:
        print("Calc baselines...")
    disp_to_center, disp_to_center_offset, baseline_ratios = get_real_baselines(camera_pos_array)

    if verbose:
        print("Disparity estimation...")
    depth_map = get_depth_map(camera_pos_array, ms_image_calibrated, disp_to_center, disp_to_center_offset, baseline_ratios, verbose=verbose)

    return depth_map

def calibrate_images(ms_image_raw, calib_matrices):
    B, C, H, W = ms_image_raw.shape
    ms_image_calibrated = ms_image_raw.new_zeros([B, C, H, W])
    for i in range(len(calib_matrices)):
        ms_image_calibrated[:, i, ...] = TorchTools.calibrate(ms_image_raw[:, i, ...].unsqueeze(dim=1), calib_matrices[i]).squeeze(dim=1)
    return ms_image_calibrated

def get_real_baselines(camera_pos_array):
    baselines = [0] * len(camera_pos_array)
    baseline_ratios = [0] * len(camera_pos_array)
    disp_to_center = [0] * len(camera_pos_array)
    disp_to_center_offset = [0] * len(camera_pos_array)
    min_baseline = 1000000
    center_pos = camera_pos_array[len(camera_pos_array)//2]
    for i in range(len(camera_pos_array)):
        current_pos = camera_pos_array[i]
        baseline = MathTools.get_baseline(current_pos[0], current_pos[1], center_pos[0], center_pos[1])
        if 0 < baseline < min_baseline:
            min_baseline = baseline

        baselines[i] = baseline

    for i in range(len(camera_pos_array)):
        baseline_ratios[i] = baselines[i]/min_baseline
        disp_to_center[i] = round(IGEV_SEARCH_RANGE * baseline_ratios[i])
        disp_to_center_offset[i] = round(IGEV_OFFSET * baseline_ratios[i])
    return disp_to_center, disp_to_center_offset, baseline_ratios

def get_depth_map(camera_pos_array, ms_image_calibrated, disp_to_center, disp_to_center_offset, baseline_ratios, verbose=False):
    img_left = ms_image_calibrated[:, len(camera_pos_array)//2, ...].unsqueeze(dim=1)
    pos_left = camera_pos_array[len(camera_pos_array)//2]
    B, C, H, W = ms_image_calibrated.shape
    disp_maps = ms_image_calibrated.new_zeros((B, len(DEPTH_ESTIMATION_CAMS), H, W))
    for i in range(len(DEPTH_ESTIMATION_CAMS)):
        cam_idx = DEPTH_ESTIMATION_CAMS[i]
        img_right = ms_image_calibrated[:, cam_idx, :, :].unsqueeze(dim=1)
        pos_right = camera_pos_array[cam_idx]
        angle = -MathTools.get_angle(pos_left[0], pos_left[1], pos_right[0], pos_right[1])
        search_range = int(math.ceil(disp_to_center[cam_idx]/DIV) * DIV)
        disp_maps[:, i, :, :] = estimate_disparity(img_left, img_right, angle, search_range, int(disp_to_center_offset[cam_idx]), verbose=verbose)
        disp_maps[:, i, :, :] /= baseline_ratios[cam_idx]

    return torch.median(disp_maps, dim=1, keepdim=True)[0]

def estimate_disparity(img_left, img_right, angle, search_range, offset, verbose=False):
    if verbose:
        print("Search range %i" % search_range)
    angle_deg = angle/math.pi * 180

    B, C, H, W = img_left.shape

    img_left = VF.rotate(img_left, angle_deg, interpolation=transforms.InterpolationMode.BILINEAR, expand=True)
    img_right = VF.rotate(img_right, angle_deg, interpolation=transforms.InterpolationMode.BILINEAR, expand=True)

    img_left = F.pad(img_left, (offset, 0, 0, 0), 'replicate')
    img_right = F.pad(img_right, (0, offset, 0, 0), 'replicate')

    patches_left, patches_right, patches_pos_data, weighting_functions = extract_patches(img_left, img_right, search_range)
    cost_patches = []
    for i in range(len(patches_left)):
        cost_patch = Inference.run(search_range, patches_left[i], patches_right[i], verbose=verbose)
        cost_patches.append(cost_patch)

    B_rot, C_rot, H_rot, W_rot = img_left.shape
    disparity = img_left.new_zeros((B_rot, H_rot, W_rot))
    cost_weights = img_left.new_zeros((B_rot, H_rot, W_rot))
    for i in range(len(cost_patches)):
        patch_x_start_padded, patch_x_end_padded, patch_y_start_padded, patch_y_end_padded = patches_pos_data[i]
        disparity[:, patch_y_start_padded:patch_y_end_padded, patch_x_start_padded:patch_x_end_padded] += cost_patches[i] * weighting_functions[i]
        cost_weights[:, patch_y_start_padded:patch_y_end_padded, patch_x_start_padded:patch_x_end_padded] += weighting_functions[i]

    disparity = disparity/cost_weights

    disparity = disparity[:, :, offset:]
    disparity -= offset

    disparity = VF.rotate(disparity, -angle_deg, interpolation=transforms.InterpolationMode.BILINEAR, expand=True)
    offset_y = (disparity.shape[1] - H)//2
    offset_x = (disparity.shape[2] - W)//2
    disparity = disparity[:, offset_y:(offset_y + H), offset_x:(offset_x + W)]

    return disparity

def extract_patches(img_left, img_right, search_range):
    B, C, H, W = img_left.shape
    num_patches_x = int(math.ceil(W/IGEV_PATCH_SIZE))
    num_patches_y = int(math.ceil(H/IGEV_PATCH_SIZE))

    patches_left = []
    patches_right = []
    patches_pos_data = []
    weighting_functions = []

    for x in range(num_patches_x):
        for y in range(num_patches_y):
            patch_x_start = x * IGEV_PATCH_SIZE
            patch_x_end = min((x + 1) * IGEV_PATCH_SIZE, W)
            patch_y_start = y * IGEV_PATCH_SIZE
            patch_y_end = min((y + 1) * IGEV_PATCH_SIZE, H)

            crop_left = 0
            crop_top = 0

            patch_x_start_padded = patch_x_start - search_range
            if patch_x_start_padded < 0:
                crop_left = abs(patch_x_start_padded)
                patch_x_start_padded = 0

            patch_x_end_padded = patch_x_end + search_range
            if patch_x_end_padded > W:
                patch_x_end_padded = W

            patch_y_start_padded = patch_y_start - IGEV_PATCH_PADDING
            if patch_y_start_padded < 0:
                crop_top = abs(patch_y_start_padded)
                patch_y_start_padded = 0

            patch_y_end_padded = patch_y_end + IGEV_PATCH_PADDING
            if patch_y_end_padded > H:
                patch_y_end_padded = H

            weighting_width = search_range + IGEV_PATCH_SIZE + search_range
            weighting_height = IGEV_PATCH_PADDING + IGEV_PATCH_SIZE + IGEV_PATCH_PADDING

            weighting = img_left.new_zeros((weighting_height, weighting_width))
            weighting = calc_weighting(weighting, search_range)
            weighting = weighting[crop_top:, crop_left:]
            weighting = weighting[:patch_y_end_padded - patch_y_start_padded, :patch_x_end_padded - patch_x_start_padded]

            patches_left.append(img_left[:, :, patch_y_start_padded:patch_y_end_padded, patch_x_start_padded:patch_x_end_padded])
            patches_right.append(img_right[:, :, patch_y_start_padded:patch_y_end_padded, patch_x_start_padded:patch_x_end_padded])
            patches_pos_data.append([patch_x_start_padded, patch_x_end_padded, patch_y_start_padded, patch_y_end_padded])
            weighting_functions.append(weighting)

    return patches_left, patches_right, patches_pos_data, weighting_functions

def calc_weighting(weighting, search_range):
    weighting[IGEV_PATCH_PADDING:IGEV_PATCH_PADDING + IGEV_PATCH_SIZE, search_range:search_range + IGEV_PATCH_SIZE] = 1

    #weight edges
    edge_base_sr = torch.arange(search_range, device=weighting.device) + 1
    edge_base_sr = torch.tile(edge_base_sr[None, :], (IGEV_PATCH_SIZE, 1))
    edge_base_sr = IGEV_WEIGHTING_RHO ** edge_base_sr
    weighting[IGEV_PATCH_PADDING:-IGEV_PATCH_PADDING, :search_range] = torch.flip(edge_base_sr, dims=[1])
    weighting[IGEV_PATCH_PADDING:-IGEV_PATCH_PADDING, -search_range:] = edge_base_sr

    edge_base_pd = torch.arange(IGEV_PATCH_PADDING, device=weighting.device) + 1
    edge_base_pd = torch.tile(edge_base_pd[:, None], (1, IGEV_PATCH_SIZE))
    edge_base_pd = IGEV_WEIGHTING_RHO ** edge_base_pd
    weighting[:IGEV_PATCH_PADDING, search_range:-search_range] = torch.flip(edge_base_pd, dims=[0])
    weighting[-IGEV_PATCH_PADDING:, search_range:-search_range] = edge_base_pd

    #weight corners
    corner_base_x = torch.arange(search_range, device=weighting.device) + 1
    corner_base_x = torch.tile(corner_base_x[None, :], (IGEV_PATCH_PADDING, 1))
    corner_base_y = torch.arange(IGEV_PATCH_PADDING, device=weighting.device) + 1
    corner_base_y = torch.tile(corner_base_y[:, None], (1, search_range))
    corner_base = torch.sqrt(corner_base_x ** 2 + corner_base_y ** 2)
    corner_base = IGEV_WEIGHTING_RHO ** corner_base

    weighting[-IGEV_PATCH_PADDING:, -search_range:] = corner_base
    weighting[:IGEV_PATCH_PADDING, -search_range:] = torch.flip(corner_base, dims=[0])
    weighting[-IGEV_PATCH_PADDING:, :search_range] = torch.flip(corner_base, dims=[1])
    weighting[:IGEV_PATCH_PADDING, :search_range] = torch.flip(corner_base, dims=[0, 1])

    return weighting
