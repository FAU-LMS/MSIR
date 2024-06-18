import math
import torch
import torchvision.transforms.functional as VF
import torchvision.transforms as transforms
import time
import matplotlib.pyplot as plt

from tools import MathTools

LAYER_THRESHOLD = 0.75
DISP_DIFF_THRESHOLD = 0.5
ADD_MASK_PIXELS = 6

def run(ms_image_raw, depth_map, calibration_matrices, camera_pos_array):
    center_cam_idx = ms_image_raw.shape[1]//2
    baselines = []
    angles = []
    min_baseline = 10000000000
    center_pos = camera_pos_array[center_cam_idx]
    for i in range(ms_image_raw.shape[1]):
        current_pos = camera_pos_array[i]
        baseline = MathTools.get_baseline(current_pos[0], current_pos[1], center_pos[0], center_pos[1])
        angle = MathTools.get_angle(current_pos[0], current_pos[1], center_pos[0], center_pos[1])
        if baseline == 0:
            baseline = 10000000000

        baselines.append(baseline)
        angles.append(angle)

        if baseline < min_baseline:
            min_baseline = baseline

    B, C, H, W = ms_image_raw.shape
    ms_image_masked = ms_image_raw.new_zeros([B, C, H, W])
    ms_image_masked[:, center_cam_idx, :] = ms_image_raw[:, center_cam_idx, :]
    ms_mask = ms_image_raw.new_zeros([B, C, H, W])
    ms_mask[:, center_cam_idx, :] = 1
    for i in range(ms_image_raw.shape[1]):
    #for i in range(3, 4):
        #Skip center cam
        if i == center_cam_idx:
            continue

        baseline_ratio = baselines[i]/min_baseline
        image, mask = warp_cam(ms_image_raw[:, i, :, :][:, None], depth_map, calibration_matrices[i], -angles[i], baseline_ratio)
        ms_image_masked[:, i, :, :] = image
        ms_mask[:, i, :, :] = mask

    return ms_image_masked * ms_mask, ms_mask

def warp_cam(img, disp, calibration_matrix, angle, baseline_ratio):
    dX = math.cos(angle)
    dY = math.sin(angle)

    torch_horizontal = torch.arange(img.size(3), device=img.device, dtype=torch.float).view(1, 1, 1, img.size(3)).expand(img.size(0), 1, img.size(2), img.size(3))
    torch_vertical = torch.arange(img.size(2), device=img.device, dtype=torch.float).view(1, 1, img.size(2), 1).expand(img.size(0), 1, img.size(2), img.size(3))

    inv_calib = torch.linalg.inv(calibration_matrix)
    denom = inv_calib[2, 0] * torch_horizontal + inv_calib[2, 1] * torch_vertical + inv_calib[2, 2]
    torch_horizontal = (inv_calib[0, 0] * torch_horizontal + inv_calib[0, 1] * torch_vertical + inv_calib[0, 2]) / denom + dX * disp * baseline_ratio
    torch_vertical = (inv_calib[1, 0] * torch_horizontal + inv_calib[1, 1] * torch_vertical + inv_calib[1, 2]) / denom + dY * disp * baseline_ratio
    mask = (torch_horizontal > -0.5) * (torch_horizontal < img.size(3) - 0.5) * (torch_vertical > -0.5) * (torch_vertical < img.size(2) - 0.5)
    tensor_grid = torch.cat([2 * torch_horizontal/(img.size(3) - 1) - 1, 2 * torch_vertical/(img.size(2) - 1) - 1], dim=1)

    img_warped = torch.nn.functional.grid_sample(input=img, grid=tensor_grid.permute(0, 2, 3, 1), mode='bilinear', padding_mode='border', align_corners=True)
    mask[mask == 1] = get_mask(disp, angle)[mask == 1]

    B, C, H, W = mask.shape
    for shift in range(1, ADD_MASK_PIXELS):
        shift_y = round(dY * shift)
        shift_x = round(dX * shift)
        shift_start_y = max(shift_y, 0)
        shift_start_x = max(shift_x, 0)
        shift_end_y = min(shift_y + mask.shape[2], mask.shape[2])
        shift_end_x = min(shift_x + mask.shape[3], mask.shape[3])
        orig_start_y = max(-shift_y, 0)
        orig_start_x = max(-shift_x, 0)
        orig_end_y = min(-shift_y + mask.shape[2], mask.shape[2])
        orig_end_x = min(-shift_x + mask.shape[3], mask.shape[3])
        shifted_mask = mask.new_zeros([B, C, H, W])
        shifted_mask[:, :, shift_start_y:shift_end_y, shift_start_x:shift_end_x] = mask[:, :, orig_start_y:orig_end_y, orig_start_x:orig_end_x]
        mask[mask == 1] = shifted_mask[mask == 1]
    return img_warped, mask

def get_mask(disp, angle):
    dX = math.cos(angle + math.pi)
    dX = int(round(dX))
    dY = math.sin(angle + math.pi)
    dY = int(round(dY))
    B, C, H, W = disp.shape
    disp = disp - torch.min(disp) + 1
    start_layer = torch.floor(torch.min(disp)).long()
    end_layer = torch.ceil(torch.max(disp)).long()
    max = disp.new_zeros([B, C, H, W])
    mask = disp.new_ones([B, C, H, W], dtype=torch.bool)
    #start = time.time()
    for d in range(end_layer, start_layer - 1, -1):
        depth_mask = (d - LAYER_THRESHOLD < disp) * (disp <= d + LAYER_THRESHOLD)
        start_x = dX * d
        start_y = dY * d
        max_start_x = -start_x
        max_start_y = -start_y
        end_x = disp.shape[3]
        end_y = disp.shape[2]
        max_end_x = disp.shape[3]
        max_end_y = disp.shape[2]
        if start_x < 0:
            end_x += start_x
            start_x = 0
        if start_y < 0:
            end_y += start_y
            start_y = 0
        if max_start_x < 0:
            max_end_x += max_start_x
            max_start_x = 0
        if max_start_y < 0:
            max_end_y += max_start_y
            max_start_y = 0

        translated_depth = disp[:, :, start_y:end_y, start_x:end_x]
        translated_depth_mask = depth_mask[:, :, start_y:end_y, start_x:end_x]

        max_mask = (max[:, :, max_start_y:max_end_y, max_start_x:max_end_x] == 0) * translated_depth_mask
        max[:, :, max_start_y:max_end_y, max_start_x:max_end_x][max_mask] = translated_depth[max_mask]

        diff_mask = translated_depth < max[:, :, max_start_y:max_end_y, max_start_x:max_end_x] - DISP_DIFF_THRESHOLD
        mask_mask = translated_depth_mask * diff_mask
        mask[:, :, start_y:end_y, start_x:end_x][mask_mask] = 0

    return mask

def get_mask_rotation(disp, angle):
    angle_deg = angle/math.pi * 180 + 180
    rotated_disp = VF.rotate(disp, angle_deg, interpolation=transforms.InterpolationMode.NEAREST, expand=True)
    rotated_disp = rotated_disp - torch.min(rotated_disp) + 1

    B, C, H, W = rotated_disp.shape
    start_layer = torch.floor(torch.min(rotated_disp)).long()
    end_layer = torch.ceil(torch.max(rotated_disp)).long()
    max = rotated_disp.new_zeros([B, C, H, W])
    mask = rotated_disp.new_ones([B, C, H, W], dtype=torch.bool)
    for d in range(end_layer, start_layer - 1, -1):
        depth_mask = (d - LAYER_THRESHOLD < rotated_disp) * (rotated_disp <= d + LAYER_THRESHOLD)
        translated_depth = rotated_disp[:, :, :, d:]
        translated_depth_mask = depth_mask[:, :, :, d:]

        max_mask = (max[:, :, :, :-d] == 0) * translated_depth_mask
        max[:, :, :, :-d][max_mask] = translated_depth[max_mask]

        diff_mask = translated_depth < max[:, :, :, :-d] - DISP_DIFF_THRESHOLD
        mask_mask = translated_depth_mask * diff_mask
        mask[:, :, :, d:][mask_mask] = 0

    mask = VF.rotate(mask, -angle_deg, interpolation=transforms.InterpolationMode.NEAREST, expand=True)
    offset_y = (mask.shape[2] - disp.shape[2])//2
    offset_x = (mask.shape[3] - disp.shape[3])//2
    mask = mask[:, :, offset_y:(offset_y + disp.shape[2]), offset_x:(offset_x + disp.shape[3])]

    return mask
