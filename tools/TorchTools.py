import torch

def calibrate(img, calibration_matrix):
    torch_horizontal = torch.arange(img.size(3), device=img.device, dtype=torch.float).view(1, 1, 1, img.size(3)).expand(img.size(0), 1, img.size(2), img.size(3))
    torch_vertical = torch.arange(img.size(2), device=img.device, dtype=torch.float).view(1, 1, img.size(2), 1).expand(img.size(0), 1, img.size(2), img.size(3))

    inv_calib = torch.linalg.inv(calibration_matrix)
    denom = inv_calib[2, 0] * torch_horizontal + inv_calib[2, 1] * torch_vertical + inv_calib[2, 2]
    torch_horizontal = (inv_calib[0, 0] * torch_horizontal + inv_calib[0, 1] * torch_vertical + inv_calib[0, 2]) / denom
    torch_vertical = (inv_calib[1, 0] * torch_horizontal + inv_calib[1, 1] * torch_vertical + inv_calib[1, 2]) / denom
    tensor_grid = torch.cat([2 * torch_horizontal/(img.size(3) - 1) - 1, 2 * torch_vertical/(img.size(2) - 1) - 1], dim=1)

    return torch.nn.functional.grid_sample(input=img, grid=tensor_grid.permute(0, 2, 3, 1), mode='bilinear', padding_mode='border', align_corners=True)