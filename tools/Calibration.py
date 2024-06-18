import cv2 as cv
import numpy as np
import torch

def run(ms_image, cols, rows):
    ms_image_uint8 = ms_image[0].numpy().copy()
    ms_image_uint8 = ms_image_uint8 * 255
    ms_image_uint8 = ms_image_uint8.astype(np.uint8)
    center_cam = ms_image_uint8[ms_image_uint8.shape[0]//2, :, :]

    print("Calibrating with " + str(cols) + " cols and " + str(rows) + " rows!")

    center_corners = find_corners(center_cam, cols, rows)

    result_matrices = []
    for i in range(ms_image_uint8.shape[0]):
        corners = find_corners(ms_image_uint8[i, :, :].copy(), cols, rows)

        matrix_affine, inliers = cv.estimateAffine2D(corners, center_corners.copy())
        matrix = np.concatenate((matrix_affine, np.array([[0, 0, 1]])), axis=0)
        result_matrices.append(matrix)

    return torch.tensor(np.array(result_matrices))

def find_corners(image, cols, rows):
    ret, corners = cv.findChessboardCorners(image, (cols - 1, rows - 1), None)
    if not ret:
        print("Chessboard not found")
        return None

    cv.cornerSubPix(image, corners, (11, 11), (-1, -1), (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 1000, 0.01))

    return corners
    
def run_corners_only(corners):
    result_matrices = []
    
    center_corners = corners[4]
    
    for i in range(len(corners)):
        matrix_affine, inliers = cv.estimateAffine2D(corners[i], center_corners.copy())
        matrix = np.concatenate((matrix_affine, np.array([[0, 0, 1]])), axis=0)
        result_matrices.append(matrix)

    return torch.tensor(np.array(result_matrices))
