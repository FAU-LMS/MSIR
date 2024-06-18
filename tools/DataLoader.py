import torch
import numpy as np
import imageio
import os
import json

def load_record(name):
    base_path = f'data/{name}/'
    camera_pos_array = load_camera_array(base_path)

    channels = len(camera_pos_array)
    ms_image_raw = None
    for i in range(channels):
        image = imageio.v3.imread(base_path + 'scene/Cam-' + str(i) + '.png')
        if ms_image_raw is None:
            ms_image_raw = torch.zeros((1, channels, image.shape[0], image.shape[1]))

        ms_image_raw[0, i, :, :] = torch.from_numpy(image.copy()/65535.)

    calibration_images = None
    for i in range(channels):
        if not os.path.exists(base_path + 'calibration/Cam-' + str(i) + '.png'):
            break
        image = imageio.v3.imread(base_path + 'calibration/Cam-' + str(i) + '.png')
        if calibration_images is None:
            calibration_images = torch.zeros((1, channels, image.shape[0], image.shape[1]))

        calibration_images[0, i, :, :] = torch.from_numpy(image.copy()/65535.)

    return ms_image_raw, calibration_images, camera_pos_array


def load_camera_array(name, file='data/camera_array.json'):
    if os.path.isfile(file):
        with open(file) as f:
            data = json.load(f)
            camera_array = np.zeros((len(data), 2))
            for i in range(len(data)):
                pos_json = data['Camera ' + str(i)]
                pos_x = pos_json['X']
                pos_y = pos_json['Y']
                camera_array[i, 0] = pos_x / 1000
                camera_array[i, 1] = pos_y / 1000

        return camera_array
    else:
        print("camera_array.json not found!")
        quit()
        
def load_calibration_corners(name):
    return np.load(f'data/{name}/calibration/corners.npy')
