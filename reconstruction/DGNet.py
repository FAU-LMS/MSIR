import torch
import torch.nn.functional as F
import pathlib

from .model import DGNet
from .option import args

model = None


def load_model(path='', gpu=True, verbose=False):
    global model
    if model is None:
        model = DGNet(args)
    if path == '':
        path = pathlib.Path(__file__).parent.resolve().__str__() + '/model/model.pt'
    if verbose:
        print(f'[**] Loading DGNet network from {path}')
    if torch.cuda.is_available() and gpu:
        torch.backends.cudnn.benchmark = True  # cudnn auto-tuner
        dict = torch.load(path, map_location='cuda')
        model.load_state_dict(dict)
        model = model.cuda()
    else:
        dict = torch.load(path, map_location='cpu')
        model.load_state_dict(dict)


def predict(ref_image, masked_image, mask):
    global model
    norm_ref_image = ref_image
    norm_masked_image = masked_image
    norm_masked_image = (norm_masked_image * mask) + (1 - mask)

    dividable = 2 ** args.num_down_layers
    pad_y, pad_x = 0, 0
    if norm_ref_image.shape[2] % dividable > 0:
        pad_y = dividable - norm_ref_image.shape[2] % dividable
    if norm_ref_image.shape[3] % dividable > 0:
        pad_x = dividable - norm_ref_image.shape[3] % dividable

    norm_ref_image = F.pad(norm_ref_image, (0, pad_x, 0, pad_y), mode='replicate')
    norm_masked_image = F.pad(norm_masked_image, (0, pad_x, 0, pad_y), mode='replicate')
    norm_mask = F.pad(mask, (0, pad_x, 0, pad_y), mode='replicate')

    model.eval()
    with torch.no_grad():
        out = model(norm_ref_image, norm_masked_image, norm_mask)

    out = (out * (1 - norm_mask)) + (norm_masked_image * norm_mask)

    if pad_y > 0:
        out = out[:, :, :-pad_y, :]
    if pad_x > 0:
        out = out[:, :, :, :-pad_x]
    out[out > 1] = 1
    out[out < 0] = 0
    return out


def run(ms_image_warped, ms_mask):
    # Channels of images are batch now
    ms_image_warped = torch.transpose(ms_image_warped, 0, 1)
    ms_mask = torch.transpose(ms_mask, 0, 1)
    channels = ms_image_warped.shape[0]
    ref_image = ms_image_warped[channels // 2]
    ms_image = predict(torch.tile(ref_image[None, :], dims=(channels, 1, 1, 1)), ms_image_warped, ms_mask)
    return torch.transpose(ms_image, 0, 1)
