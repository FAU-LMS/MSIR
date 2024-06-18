import os
import torch
import torch.nn.parallel
import torch.nn.functional as F
import argparse

from .igev_stereo import IGEVStereo

model_dict = {}

def load_args(max_disp):
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='igev-stereo', help="name your experiment")
    parser.add_argument('--restore_ckpt', default=None, help="")
    parser.add_argument('--mixed_precision', default=True, action='store_true', help='use mixed precision')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=4, help="batch size used during training.")
    parser.add_argument('--train_datasets', nargs='+', default=['sceneflow'], help="training datasets.")
    parser.add_argument('--lr', type=float, default=0.0002, help="max learning rate.")
    parser.add_argument('--num_steps', type=int, default=200000, help="length of training schedule.")
    parser.add_argument('--image_size', type=int, nargs='+', default=[320, 736], help="size of the random image crops used during training.")
    parser.add_argument('--train_iters', type=int, default=22, help="number of updates to the disparity field in each forward pass.")
    parser.add_argument('--wdecay', type=float, default=.00001, help="Weight decay in optimizer.")

    # Validation parameters
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during validation forward pass')

    # Architecure choices
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=2, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
    parser.add_argument('--max_disp', type=int, default=max_disp, help="max disp of geometry encoding volume")

    # Data augmentation
    parser.add_argument('--do_flip', default=False, choices=['h', 'v'], help='flip the images horizontally or vertically')
    parser.add_argument('--spatial_scale', type=float, nargs='+', default=[-0.2, 0.4], help='re-scale the images randomly')
    parser.add_argument('--noyjitter', action='store_true', help='don\'t simulate imperfect rectification')
    args = parser.parse_args()
    return args

def load_model(model_path, max_disp, gpu=True, verbose=False):
    global model_dict
    if max_disp in model_dict.keys():
        return
    else:
        model = torch.nn.DataParallel(IGEVStereo(load_args(max_disp)))
        if gpu:
            model = model.cuda()
        else:
            model = model.cpu()

        if os.path.isfile(model_path):
            if verbose:
                print(f'[**] Loading igev network from {model_path}')
            state_dict = torch.load(model_path) #, map_location=torch.device('cpu')
            model.load_state_dict(state_dict)
            model = model.module
        else:
            print("Model not found!")

        model_dict[max_disp] = model

def run(max_disp, img_l, img_r, iters=12, verbose=False):
    global model_dict

    if verbose:
        print("Inference (IGEV Stereo) run with patch size " + str(img_l.shape) + " and max disp " + str(max_disp))

    if not max_disp in model_dict:
        print("Did you forget to call load_models?")
        quit()
    model = model_dict[max_disp]

    dividable = 2 ** 5
    pad_y, pad_x = 0, 0
    if img_l.shape[2] % dividable > 0:
        pad_y = dividable - img_l.shape[2] % dividable
    if img_l.shape[3] % dividable > 0:
        pad_x = dividable - img_l.shape[3] % dividable
    img_l = F.pad(img_l, (0, pad_x, 0, pad_y), mode='replicate')
    img_r = F.pad(img_r, (0, pad_x, 0, pad_y), mode='replicate')

    img_l = torch.tile(img_l, (1, 3, 1, 1))
    img_r = torch.tile(img_r, (1, 3, 1, 1))

    model.eval()
    with torch.no_grad():
        disparity = model(img_l, img_r, iters=iters, test_mode=True)[0]

    if pad_y > 0:
        disparity = disparity[:, :-pad_y, :]
    if pad_x > 0:
        disparity = disparity[:, :, :-pad_x]

    return disparity
