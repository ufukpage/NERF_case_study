"""Produces `nerf_output.png`."""
import cv2
import numpy as np
import torch
import torch.nn as nn

import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
from dataset import CustomDataset
from model import NeRF
from nerf_utils import batchify_rays

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = CustomDataset("D:\\nerf_synthetic/lego", 'test',
                        img_wh=(400, 400))

N_samples = 64
N_importance = 64
nerf_coarse = NeRF(3, 10, d_viewdirs=4)
nerf_fine = NeRF(3, 10, d_viewdirs=4)
models = [nerf_coarse, nerf_fine]
ckpt_path = 'lego.ckpt'


def load_checkpoint_custom(model, ckpt_path, model_name):
    state_dct = torch.load(ckpt_path, map_location=torch.device('cpu'))
    state_dct = {k: v for k, v in state_dct.items() if model_name in k}
    model_dct = model.state_dict()
    checkpoint_ = {k: v for (k, _), (_, v) in zip(model_dct.items(), state_dct.items())}
    model.load_state_dict(checkpoint_)


def decode_batch(batch):
    rays = batch['rays']  # (B, 8)
    rgbs = batch['rgbs']  # (B, 3)
    return rays, rgbs


def visualize_depth(depth, cmap=cv2.COLORMAP_JET):
    x = depth.cpu().numpy()
    x = np.nan_to_num(x)
    mi = np.min(x)
    ma = np.max(x)
    x = (x - mi) / (ma - mi + 1e-8)
    x = (255 * x).astype(np.uint8)
    x_ = Image.fromarray(cv2.applyColorMap(x, cmap))
    x_ = T.ToTensor()(x_)  # (3, H, W)
    return x_


def render(rays, chunk=1024 * 32, **kwargs):
    """Render rays
       Args:

         chunk: int, Maximum number of rays to process simultaneously. Used to
           control maximum memory usage. Does not affect final results.
         rays: (N_rays, 11), Ray origin and direction

       Returns:
         rgb_map: [batch_size, 3]. Predicted RGB values for rays.
         disp_map: [batch_size]. Disparity map. Inverse of depth.
         acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
         extras: dict with everything returned by render_rays().
       """
    # viewing direction
    rays_d = rays[:, 3:6]
    viewdirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    viewdirs = torch.reshape(viewdirs, [-1, 3]).float()
    rays = torch.cat([rays, viewdirs], -1)
    rays = rays.to(device)

    all_ret = batchify_rays(rays, chunk, **kwargs)
    sh = rays_d.shape # [..., 3]
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)
    return all_ret


if __name__ == "__main__":
    load_checkpoint_custom(nerf_coarse, ckpt_path, model_name='nerf_coarse')
    load_checkpoint_custom(nerf_fine, ckpt_path, model_name='nerf_fine')

    rays, rgbs = decode_batch(dataset[10])
    img_gt = rgbs.view(400, 400, 3)  # Ground truth image

    #  TODO: You need to add your rendering implementation right here
    #  As you can see, it should retrieve a results dict with rgb, opacity and
    #  depth information from the fine model. RGB is enough, but opacity and
    #  depth should be intuitive. Note that rendering may take 5+ mins on a CPU.

    models = [nn.DataParallel(model).to(device) for model in models]

    with torch.no_grad():
        results = render(rays, N_samples=N_samples, N_importance=N_importance, models=models)

    #""""
    img_pred = results['rgb_fine'].view(400, 400, 3).cpu().numpy()
    alpha_pred = results['opacity_fine'].view(400, 400).cpu().numpy()
    depth_pred = results['depth_fine'].view(400,400)

    plt.subplots(figsize=(12, 8))
    plt.tight_layout()
    plt.subplot(221)
    plt.title('GT')
    plt.imshow(img_gt)
    plt.subplot(222)
    plt.title('pred')
    plt.imshow(img_pred)
    plt.subplot(223)
    plt.title('alpha')
    plt.imshow(alpha_pred)
    plt.subplot(224)
    plt.title('depth')
    plt.imshow(visualize_depth(depth_pred).permute(1,2,0))
    plt.show()
    #"""
