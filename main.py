"""Produces `nerf_output.png`."""
import torch
import torchvision.transforms as T
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from model import NeRF
from dataset import CustomDataset

dataset = CustomDataset("nerf_synthetic/lego", 'test',
                         img_wh=(400, 400))

N_samples = 64
N_importance = 64
nerf_coarse = NeRF(3,10, d_viewdirs=4)
nerf_fine = NeRF(3,10, d_viewdirs=4)
models = [nerf_coarse, nerf_fine]
ckpt_path = 'lego.ckpt'

def load_checkpoint_custom(model, ckpt_path, model_name):
    state_dct = torch.load(ckpt_path, map_location=torch.device('cpu'))
    state_dct = {k:v for k,v in state_dct.items() if model_name in k}
    model_dct = model.state_dict()
    checkpoint_= {k:v for (k, _), (_, v) in zip(model_dct.items(), state_dct.items())}
    model.load_state_dict(checkpoint_)


def decode_batch(batch):
    rays = batch['rays'] # (B, 8)
    rgbs = batch['rgbs'] # (B, 3)
    return rays, rgbs

def visualize_depth(depth, cmap=cv2.COLORMAP_JET):
    x = depth.cpu().numpy()
    x = np.nan_to_num(x)
    mi = np.min(x)
    ma = np.max(x)
    x = (x-mi)/(ma-mi+1e-8)
    x = (255*x).astype(np.uint8)
    x_ = Image.fromarray(cv2.applyColorMap(x, cmap))
    x_ = T.ToTensor()(x_) # (3, H, W)
    return x_


if __name__ == "__main__":
    load_checkpoint_custom(nerf_coarse, ckpt_path, model_name='nerf_coarse')
    load_checkpoint_custom(nerf_fine, ckpt_path, model_name='nerf_fine')

    rays, rgbs = decode_batch(dataset[10])
    img_gt = rgbs.view(400, 400, 3) # Ground truth image
    
    # TODO: You need to add your rendering implementation right here
    # As you can see, it should retrieve a results dict with rgb, opacity and 
    # depth information from the fine model. RGB is enough, but opacity and 
    # depth should be intuitive. Note that rendering may take 5+ mins on a CPU.
    
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
