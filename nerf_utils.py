# All functions are based on https://github.com/yenchenlin/nerf-pytorch/
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

DEBUG = True


def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn

    def ret(pts, dirs):
        return torch.cat([fn(pts[i:i + chunk], dirs[i:i + chunk]) for i in range(0, pts.shape[0], chunk)], 0)

    return ret


def nerf_forward(pts, viewdirs, network, netchunk=1024 * 64):
    inputs_flat = torch.reshape(pts, [-1, pts.shape[-1]])
    input_dirs = viewdirs[:, None].expand(pts.shape)
    input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])

    outputs_flat = batchify(network, netchunk)(inputs_flat, input_dirs_flat)
    return torch.reshape(outputs_flat, list(pts.shape[:-1]) + [outputs_flat.shape[-1]])


def volumetric_render(raw, z_vals, rays_d):
    """
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1. - torch.exp(-act_fn(raw) * dists)

    dists = z_vals[..., 1:] - z_vals[..., :-1] # distance between adjacent samples
    dists = torch.cat([dists, torch.Tensor([1e10]).to(dists.device)
                      .expand(dists[..., :1].shape)], -1)  # [N_rays, N_samples] to match dims

    dists = dists * torch.norm(rays_d[..., None, :], dim=-1) # give direction to distances

    rgb = torch.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3] # get rgb values for each 3D point

    dists = dists.to(raw.device)
    density = raw2alpha(raw[..., 3], dists)  # [N_rays, N_samples] # get alpha value or DENSITY for each 3D point

    # we use minus 1 to precent effect of 1 in density variabla
    # cumprod provides us transparancy and volume summation together
    transparancy = torch.cumprod(torch.cat(
        [torch.ones((density.shape[0], 1), device=density.device), 1. - density + 1e-10], -1), -1)
    weights = density * transparancy[:, :-1]
    rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]

    z_vals = z_vals.to(weights.device)
    depth_map = torch.sum(weights * z_vals, -1)
    acc_map = torch.sum(weights, -1)

    rgb_map = rgb_map + (1. - acc_map[..., None])

    return rgb_map, acc_map, weights, depth_map


# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False):
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Invert CDF
    u = u.contiguous().to(cdf.device)
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape).to(inds_g.device), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


def render_rays(ray_batch,
                models,
                N_samples=64,
                N_importance=64,
                retraw=False,
                lindisp=False,
                perturb=0.,
                pytest=False):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      N_samples: int. Number of different times to sample along each ray.

      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
    """
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  # [N_rays, 3] each
    viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[..., 6:8], [-1, 1, 2])
    near, far = bounds[..., 0], bounds[..., 1]  # [-1,1]
    device = ray_batch.device
    t_vals = torch.linspace(0., 1., steps=N_samples).to(device)
    # sample coarse depth points
    if not lindisp:
        z_vals = near * (1. - t_vals) + far * (t_vals)
    else:
        z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * (t_vals))

    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.Tensor(t_rand)

        z_vals = lower + (upper - lower) * t_rand

    # generate 3D points
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples, 3]

    raw = nerf_forward(pts, viewdirs, models[0])

    z_vals = z_vals.to(raw.device)
    rgb_map, acc_map, weights, depth_map = volumetric_render(raw, z_vals, rays_d)

    # importance sampling
    if N_importance > 0:

        z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        # sample fine depth points
        z_samples = sample_pdf(z_vals_mid, weights[..., 1:-1], N_importance, det=(perturb == 0.), pytest=pytest)
        z_samples = z_samples.detach()

        # gather coarse and fine depth points and sort
        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)

        # sample new points with respect to new z_vals
        rays_d = rays_d.to(z_vals.device)
        rays_o = rays_o.to(z_vals.device)
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples + N_importance, 3]

        raw = nerf_forward(pts, viewdirs, models[1])

        rgb_map, acc_map, weights, depth_map = volumetric_render(raw, z_vals, rays_d)

    ret = {'rgb_fine': rgb_map, 'depth_fine': depth_map, 'opacity_fine': acc_map}
    if retraw:
        ret['raw'] = raw

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")
    return ret


def batchify_rays(rays_flat, chunk=1024 * 32, **kwargs):
    """Render rays in smaller minibatches to avoid Out-Of-Memory.
    """
    all_ret = {}
    for i in tqdm(range(0, rays_flat.shape[0], chunk)):
        ret = render_rays(rays_flat[i:i + chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret