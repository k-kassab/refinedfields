import math
import torch
import numpy as np

def psnr(rgb, gt):
    """Calculates the PSNR metric

    Args:
        rgb (torch.Tensor): Image tensor of shape [H,W,3] \in [0,1]
        gt (torch.Tensor): Ground truth image tensor of shape [H,W,3] \in [0,1]

    Returns:
        (float): The PSNR score
    """
    try:
        assert (rgb.max() <= 1.05 and rgb.min() >= -0.05)
    except:
        print(f"rgb values exceeded limits: [{rgb.min(), rgb.max()}] not in [-0.05, 1.05]")
    assert (gt.max() <= 1.05 and gt.min() >= -0.05)
    assert (rgb.shape[-1] == 3)
    assert (gt.shape[-1] == 3)

    mse = torch.mean((rgb[..., :3] - gt[..., :3]) ** 2).item()
    return 10 * math.log10(1.0 / mse)

def ssim(rgb, gts):
    """
    Modified from https://github.com/google/mipnerf/blob/16e73dfdb52044dcceb47cda5243a686391a6e0f/internal/math.py#L58
    """
    filter_size = 11
    filter_sigma = 1.5
    k1 = 0.01
    k2 = 0.03
    max_val = 1.0
    rgb = rgb.cpu().numpy()
    gts = gts.cpu().numpy()
    assert len(rgb.shape) == 3
    assert rgb.shape[-1] == 3
    assert rgb.shape == gts.shape
    import scipy.signal

    # Construct a 1D Gaussian blur filter.
    hw = filter_size // 2
    shift = (2 * hw - filter_size + 1) / 2
    f_i = ((np.arange(filter_size) - hw + shift) / filter_sigma)**2
    filt = np.exp(-0.5 * f_i)
    filt /= np.sum(filt)

    # Blur in x and y (faster than the 2D convolution).
    def convolve2d(z, f):
        return scipy.signal.convolve2d(z, f, mode='valid')

    filt_fn = lambda z: np.stack([
        convolve2d(convolve2d(z[..., i], filt[:, None]), filt[None, :])
        for i in range(z.shape[-1])], -1)
    mu0 = filt_fn(rgb)
    mu1 = filt_fn(gts)
    mu00 = mu0 * mu0
    mu11 = mu1 * mu1
    mu01 = mu0 * mu1
    sigma00 = filt_fn(rgb**2) - mu00
    sigma11 = filt_fn(gts**2) - mu11
    sigma01 = filt_fn(rgb * gts) - mu01

    # Clip the variances and covariances to valid values.
    # Variance must be non-negative:
    sigma00 = np.maximum(0., sigma00)
    sigma11 = np.maximum(0., sigma11)
    sigma01 = np.sign(sigma01) * np.minimum(
        np.sqrt(sigma00 * sigma11), np.abs(sigma01))
    c1 = (k1 * max_val)**2
    c2 = (k2 * max_val)**2
    numer = (2 * mu01 + c1) * (2 * sigma01 + c2)
    denom = (mu00 + mu11 + c1) * (sigma00 + sigma11 + c2)
    ssim_map = numer / denom
    return np.mean(ssim_map)
