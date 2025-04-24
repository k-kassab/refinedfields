import glob
import os
from typing import List, Optional

import torch
import torchvision.transforms.functional as TF
from PIL import Image

def write_png(path, data):
    """Writes an PNG image to some path.

    Args:
        path (str): Path to save the PNG.
        data (np.array): HWC image.

    Returns:
        (void): Writes to path.
    """
    Image.fromarray(data).save(path)


def read_png(file_name: str, resize_h: Optional[int] = None, resize_w: Optional[int] = None) -> torch.Tensor:
    """Reads a PNG image from path, potentially resizing it.
    """
    img = Image.open(file_name).convert('RGB')  # PIL outputs BGR by default
    if resize_h is not None and resize_w is not None:
        img.resize((resize_w, resize_h), Image.LANCZOS)
    img = TF.to_tensor(img)  # TF converts to C, H, W
    img = img.permute(1, 2, 0).contiguous()  # H, W, C
    return img


def glob_imgs(path, exts=None):
    """Utility to find images in some path.

    Args:
        path (str): Path to search images in.
        exts (list of str): List of extensions to try.

    Returns:
        (list of str): List of paths that were found.
    """
    if exts is None:
        exts = ['*.png', '*.PNG', '*.jpg', '*.jpeg', '*.JPG', '*.JPEG']
    imgs = []
    for ext in exts:
        imgs.extend(glob.glob(os.path.join(path, ext)))
    return imgs


