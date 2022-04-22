'''
utility functions
'''

# %%
import numpy as np

import ct_projector.projector.numpy as ct_base


# %%
def get_fov_mask(fov_size, img_shape):
    xx, yy = np.meshgrid(np.arange(0, img_shape[-2]), np.arange(0, img_shape[-1]))
    cx = img_shape[-2] / 2
    cy = img_shape[-1] / 2
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    mask = np.where(dist < fov_size / 2, 1, 0)

    return mask


def mask_fov_para(projector: ct_base.ct_projector, img: np.array):
    fov_size = projector.du * projector.nu / projector.dx

    mask = get_fov_mask(fov_size, img.shape)
    img = img * mask[np.newaxis, np.newaxis]

    return img


def mask_fov_fan(projector: ct_base.ct_projector, img: np.array):
    fov_size = projector.du * projector.nu * projector.dso / projector.dsd / projector.dx

    mask = get_fov_mask(fov_size, img.shape)
    img = img * mask[np.newaxis, np.newaxis]

    return img
