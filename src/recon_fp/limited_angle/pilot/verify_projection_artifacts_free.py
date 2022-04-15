'''
Verify that the forward projection are artifacts free.
For FBP-based reconstruction the artifacts are strong.
May be iterative reconstruction will give better results.
'''

# %%
import h5py
import matplotlib.pyplot as plt
import numpy as np
import cupy as cp
import os
import scipy.ndimage

import ct_projector.projector.numpy as ct_projector
import ct_projector.projector.numpy.fan_equiangluar as ct_fan

import ct_projector.projector.cupy as ct_projector_cp
import ct_projector.projector.cupy.fan_equiangular as ct_fan_cp
import ct_projector.recon.cupy as ct_recon_cp

from recon_fp.noise2noise.locations import input_data_dir


# %%
def parker_weighting(
    projector: ct_projector.ct_projector,
    angles: np.array
) -> np.array:
    da = projector.du / projector.dsd
    gamma = (angles[-1] - np.pi) / 2
    # if gamma < delta:
    #     gamma = delta
    alphas = ((np.arange(projector.nu) - (projector.nu - 1) / 2) - projector.off_u) * da
    weights = []
    for beta in angles:
        w = np.zeros([projector.nu], np.float32)
        res1 = (np.sin(np.pi / 4 * beta / (gamma - alphas)))**2
        res2 = (np.sin(np.pi / 4 * (np.pi + 2 * gamma - beta) / (gamma + alphas)))**2
        i1 = np.where(beta < 2 * gamma - 2 * alphas)[0]
        i2 = np.where(beta <= np.pi - 2 * alphas)[0]
        w = res2
        w[i2] = 1
        w[i1] = res1[i1]
        weights.append(w)
    weights = np.array(weights)[np.newaxis, :, np.newaxis, :]

    return weights


def riess_weighting(
    projector: ct_projector.ct_projector,
    angles: np.array,
    sigma=30,
    theta=10 * np.pi / 180
) -> np.array:
    da = projector.du / projector.dsd
    gamma = (angles[-1] - np.pi) / 2
    # if gamma < delta:
    #     gamma = delta
    alphas = ((np.arange(projector.nu) - (projector.nu - 1) / 2) - projector.off_u) * da
    weights = []
    for beta in angles:
        w = np.ones([projector.nu], np.float32)
        res1 = (np.sin(np.pi / 4 * beta / (gamma - alphas)))**2
        res2 = (np.sin(np.pi / 4 * (np.pi + 2 * gamma - beta) / (gamma + alphas)))**2
        i1 = np.where(beta < 2 * gamma - 2 * alphas)[0]
        i2 = np.where(beta < -2 * gamma + 2 * alphas)[0]
        i3 = np.where(beta > np.pi - 2 * alphas)[0]
        i4 = np.where(beta > np.pi + 2 * gamma + 2 * alphas)[0]
        w[i4] = 2 - res2[i4]
        w[i3] = res2[i3]
        w[i2] = 2 - res1[i2]
        w[i1] = res1[i1]

        # gaussian smoothing
        wSmooth = scipy.ndimage.gaussian_filter1d(w, sigma, mode='nearest')
        ind = np.where(np.all((beta > theta, beta < np.pi + 2 * gamma - theta)))
        wSmooth[ind] = w[ind]

        weights.append(wSmooth)
    weights = np.array(weights)[np.newaxis, :, np.newaxis, :]

    return weights


# %%
device = 0
img_norm = 0.019
islice = 80
nangles = 0.33333
geometry = os.path.join(input_data_dir, 'lowdoseCTsets/geometry.cfg')
input_filename = os.path.join(input_data_dir, 'lowdoseCTsets/L291_full_sino.mat')

# %%
print('Reading projection...', flush=True)
with h5py.File(input_filename, 'r') as f:
    prjs = np.copy(f['sino']).transpose([1, 0, 2]).astype(np.float32) / img_norm
    prjs = np.copy(prjs, 'C')[np.newaxis]
print('Done')
prj = prjs[..., [islice], :]

# %%
# reconstruction
ct_projector.set_device(0)
projector = ct_projector.ct_projector()
projector.from_file(geometry)
projector.nz = 1
projector.nv = 1
angles = projector.get_angles()
angles_limited = np.copy(angles[:int(len(angles) * nangles)], 'C')

fprj = ct_fan.ramp_filter(projector, prj)
ref = ct_fan.fbp_bp(projector, fprj, angles)

# %%
# use forward projection to avoid the reduced dose
prj_ref = ct_fan.distance_driven_fp(projector, ref, angles)
fprj = ct_fan.ramp_filter(projector, prj_ref, 'rl')
img_ref = ct_fan.fbp_bp(projector, fprj, angles)

# %%
weights = riess_weighting(projector, angles_limited) * \
    (angles_limited[-1] - angles_limited[0]) / (projector.nview - 1) * projector.nview / np.pi

# %%
prj_limited = ct_fan.distance_driven_fp(projector, ref, angles_limited)
fprj = ct_fan.ramp_filter(projector, prj_limited * weights, 'rl')
img_limited = ct_fan.fbp_bp(projector, fprj, angles_limited)

# %%
plt.figure(figsize=[18, 6])
plt.subplot(131)
plt.imshow(img_ref[0, 0], 'gray', vmin=0.84, vmax=1.24)
plt.subplot(132)
plt.imshow(img_limited[0, 0], 'gray', vmin=0.84, vmax=1.24)
plt.subplot(133)
plt.imshow(np.abs(img_ref - img_limited)[0, 0], 'gray', vmin=0, vmax=0.1)

# %%
# interleaved forward projection
fp = ct_fan.distance_driven_fp(projector, img_limited, angles_limited)

fprj = ct_fan.ramp_filter(projector, fp, 'rl')
recon_fp = ct_fan.fbp_bp(projector, fprj, angles_limited)

# %%
plt.figure(figsize=[12, 6])
plt.subplot(121)
plt.imshow(recon_fp[0, 0], 'gray', vmin=0.84, vmax=1.24)
plt.subplot(122)
plt.imshow(np.abs(recon_fp - img_limited)[0, 0], 'gray', vmin=0, vmax=0.2)

# %%
cp.cuda.Device(device).use()
projector_cp = ct_projector_cp.ct_projector()
projector_cp.from_file(geometry)
projector_cp.nz = 1
projector_cp.nv = 1

fp_cp = cp.array(fp, order='C')
recon_cp = cp.array(img_ref, order='C')
nesterov_cp = cp.array(img_ref, order='C')
angles_cp = cp.array(angles_limited, order='C')

projector_cp.set_projector(ct_fan_cp.distance_driven_fp, angles=angles_cp)
projector_cp.set_backprojector(ct_fan_cp.distance_driven_bp, angles=angles_cp)
norm_img_cp = projector_cp.calc_norm_img()

for i in range(100):
    if (i + 1) % 10 == 0:
        print(i + 1, end=',', flush=True)
    recon_cp, nesterov_cp = ct_recon_cp.nesterov_acceleration(
        ct_recon_cp.sqs_gaussian_one_step,
        img=recon_cp,
        img_nesterov=nesterov_cp,
        projector=projector_cp,
        prj=fp_cp,
        norm_img=norm_img_cp,
        projector_norm=1,
        beta=0
    )
print('')
recon_ir = recon_cp.get()

# %%
plt.figure(figsize=[12, 12])
plt.subplot(221)
plt.imshow(img_ref[0, 0], 'gray', vmin=0.84, vmax=1.24)
plt.subplot(222)
plt.imshow(img_limited[0, 0], 'gray', vmin=0.84, vmax=1.24)
plt.subplot(223)
plt.imshow(recon_ir[0, 0], 'gray', vmin=0.84, vmax=1.24)
plt.subplot(224)
plt.imshow(np.abs(img_limited - img_ref)[0, 0], 'gray', vmin=-0.1, vmax=0.1)

# %%
