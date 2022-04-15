'''
Verify that the forward interleave projection are noise-independent.
According to the results, they are not independent so Noise2Noise is not applicable.
'''

# %%
import h5py
import matplotlib.pyplot as plt
import numpy as np
import cupy as cp
import os

import ct_projector.projector.numpy as ct_projector
import ct_projector.projector.numpy.fan_equiangluar as ct_fan

import ct_projector.projector.cupy as ct_projector_cp
import ct_projector.projector.cupy.fan_equiangular as ct_fan_cp
import ct_projector.recon.cupy as ct_recon_cp

from recon_fp.noise2noise.locations import input_data_dir

# %%
device = 0
img_norm = 0.019
islice = 80
dose_rate = 4
n0 = 2e5
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
# add noise the projection
rng = np.random.default_rng(0)
prj_ld = prj + np.sqrt((1 - 1 / dose_rate) * dose_rate * np.exp(prj * img_norm) / n0) \
    * rng.normal(size=prj.shape) / img_norm

# %%
# reconstruction
ct_projector.set_device(0)
projector = ct_projector.ct_projector()
projector.from_file(geometry)
projector.nz = 1
projector.nv = 1
angles = projector.get_angles()

fprj = ct_fan.ramp_filter(projector, prj)
ref = ct_fan.fbp_bp(projector, fprj, angles)

fprj = ct_fan.ramp_filter(projector, prj_ld)
img_ld = ct_fan.fbp_bp(projector, fprj, angles)

# %%
plt.figure(figsize=[18, 6])
plt.subplot(131)
plt.imshow(ref[0, 0], 'gray', vmin=0.84, vmax=1.24)
plt.subplot(132)
plt.imshow(img_ld[0, 0], 'gray', vmin=0.84, vmax=1.24)
plt.subplot(133)
plt.imshow(np.abs(ref - img_ld)[0, 0], 'gray', vmin=0, vmax=0.1)

# %%
# interleaved projection
even_inds = np.arange(0, len(angles), 2)
inds1 = even_inds + rng.integers(0, 2, len(even_inds))
inds1[inds1 > len(angles)] = len(angles)
inds2 = np.array([i for i in range(len(angles)) if i not in inds1])

fprj = ct_fan.ramp_filter(projector, np.copy(prj_ld[:, inds1, ...], 'C'))
recon_1 = ct_fan.fbp_bp(projector, fprj, angles[inds1])

fprj = ct_fan.ramp_filter(projector, np.copy(prj_ld[:, inds2, ...], 'C'))
recon_2 = ct_fan.fbp_bp(projector, fprj, angles[inds2])

# %%
plt.figure(figsize=[18, 6])
plt.subplot(131)
plt.imshow(recon_1[0, 0], 'gray', vmin=0.84, vmax=1.24)
plt.subplot(132)
plt.imshow(recon_2[0, 0], 'gray', vmin=0.84, vmax=1.24)
plt.subplot(133)
plt.imshow(np.abs(recon_1 - recon_2)[0, 0], 'gray', vmin=0, vmax=0.2)

# %%
# interleaved forward projection
fp = ct_fan.distance_driven_fp(projector, img_ld, angles)

fprj = ct_fan.ramp_filter(projector, np.copy(fp[:, inds1, ...], 'C'), 'rl')
recon_fp_1 = ct_fan.fbp_bp(projector, fprj, angles[inds1])

fprj = ct_fan.ramp_filter(projector, np.copy(fp[:, inds2, ...], 'C'), 'rl')
recon_fp_2 = ct_fan.fbp_bp(projector, fprj, angles[inds2])

# %%
plt.figure(figsize=[18, 6])
plt.subplot(131)
plt.imshow(recon_fp_1[0, 0], 'gray', vmin=0.84, vmax=1.24)
plt.subplot(132)
plt.imshow(recon_fp_2[0, 0], 'gray', vmin=0.84, vmax=1.24)
plt.subplot(133)
plt.imshow(np.abs(recon_fp_1 - recon_fp_2)[0, 0], 'gray', vmin=0, vmax=0.2)

# %%
fprj = ct_fan.ramp_filter(projector, fp, 'rl')
recon_fp = ct_fan.distance_driven_bp(projector, fprj, angles, True)

plt.imshow(np.abs(recon_fp - img_ld)[0, 0], 'gray', vmin=0, vmax=0.1)

# %%
cp.cuda.Device(device).use()
projector_cp = ct_projector_cp.ct_projector()
projector_cp.from_file(geometry)
projector_cp.nz = 1
projector_cp.nv = 1

fp_cp = cp.array(fp, order='C')
# recon_cp = cp.array(recon_fp, order='C')
# nesterov_cp = cp.array(recon_fp, order='C')
recon_cp = cp.zeros(recon_fp.shape, cp.float32)
nesterov_cp = cp.zeros(recon_fp.shape, cp.float32)
angles_cp = cp.array(angles, order='C')

projector_cp.set_projector(ct_fan_cp.distance_driven_fp, angles=angles_cp)
projector_cp.set_backprojector(ct_fan_cp.distance_driven_bp, angles=angles_cp)
norm_img_cp = projector_cp.calc_norm_img()

for i in range(1000):
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
plt.figure(figsize=[12, 6])
plt.subplot(121)
plt.imshow(recon_ir[0, 0], 'gray', vmin=0.84, vmax=1.24)
plt.subplot(122)
plt.imshow(np.abs(recon_ir - img_ld)[0, 0], 'gray', vmin=0, vmax=0.1)

# %%
fp_cp = cp.array(fp[:, inds1, ...], order='C')
recon_cp = cp.zeros(recon_fp.shape, cp.float32)
nesterov_cp = cp.zeros(recon_fp.shape, cp.float32)
angles_cp = cp.array(angles[inds1], order='C')

projector_cp.set_projector(ct_fan_cp.distance_driven_fp, angles=angles_cp)
projector_cp.set_backprojector(ct_fan_cp.distance_driven_bp, angles=angles_cp)
norm_img_cp = projector_cp.calc_norm_img()

for i in range(1000):
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
recon_ir_1 = recon_cp.get()

# %%
fp_cp = cp.array(fp[:, inds2, ...], order='C')
recon_cp = cp.zeros(recon_fp.shape, cp.float32)
nesterov_cp = cp.zeros(recon_fp.shape, cp.float32)
angles_cp = cp.array(angles[inds2], order='C')

projector_cp.set_projector(ct_fan_cp.distance_driven_fp, angles=angles_cp)
projector_cp.set_backprojector(ct_fan_cp.distance_driven_bp, angles=angles_cp)
norm_img_cp = projector_cp.calc_norm_img()

for i in range(1000):
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
recon_ir_2 = recon_cp.get()

# %%
plt.figure(figsize=[18, 6])
plt.subplot(131)
plt.imshow(recon_ir_1[0, 0], 'gray', vmin=0.84, vmax=1.24)
plt.subplot(132)
plt.imshow(recon_ir_2[0, 0], 'gray', vmin=0.84, vmax=1.24)
plt.subplot(133)
plt.imshow(np.abs(recon_ir_1 - recon_ir_2)[0, 0], 'gray', vmin=0, vmax=0.2)

# %%
