'''
Verify that the forward projection are artifacts free.
The projections are not artifacts free, but they can be used in a limited way for correction.
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
down_sample = 16
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
angles_sparse = np.copy(angles[::down_sample], 'C')

fprj = ct_fan.ramp_filter(projector, prj)
ref = ct_fan.fbp_bp(projector, fprj, angles)

# %%
# use forward projection to avoid the reduced dose
prj_ref = ct_fan.distance_driven_fp(projector, ref, angles)
fprj = ct_fan.ramp_filter(projector, prj_ref, 'rl')
img_ref = ct_fan.fbp_bp(projector, fprj, angles)

prj_sparse = ct_fan.distance_driven_fp(projector, ref, angles_sparse)
fprj = ct_fan.ramp_filter(projector, prj_sparse, 'rl')
img_sparse = ct_fan.fbp_bp(projector, fprj, angles_sparse)

# %%
plt.figure(figsize=[18, 6])
plt.subplot(131)
plt.imshow(img_ref[0, 0], 'gray', vmin=0.84, vmax=1.24)
plt.subplot(132)
plt.imshow(img_sparse[0, 0], 'gray', vmin=0.84, vmax=1.24)
plt.subplot(133)
plt.imshow(np.abs(img_ref - img_sparse)[0, 0], 'gray', vmin=0, vmax=0.1)

# %%
# interleaved forward projection
fp = ct_fan.distance_driven_fp(projector, img_sparse, angles_sparse)

fprj = ct_fan.ramp_filter(projector, fp, 'rl')
recon_fp = ct_fan.fbp_bp(projector, fprj, angles_sparse)

# %%
plt.figure(figsize=[12, 6])
plt.subplot(121)
plt.imshow(recon_fp[0, 0], 'gray', vmin=0.84, vmax=1.24)
plt.subplot(122)
plt.imshow(np.abs(recon_fp - img_sparse)[0, 0], 'gray', vmin=0, vmax=0.2)

# %%
cp.cuda.Device(device).use()
projector_cp = ct_projector_cp.ct_projector()
projector_cp.from_file(geometry)
projector_cp.nz = 1
projector_cp.nv = 1

fp_cp = cp.array(fp, order='C')
recon_cp = cp.array(img_ref, order='C')
nesterov_cp = cp.array(img_ref, order='C')
angles_cp = cp.array(angles_sparse, order='C')

projector_cp.set_projector(ct_fan_cp.distance_driven_fp, angles=angles_cp)
projector_cp.set_backprojector(ct_fan_cp.distance_driven_bp, angles=angles_cp)
norm_img_cp = projector_cp.calc_norm_img()

for i in range(20):
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
plt.imshow(img_ref[0, 0, 96:-96, 96:-96], 'gray', vmin=0.84, vmax=1.24)
plt.subplot(222)
plt.imshow(img_sparse[0, 0, 96:-96, 96:-96], 'gray', vmin=0.84, vmax=1.24)
plt.subplot(223)
plt.imshow(recon_ir[0, 0, 96:-96, 96:-96], 'gray', vmin=0.84, vmax=1.24)
plt.subplot(224)
plt.imshow(np.abs(img_sparse - img_ref)[0, 0, 96:-96, 96:-96], 'gray', vmin=-0.1, vmax=0.1)

# %%
