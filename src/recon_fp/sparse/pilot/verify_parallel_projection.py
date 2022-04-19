'''
Verify that the artifacts in forward projection of parallel recon.
'''

# %%
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
import cupy as cp

import ct_projector.projector.numpy as ct_projector
import ct_projector.projector.numpy.fan_equiangluar as ct_fan
import ct_projector.projector.numpy.parallel as ct_para

import ct_projector.projector.cupy as ct_projector_cp
import ct_projector.projector.cupy.parallel as ct_para_cp
import ct_projector.prior.cupy as ct_prior_cp
import ct_projector.recon.cupy as ct_recon_cp

from recon_fp.sparse.locations import input_data_dir

# %%
device = 0
img_norm = 0.019
islice = 80
margin = 96
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
def mask_fov(img, fov_size):
    xx, yy = np.meshgrid(np.arange(0, img.shape[2]), np.arange(0, img.shape[3]))
    cx = img.shape[2] / 2
    cy = img.shape[3] / 2
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    mask = np.where(dist < fov_size / 2, 1, 0)
    img = img * mask[np.newaxis, np.newaxis]

    return img.astype(np.float32)


# %%
# reconstruction
ct_projector.set_device(0)
projector_fan = ct_projector.ct_projector()
projector_fan.from_file(geometry)
projector_fan.nz = 1
projector_fan.nv = 1
angles = projector_fan.get_angles()
angles_sparse = np.copy(angles[::down_sample], 'C')

fprj = ct_fan.ramp_filter(projector_fan, prj, 'rl')
ref = ct_fan.fbp_bp(projector_fan, fprj, angles)

# %%
projector_para = ct_projector.ct_projector()
projector_para.from_file(geometry)
projector_para.nz = 1
projector_para.nv = 1
projector_para.du = projector_para.du * projector_para.dso / projector_para.dsd
fov_size = projector_para.du * projector_para.nu / projector_para.dx
ref = mask_fov(ref, fov_size)

prj_ref = ct_para.distance_driven_fp(projector_para, ref, angles)
fprj = ct_para.ramp_filter(projector_para, prj_ref, 'hann')
img_ref = ct_para.distance_driven_bp(projector_para, fprj, angles, True)

prj_sparse = ct_para.distance_driven_fp(projector_para, ref, angles_sparse)
fprj = ct_para.ramp_filter(projector_para, prj_sparse, 'hann')
img_sparse = ct_para.distance_driven_bp(projector_para, fprj, angles_sparse, True)

# %%
img_sparse = mask_fov(img_sparse, fov_size)

# %%
plt.figure(figsize=[18, 6])
plt.subplot(131)
plt.imshow(img_ref[0, 0, margin:-margin, margin:-margin], 'gray', vmin=0.84, vmax=1.24)
plt.subplot(132)
plt.imshow(img_sparse[0, 0, margin:-margin, margin:-margin], 'gray', vmin=0.84, vmax=1.24)
plt.subplot(133)
plt.imshow(np.abs(img_ref - img_sparse)[0, 0, margin:-margin, margin:-margin], 'gray', vmin=0, vmax=0.1)

# %%
# forward projection
fp = ct_para.distance_driven_fp(projector_para, img_sparse, angles_sparse)

fprj = ct_para.ramp_filter(projector_para, fp, 'rl')
recon_fp = ct_para.distance_driven_bp(projector_para, fprj, angles_sparse, True)

# %%
plt.figure(figsize=[12, 6])
plt.subplot(121)
plt.imshow(recon_fp[0, 0, margin:-margin, margin:-margin], 'gray', vmin=0.84, vmax=1.24)
plt.subplot(122)
plt.imshow(np.abs(recon_fp - img_sparse)[0, 0, margin:-margin, margin:-margin], 'gray', vmin=0, vmax=0.2)


# %%
def get_matrix_norm(
    projector: ct_projector.ct_projector,
    angles: np.array,
    fov_size: float,
    filter_type: str = 'rl',
    niter: int = 10,
    seed: int = 0
):
    np.random.seed(seed)
    x = np.random.uniform(size=[1, len(angles), projector.nv, projector.nu])
    x = x.astype(np.float32)
    x = x / np.linalg.norm(x)

    for i in range(niter):
        print(i, end=',', flush=True)
        fx = ct_para.ramp_filter(projector, x, filter_type)
        recon = ct_para.distance_driven_bp(projector, fx, angles, True)
        recon = mask_fov(recon, fov_size)
        norm = np.linalg.norm(recon)

        print(norm)

        fp = ct_para.distance_driven_fp(projector, recon, angles)
        x = ct_para.ramp_filter(projector, fp, filter_type)
        x = x / np.linalg.norm(x)
    print('')

    return norm


def weight_dd_fp_for_fbp(prjs: np.array, angles: np.array, dx: float, dy: float):
    '''
    Apply weight to the distance driven fp to make it the transpose of FBP backprojector.
    '''

    cos_angles = np.abs(np.cos(angles))
    sin_angles = np.abs(np.sin(angles))
    wy = dy / cos_angles
    wx = dx / sin_angles

    weights = np.ones(len(angles))
    weights[cos_angles > sin_angles] = wy[cos_angles > sin_angles]
    weights[cos_angles <= sin_angles] = wx[cos_angles <= sin_angles]

    weights = weights.astype(np.float32)
    prjs = prjs / weights[np.newaxis, :, np.newaxis, np.newaxis]

    return prjs.astype(np.float32)


# %%
# cp.cuda.Device(0).use()
# cufp = cp.array(fp, order='C')
# cuimg_sparse = cp.array(img_sparse, order='C')
# cuangles = cp.array(angles, order='C')

mat_norm = get_matrix_norm(projector_para, angles_sparse, fov_size)

# %%
# try basic iteration
# prj_recon = np.copy(prj_sparse)
prj_recon = np.copy(fp)
step_size = 0.75
alpha = 0.99
for i in range(1000):
    fprj = ct_para.ramp_filter(projector_para, prj_recon, 'rl')
    img_recon = ct_para.distance_driven_bp(projector_para, fprj, angles_sparse, True)
    img_recon = mask_fov(img_recon, fov_size)

    diff = img_recon - img_sparse

    fp_diff = ct_para.distance_driven_fp(projector_para, diff, angles_sparse)
    fp_diff = weight_dd_fp_for_fbp(fp_diff, angles_sparse, projector_para.dx, projector_para.dy)
    prj_diff = ct_para.ramp_filter(projector_para, fp_diff, 'rl')

    prj_recon = prj_recon - prj_diff * step_size / mat_norm
    step_size *= alpha

    if (i + 1) % 100 == 0:
        print(i + 1, np.sqrt(np.mean(diff[0, 0, margin:-margin, margin:-margin]**2)))

# %%
plt.figure(figsize=[16, 12])
plt.subplot(231)
plt.imshow(img_sparse[0, 0, margin:-margin, margin:-margin], 'gray', vmin=0.84, vmax=1.24)
plt.subplot(232)
plt.imshow(img_recon[0, 0, margin:-margin, margin:-margin], 'gray', vmin=0.84, vmax=1.24)
plt.subplot(233)
plt.imshow(np.abs(img_sparse - img_recon)[0, 0, margin:-margin, margin:-margin], 'gray', vmin=0, vmax=0.2)
plt.subplot(234)
plt.imshow(recon_fp[0, 0, margin:-margin, margin:-margin], 'gray', vmin=0.84, vmax=1.24)
plt.subplot(235)
plt.imshow((fp - prj_sparse)[0, :, 0, :], 'gray', vmin=-50, vmax=50, aspect='auto')
plt.subplot(236)
plt.imshow((prj_recon - prj_sparse)[0, :, 0, :], 'gray', vmin=-50, vmax=50, aspect='auto')


# %%
cp.cuda.Device(device).use()
projector_cp = ct_projector_cp.ct_projector()
projector_cp.from_file(geometry)
projector_cp.nz = 1
projector_cp.nv = 1
projector_cp.du = projector_para.du

fp_cp = cp.array(prj_recon, order='C')
recon_cp = cp.array(img_ref, order='C')
nesterov_cp = cp.array(img_ref, order='C')
angles_cp = cp.array(angles_sparse, order='C')

projector_cp.set_projector(ct_para_cp.distance_driven_fp, angles=angles_cp)
projector_cp.set_backprojector(ct_para_cp.distance_driven_bp, angles=angles_cp)
norm_img_cp = projector_cp.calc_norm_img()

for i in range(1000):
    if (i + 1) % 10 == 0:
        print(i + 1, end=',', flush=True)
    recon_cp, nesterov_cp = ct_recon_cp.nesterov_acceleration(
        ct_recon_cp.sqs_gaussian_one_step,
        recon_kwargs={
            'projector': projector_cp,
            'prj': fp_cp,
            'norm_img': norm_img_cp,
            'projector_norm': 1,
            'beta': 0
        },
        img=recon_cp,
        img_nesterov=nesterov_cp,
    )
print('')
recon_ir = recon_cp.get()

# %%
plt.figure(figsize=[12, 12])
plt.subplot(221)
plt.imshow(img_ref[0, 0, margin:-margin, margin:-margin], 'gray', vmin=0.84, vmax=1.24)
plt.subplot(222)
plt.imshow(img_sparse[0, 0, margin:-margin, margin:-margin], 'gray', vmin=0.84, vmax=1.24)
plt.subplot(223)
plt.imshow(recon_ir[0, 0, margin:-margin, margin:-margin], 'gray', vmin=0.84, vmax=1.24)

plt.subplot(224)
plt.imshow((recon_ir - img_ref)[0, 0, margin:-margin, margin:-margin], 'gray', vmin=-0.1, vmax=0.1)

# %%
# TV reconstruction
ntv_iter = 1000
nsubsets = 12
nesterov = 0.5
beta = 0.0001
zero_init = True

beta = 2e-4
prj_sparse_cp = cp.array(prj_sparse, order='C')

# beta = 1e-4
# prj_sparse_cp = cp.array(prj_recon, order='C')

# beta = 1e-3
# prj_sparse_cp = cp.array(fp, order='C')

projector_norm = projector_cp.calc_projector_norm()
norm_img_cp = projector_cp.calc_norm_img() / projector_norm / projector_norm

if zero_init:
    recon_cp = cp.zeros(img_sparse.shape, cp.float32)
    nesterov_cp = cp.zeros(img_sparse.shape, cp.float32)
else:
    recon_cp = cp.array(img_sparse, order='C')
    nesterov_cp = cp.array(img_sparse, order='C')

for i in range(ntv_iter):
    for isubset in range(nsubsets):
        inds = np.arange(isubset, len(angles_cp), nsubsets)
        angles_current = cp.copy(angles_cp[inds], 'C')
        prj_current = cp.copy(prj_sparse_cp[:, inds, ...], 'C')

        recon_cp = ct_recon_cp.sqs_one_step(
            projector_cp,
            nesterov_cp,
            prj_current,
            norm_img_cp,
            projector_norm,
            beta,
            ct_prior_cp.tv_sqs,
            {'weights': [1, 1, 1]},
            nsubsets,
            {'angles': angles_current},
            {'angles': angles_current},
            return_loss=False
        )

        nesterov_cp = recon_cp + nesterov * (recon_cp - nesterov_cp)

    _, data_loss, tv_loss = ct_recon_cp.sqs_one_step(
        projector_cp,
        recon_cp,
        prj_sparse_cp,
        norm_img_cp,
        projector_norm,
        beta,
        ct_prior_cp.tv_sqs,
        {'weights': [1, 1, 1]},
        return_loss=True
    )

    if (i + 1) % 100 == 0:
        print(i + 1, data_loss, tv_loss)

# %%
recon = recon_cp.get()
plt.figure(figsize=[16, 8])
plt.subplot(121)
plt.imshow(img_sparse[0, 0, margin:-margin, margin:-margin], 'gray', vmin=0.84, vmax=1.24)
plt.subplot(122)
plt.imshow(recon[0, 0, margin:-margin, margin:-margin], 'gray', vmin=0.84, vmax=1.24)
err_img = (recon - img_ref)
err_img = mask_fov(err_img, fov_size)
print(np.sqrt(np.mean(err_img**2)))

# %%
