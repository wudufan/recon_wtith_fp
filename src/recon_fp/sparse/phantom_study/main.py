'''
Forbild phantom study
'''

# %%
import argparse
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import cupy as cp
import sys
import os
import SimpleITK as sitk
import imageio

from recon_fp.sparse.locations import working_dir

import ct_projector.projector.cupy as ct_base
import ct_projector.projector.cupy.parallel as ct_proj
import ct_projector.prior.cupy as ct_prior
import ct_projector.recon.cupy as ct_recon


# %%
def get_args(default_args=[]):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/forbild512.mat')
    parser.add_argument('--geometry', default='data/geometry_para.cfg')
    parser.add_argument('--nprj_sparse', type=int, default=120)
    parser.add_argument('--start_angle', type=float, default=0)

    parser.add_argument('--recover_niters', type=int, default=5000)

    parser.add_argument('--filter', default='rl')
    parser.add_argument('--beta_tv', type=float, default=5e-5)
    parser.add_argument('--nsubsets', type=int, default=15)
    parser.add_argument('--niters', type=int, default=1000)
    parser.add_argument('--noise_n0', type=float, default=-1)
    parser.add_argument('--noise_norm', type=float, default=0.019)

    if 'ipykernel' in sys.argv[0]:
        args = parser.parse_args(default_args)
        args.debug = True
    else:
        args = parser.parse_args()
        args.debug = False

    for k in vars(args):
        print(k, '=', getattr(args, k), flush=True)

    return args


# %%
def tv_reconstruction(
    projector: ct_base.ct_projector,
    cuimg0: cp.array,
    cuprjs: cp.array,
    cuangles: cp.array,
    beta: float,
    nsubsets: int,
    niters: int = 1000,
    nesterov: float = 0.5
):
    projector.set_projector(ct_proj.distance_driven_fp, angles=cuangles)
    projector.set_backprojector(ct_proj.distance_driven_bp, angles=cuangles, is_fbp=False)

    projector_norm = projector.calc_projector_norm()
    cunorm_img = projector.calc_norm_img() / projector_norm / projector_norm

    curecon = cp.copy(cuimg0, 'C')
    cunesterov = cp.copy(cuimg0, 'C')

    for i in range(niters):
        for isubset in range(nsubsets):
            # get subset
            inds = np.arange(isubset, len(cuangles), nsubsets)
            angles_current = cp.copy(cuangles[inds], 'C')
            prjs_current = cp.copy(cuprjs[:, inds, ...], 'C')

            curecon_new = ct_recon.sqs_one_step(
                projector,
                cunesterov,
                prjs_current,
                cunorm_img,
                projector_norm,
                beta,
                ct_prior.tv_sqs,
                {'weights': [1, 1, 1]},
                nsubsets,
                {'angles': angles_current},
                {'angles': angles_current},
                return_loss=False
            )

            cunesterov = curecon_new + nesterov * (curecon_new - curecon)
            curecon = curecon_new

        # calculate loss
        _, data_loss, tv_loss = ct_recon.sqs_one_step(
            projector,
            curecon,
            cuprjs,
            cunorm_img,
            projector_norm,
            beta,
            ct_prior.tv_sqs,
            {'weights': [1, 1, 1]},
            return_loss=True
        )

        if (i + 1) % 100 == 0:
            print(i + 1, data_loss, tv_loss)

    return curecon


# %%
def mask_fov(projector: ct_base.ct_projector, img: cp.array):
    fov_size = projector.du * projector.nu / projector.dx

    xx, yy = np.meshgrid(np.arange(0, img.shape[2]), np.arange(0, img.shape[3]))
    cx = img.shape[2] / 2
    cy = img.shape[3] / 2
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    mask = np.where(dist < fov_size / 2, 1, 0)
    mask = cp.array(mask, cp.float32)
    img = img * mask[np.newaxis, np.newaxis]

    return img.astype(cp.float32)


# %%
def fbp_from_forward_projection(
    projector: ct_base.ct_projector,
    img: cp.array,
    cuangles: cp.array
):
    cufp = ct_proj.distance_driven_fp(projector, img, cuangles)
    cufprj = ct_proj.ramp_filter(projector, cufp, 'rl')
    curecon = ct_proj.distance_driven_bp(projector, cufprj, cuangles, True)

    return cufp, curecon


# %%
def calc_fbp_matrix_norm(
    projector: ct_base.ct_projector,
    cuangles: cp.array,
    niter: int = 10,
    seed: int = 0
):
    cp.random.seed(seed)
    x = cp.random.uniform(size=[1, len(cuangles), projector.nv, projector.nu])
    x = x.astype(cp.float32)
    x = x / cp.linalg.norm(x)

    for i in range(niter):
        print(i, end=',', flush=True)
        fx = ct_proj.ramp_filter(projector, x, 'rl')
        recon = ct_proj.distance_driven_bp(projector, fx, cuangles, True)
        recon = mask_fov(projector, recon)
        norm = cp.linalg.norm(recon)

        fp = ct_proj.distance_driven_fp(projector, recon, cuangles)
        fp = weight_dd_fp_for_fbp(fp, cuangles, projector.dx, projector.dy)
        x = ct_proj.ramp_filter(projector, fp, 'rl')
        x = x / cp.linalg.norm(x)

        # print(
        #     cp.linalg.norm(fx),
        #     cp.linalg.norm(recon),
        #     cp.linalg.norm(fp),
        #     cp.linalg.norm(x),
        # )
    print('')

    return norm


def weight_dd_fp_for_fbp(cuprjs: cp.array, cuangles: cp.array, dx: float, dy: float):
    '''
    Apply weight to the distance driven fp to make it the transpose of FBP backprojector.
    '''

    cos_angles = cp.abs(cp.cos(cuangles))
    sin_angles = cp.abs(cp.sin(cuangles))
    wy = dy / cos_angles
    wx = dx / sin_angles

    weights = cp.ones(len(cuangles))
    weights[cos_angles > sin_angles] = wy[cos_angles > sin_angles]
    weights[cos_angles <= sin_angles] = wx[cos_angles <= sin_angles]

    weights = weights.astype(cp.float32)
    cuprjs = cuprjs / weights[cp.newaxis, :, cp.newaxis, cp.newaxis]

    return cuprjs.astype(cp.float32)


def recover_projection_from_img(
    projector: ct_base.ct_projector,
    cuprj0: cp.array,
    cuimg: cp.array,
    cuangles: cp.array,
    niters: int = 1000,
    stepsize: float = 1,
    alpha: float = 1,
    nesterov: float = 0.5
):
    mat_norm = calc_fbp_matrix_norm(projector, cuangles, niter=10)
    print(mat_norm)

    cuprj = cp.copy(cuprj0, 'C')
    cunesterov = cp.copy(cuprj0, 'C')

    for i in range(niters):
        cufprj = ct_proj.ramp_filter(projector, cunesterov, 'rl')
        curecon = ct_proj.distance_driven_bp(projector, cufprj, cuangles, True)
        curecon = mask_fov(projector, curecon)

        cudiff = curecon - cuimg

        cufp_diff = ct_proj.distance_driven_fp(projector, cudiff, cuangles)
        cufp_diff = weight_dd_fp_for_fbp(cufp_diff, cuangles, projector.dx, projector.dy)
        cuprj_diff = ct_proj.ramp_filter(projector, cufp_diff, 'rl')

        cuprj_new = cunesterov - cuprj_diff * stepsize / mat_norm / mat_norm
        stepsize *= alpha

        cunesterov = cuprj_new + nesterov * (cuprj_new - cuprj)
        cuprj = cuprj_new

        if (i + 1) % 100 == 0:
            print(i + 1, cp.sqrt(cp.mean(cudiff**2)))

    return cuprj


# %%
# cg algorithm
def calc_cg_Ax(
    projector: ct_base.ct_projector,
    cuprj: cp.array,
    cuangles: cp.array
):
    '''
    In CG algorithm, rewrite the original problem
    0.5*||Ax-b||^2_2 --> 0.5*x^TA^TAx - x^TA^Tb
    The CG algorithm works for
    0.5 * x^TAx - x^Tb
    Hence
    A^TA --> A
    A^Tb --> b
    '''
    cufprj = ct_proj.ramp_filter(projector, cuprj, 'rl')
    curecon = ct_proj.distance_driven_bp(projector, cufprj, cuangles, True)
    curecon = mask_fov(projector, curecon)
    cufp = ct_proj.distance_driven_fp(projector, curecon, cuangles)
    cufp = weight_dd_fp_for_fbp(cufp, cuangles, projector.dx, projector.dy)
    cures = ct_proj.ramp_filter(projector, cufp, 'rl')

    return cures


def calc_cg_b(
    projector: ct_base.ct_projector,
    cuimg: cp.array,
    cuangles: cp.array
):
    '''
    b in CG is A^Tb
    '''
    cufp = ct_proj.distance_driven_fp(projector, cuimg, cuangles)
    cufp = weight_dd_fp_for_fbp(cufp, cuangles, projector.dx, projector.dy)
    cures = ct_proj.ramp_filter(projector, cufp, 'rl')

    return cures


def recover_projection_from_img_cg(
    projector: ct_base.ct_projector,
    cuprj0: cp.array,
    cuimg: cp.array,
    cuangles: cp.array,
    niters: int = 1000,
    stepsize: float = 1,
    alpha: float = 1
):
    b = calc_cg_b(projector, cuimg, cuangles)
    x_k = cp.copy(cuprj0, 'C')
    ax = calc_cg_Ax(projector, x_k, cuangles)
    r_k = b - ax
    p_k = cp.copy(r_k, 'C')
    for i in range(niters):
        ap_k = calc_cg_Ax(projector, p_k, cuangles)
        alpha_k = cp.sum(r_k * r_k) / cp.sum(p_k * ap_k)
        x_k_1 = x_k + alpha_k * p_k
        r_k_1 = r_k - alpha_k * ap_k
        beta_k = cp.sum(r_k_1 * r_k_1) / cp.sum(r_k * r_k)
        p_k_1 = r_k_1 + beta_k * p_k

        x_k = cp.copy(x_k_1)
        r_k = cp.copy(r_k_1)
        p_k = cp.copy(p_k_1)

        # get data loss
        cufprj = ct_proj.ramp_filter(projector, x_k, 'rl')
        curecon = ct_proj.distance_driven_bp(projector, cufprj, cuangles, True)
        curecon = mask_fov(projector, curecon)
        cudiff = curecon - cuimg

        if (i + 1) % 100 == 0:
            print(i + 1, cp.linalg.norm(cudiff))

    return x_k


# %%
def save_results(output_dir: str, results: dict, args: argparse.Namespace, vmin=1.025, vmax=1.075):
    output_dir = os.path.join(
        output_dir,
        'sparse_{0}_n0_{1}_beta_{2}'.format(args.nprj_sparse, args.noise_n0, args.beta_tv)
    )
    os.makedirs(output_dir, exist_ok=True)

    # save the config
    with open(os.path.join(output_dir, 'config.cfg'), 'w') as f:
        f.write(__file__ + '\n')
        for k in vars(args):
            f.write('{0} = {1}\n'.format(k, getattr(args, k)))

    # save the results
    for name in results:
        # save the nii image
        img = results[name].get()[0]
        img = img * 1000 - 1000
        img = img.astype(np.int16)
        sitk_img = sitk.GetImageFromArray(img)
        sitk.WriteImage(sitk_img, os.path.join(output_dir, name + '.nii.gz'))

        # save the screen shot
        img = results[name].get()[0]
        img = img[img.shape[0] // 2]
        img = (img - vmin) / (vmax - vmin) * 255
        img[img < 0] = 0
        img[img > 255] = 255
        img = img.astype(np.uint8)
        imageio.imwrite(os.path.join(output_dir, name + '.png'), img)


# %%
def main(args):
    mat = scipy.io.loadmat(args.input)
    phantom = mat['ph']
    phantom = phantom[np.newaxis, np.newaxis]
    vmin = 1.025
    vmax = 1.075

    if args.debug:
        plt.figure(figsize=[8, 8])
        plt.imshow(phantom[0, 0], 'gray', vmin=1, vmax=1.1)
        plt.show()

    # read projector
    projector = ct_base.ct_projector()
    projector.from_file(args.geometry)
    angles_full = projector.get_angles()

    cuphantom = cp.array(phantom, order='C')
    cuangles_full = cp.array(angles_full, order='C')
    cuprj_full = ct_proj.distance_driven_fp(projector, cuphantom, cuangles_full)
    cufprj = ct_proj.ramp_filter(projector, cuprj_full, args.filter)
    curecon_full = ct_proj.distance_driven_bp(projector, cufprj, cuangles_full, True)
    recon_full = curecon_full.get()

    if args.debug:
        plt.figure(figsize=[8, 8])
        plt.imshow(recon_full[0, 0], 'gray', vmin=vmin, vmax=vmax)
        plt.show()

    projector.nview = args.nprj_sparse
    angles = projector.get_angles() + args.start_angle * np.pi / 180
    cuangles = cp.array(angles, order='C')
    cuprj = ct_proj.distance_driven_fp(projector, cuphantom, cuangles)

    # add noise
    cp.random.seed(0)
    if args.noise_n0 > 0:
        nphoton = args.noise_n0 * cp.exp(-cuprj * args.noise_norm)
        cuprj = -cp.log(cp.random.poisson(nphoton) / args.noise_n0) / args.noise_norm
        cuprj = cuprj.astype(cp.float32)

    cufprj = ct_proj.ramp_filter(projector, cuprj, args.filter)
    curecon = ct_proj.distance_driven_bp(projector, cufprj, cuangles, True)
    curecon = mask_fov(projector, curecon)
    recon = curecon.get()

    if args.debug:
        plt.figure(figsize=[8, 8])
        plt.imshow(recon[0, 0], 'gray', vmin=vmin, vmax=vmax)
        plt.show()

    # TV reconstruction
    curecon_tv = tv_reconstruction(
        projector,
        cp.zeros(curecon.shape, cp.float32),
        cuprj,
        cuangles,
        args.beta_tv,
        args.nsubsets,
        args.niters
    )
    rmse_tv = cp.sqrt(cp.mean((curecon_tv - cuphantom)**2))

    if args.debug:
        print('RMSE TV = {0}'.format(rmse_tv))
        plt.figure(figsize=[8, 8])
        plt.imshow(curecon_tv.get()[0, 0], 'gray', vmin=vmin, vmax=vmax)
        plt.show()

    # forward projection
    cufp, curecon_fp = fbp_from_forward_projection(projector, curecon, cuangles)
    curecon_fp = mask_fov(projector, curecon_fp)
    recon_fp = curecon_fp.get()
    if args.debug:
        plt.figure(figsize=[8, 8])
        plt.imshow(recon_fp[0, 0], 'gray', vmin=vmin, vmax=vmax)
        plt.show()

    # recover projection
    cuprj_recover = recover_projection_from_img(
        projector, cufp, curecon, cuangles, stepsize=1, niters=args.recover_niters, alpha=0.9999
    )
    # cuprj_recover = recover_projection_from_img_cg(
    #     projector, cufp, curecon, cuangles, 1000
    # )
    cufprj = ct_proj.ramp_filter(projector, cuprj_recover, 'rl')
    curecon_recover = ct_proj.distance_driven_bp(projector, cufprj, cuangles, True)
    curecon_recover = mask_fov(projector, curecon_recover)
    recon_recover = curecon_recover.get()
    if args.debug:
        plt.figure(figsize=[8, 8])
        plt.imshow(recon_recover[0, 0], 'gray', vmin=vmin, vmax=vmax)
        plt.show()

    fp_diff = (cufp - cuprj).get()
    recover_diff = (cuprj_recover - cuprj).get()
    if args.debug:
        plt.figure(figsize=[16, 8])
        plt.subplot(121)
        plt.imshow(fp_diff[0, :, 0, :], 'gray', aspect='auto', vmin=-10, vmax=10)
        plt.subplot(122)
        plt.imshow(recover_diff[0, :, 0, :], 'gray', aspect='auto', vmin=-10, vmax=10)
        plt.show()

        plt.figure(figsize=[16, 8])
        plt.subplot(121)
        plt.imshow(cuprj.get()[0, :, 0, :], 'gray', aspect='auto', vmin=0, vmax=400)
        plt.subplot(122)
        plt.imshow(cufp.get()[0, :, 0, :], 'gray', aspect='auto', vmin=0, vmax=400)
        plt.show()

    curecon_tv_recover = tv_reconstruction(
        projector,
        cp.zeros(curecon.shape, cp.float32),
        cuprj_recover,
        cuangles,
        args.beta_tv,
        args.nsubsets,
        args.niters
    )
    rmse_tv_recover = cp.sqrt(cp.mean((curecon_tv_recover - cuphantom)**2))

    curecon_tv_fp = tv_reconstruction(
        projector,
        cp.zeros(curecon.shape, cp.float32),
        cufp,
        cuangles,
        args.beta_tv,
        args.nsubsets,
        args.niters
    )
    rmse_tv_fp = cp.sqrt(cp.mean((curecon_tv_fp - cuphantom)**2))

    if args.debug:
        plt.figure(figsize=[18, 6])
        plt.subplot(131)
        plt.imshow(curecon_tv.get()[0, 0], 'gray', vmin=vmin, vmax=vmax)
        plt.title(str(rmse_tv))
        plt.subplot(132)
        plt.imshow(curecon_tv_recover.get()[0, 0], 'gray', vmin=vmin, vmax=vmax)
        plt.title(str(rmse_tv_recover))
        plt.subplot(133)
        plt.imshow(curecon_tv_fp.get()[0, 0], 'gray', vmin=vmin, vmax=vmax)
        plt.title(str(rmse_tv_fp))
        plt.show()

    # save results
    save_results(
        os.path.join(working_dir, 'sparse'),
        {
            'phantom': cuphantom,
            'full': curecon_full,
            'fbp': curecon,
            'fbp_fp': curecon_fp,
            'fbp_recover': curecon_recover,
            'tv': curecon_tv,
            'tv_fp': curecon_tv_fp,
            'tv_recover': curecon_tv_recover,
            'prj': cuprj.transpose([0, 2, 1, 3]),
            'prj_fp': cufp.transpose([0, 2, 1, 3]),
            'prj_recover': cuprj_recover.transpose([0, 2, 1, 3]),
        },
        args,
        vmin,
        vmax
    )
    save_results(
        os.path.join(working_dir, 'sparse'),
        {
            'prj_fp_diff': cp.abs(cufp - cuprj).transpose([0, 2, 1, 3]),
            'prj_recover_diff': cp.abs(cuprj_recover - cuprj).transpose([0, 2, 1, 3]),
        },
        args,
        0,
        10
    )


# %%
if __name__ == '__main__':
    args = get_args([
        '--noise_n0', '-1',
        '--beta_tv', '5e-4',
        '--niters', '1',
        '--recover_niters', '1',
    ])
    main(args)

# %%
